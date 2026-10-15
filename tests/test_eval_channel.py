"""
The host end of the conversation channel: the wire protocol, the relay, the tunnel.

Interfaces: thalamus_eval.channel — encode/decode, Relay, ScriptedPrompter, DriverThread,
run_cli; thalamus_eval.chain — SessionPlan.conversation; thalamus_eval.vm —
conversation_session and the session kinds job_spec accepts.
Infrastructure: a stand-in guest end on a unix socket (channel_fakes.FakeGuest) and real
local TCP servers; no VM, no model.
Scope: a frame the direction's table does not allow, an over-long line, a foreign version
or a malformed field closes the channel; the host is the only side that opens a tunnel,
only to a declared port, only on loopback; the relay carries a turn, a question and its
answer, the end of the session and the worker's usage; a vanished guest is an event.
"""

import asyncio
import base64
import json

import pytest

from thalamus_eval import channel, vm
from channel_fakes import FakeGuest, scripted_worker


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def frame(**kw):
    return channel.encode(kw)


# ------------------------------------------------------------------------------ protocol


class TestWire:
    def test_a_frame_is_one_line_with_version_and_type(self):
        line = channel.encode({"t": "end_session"})
        assert line.endswith(b"\n") and line.count(b"\n") == 1
        assert json.loads(line) == {"v": 1, "t": "end_session"}

    def test_each_direction_has_its_own_table(self):
        turn = channel.encode({"t": "user_turn", "id": "t1", "text": "hi"})
        assert channel.decode(turn, channel.HOST_TO_GUEST)["text"] == "hi"
        with pytest.raises(channel.ProtocolError, match="not allowed"):
            channel.decode(turn, channel.GUEST_TO_HOST)
        opened = channel.encode({"t": "tunnel_open", "sid": 1, "port": 8000})
        with pytest.raises(channel.ProtocolError, match="not allowed"):
            channel.decode(opened, channel.GUEST_TO_HOST)     # the guest never opens a stream

    @pytest.mark.parametrize("raw,why", [
        (b"not json\n", "not JSON"),
        (b"[1]\n", "JSON object"),
        (b'{"t":"ready"}\n', "version"),
        (b'{"v":2,"t":"ready"}\n', "version"),
        (b'{"v":1,"t":"mystery"}\n', "not allowed"),
        (b'{"v":1}\n', "not allowed"),
        (b'{"v":1,"t":"turn_result"}\n', "malformed"),
        (b'{"v":1,"t":"tunnel_data","sid":1,"data":"!!!"}\n', "malformed"),
        (b'{"v":1,"t":"tunnel_data","sid":true,"data":""}\n', "malformed"),
        (b'{"v":1,"t":"tunnel_close","sid":0}\n', "malformed"),
        (b'{"v":1,"t":"hello","role":"host"}\n', "malformed"),
    ])
    def test_a_frame_the_guest_may_not_send_is_refused(self, raw, why):
        with pytest.raises(channel.ProtocolError, match=why):
            channel.decode(raw, channel.GUEST_TO_HOST)

    def test_a_chunk_over_the_bound_is_refused(self):
        big = base64.b64encode(b"x" * (channel.MAX_CHUNK + 1)).decode()
        raw = json.dumps({"v": 1, "t": "tunnel_data", "sid": 1, "data": big}).encode() + b"\n"
        with pytest.raises(channel.ProtocolError, match="malformed"):
            channel.decode(raw, channel.GUEST_TO_HOST)
        ok = base64.b64encode(b"x" * channel.MAX_CHUNK).decode()
        channel.decode(json.dumps({"v": 1, "t": "tunnel_data", "sid": 1, "data": ok}).encode()
                       + b"\n", channel.GUEST_TO_HOST)

    def test_an_overlong_line_is_refused_both_ways(self):
        with pytest.raises(channel.ProtocolError, match="limit"):
            channel.encode({"t": "user_turn", "id": "t", "text": "x" * channel.MAX_LINE})
        with pytest.raises(channel.ProtocolError, match="exceeds"):
            channel.decode(b"x" * (channel.MAX_LINE + 1), channel.GUEST_TO_HOST)

    def test_a_turn_over_the_text_bound_is_refused_by_the_guest_table(self):
        raw = json.dumps({"v": 1, "t": "user_turn", "id": "t",
                          "text": "x" * (channel.MAX_TURN_TEXT + 1)}).encode() + b"\n"
        with pytest.raises(channel.ProtocolError, match="malformed"):
            channel.decode(raw, channel.HOST_TO_GUEST)


# -------------------------------------------------------------------------------- relay


@pytest.fixture
def sock(tmp_path):
    return tmp_path / "relay.sock"


class TestRelay:
    def test_a_scripted_conversation_runs_to_the_end(self, sock):
        async def go():
            guest = FakeGuest(sock, on_frame=scripted_worker(None))
            await guest.start()
            try:
                relay = await channel.Relay.connect(sock, wait_s=5, hello_s=5)
                async with relay:
                    out = await channel.ScriptedPrompter(["one", "two"]).run(relay)
            finally:
                await guest.stop()
            return guest, out
        guest, out = run(go())
        assert [f["text"] for f in guest.received if f["t"] == "user_turn"] == ["one", "two"]
        assert guest.received[-1]["t"] == "end_session"
        assert out["turns"] == 2 and out["ended"] is True
        assert out["context_tokens"] == 2000 and out["total_cost_usd"] == 0.02
        assert [e["t"] for e in out["events"]].count("assistant") == 2
        assert guest.violations == []

    def test_a_question_is_answered_by_the_prompter_and_the_turn_finishes(self, sock):
        ask = [{"question": "Which colour?", "options": [{"label": "red"}, {"label": "blue"}]}]

        async def go():
            guest = FakeGuest(sock, on_frame=scripted_worker(None, ask=ask))
            await guest.start()
            try:
                async with await channel.Relay.connect(sock, wait_s=5, hello_s=5) as relay:
                    out = await channel.ScriptedPrompter(
                        ["build it"], answers={"Which colour?": "blue"}).run(relay)
            finally:
                await guest.stop()
            return guest, out
        guest, out = run(go())
        answer = [f for f in guest.received if f["t"] == "answer"][0]
        assert answer == {"v": 1, "t": "answer", "qid": "q1",
                          "answers": {"Which colour?": "blue"}}
        assert out["turns"] == 1

    def test_connect_waits_for_the_socket_and_for_the_guest_hello(self, sock):
        async def go():
            guest = FakeGuest(sock, hello=False)

            async def late():
                await asyncio.sleep(0.5)
                await guest.start()
                await guest.connected.wait()
                await asyncio.sleep(0.5)
                await guest.send({"t": "hello", "role": "guest"})
            task = asyncio.create_task(late())
            relay = await channel.Relay.connect(sock, wait_s=5, hello_s=5)
            await relay.close()
            await task
            await guest.stop()
            return relay
        assert run(go()).guest_hello["role"] == "guest"

    def test_connect_gives_up_with_a_reason(self, sock):
        with pytest.raises(channel.ChannelClosed, match="did not appear"):
            run(channel.Relay.connect(sock, wait_s=0.3))

    def test_no_hello_is_a_closed_channel_not_a_hang(self, sock):
        async def go():
            guest = FakeGuest(sock, hello=False)
            await guest.start()
            try:
                await channel.Relay.connect(sock, wait_s=5, hello_s=0.5)
            finally:
                await guest.stop()
        with pytest.raises(channel.ChannelClosed, match="no hello"):
            run(go())

    def test_a_vanished_guest_is_an_event(self, sock):
        async def go():
            guest = FakeGuest(sock)
            await guest.start()
            relay = await channel.Relay.connect(sock, wait_s=5, hello_s=5)
            guest.writer.close()
            events = [e async for e in relay.events()]
            await relay.close()
            await guest.stop()
            return events
        events = run(go())
        assert events[-1]["t"] == "closed" and "closed" in events[-1]["error"]

    @pytest.mark.parametrize("garbage", [
        b"{not json}\n", b'{"v":1,"t":"tunnel_open","sid":1,"port":80}\n',
        b'{"v":9,"t":"ready"}\n', b"x" * (channel.MAX_LINE + 10) + b"\n"])
    def test_a_bad_frame_from_the_guest_closes_the_channel(self, sock, garbage):
        async def go():
            guest = FakeGuest(sock)
            await guest.start()
            relay = await channel.Relay.connect(sock, wait_s=5, hello_s=5)
            await guest.send_raw(garbage)
            events = [e async for e in relay.events()]
            with pytest.raises(channel.ChannelClosed):
                await relay.send_user_turn("anyone there")
            await guest.stop()
            return events, relay
        events, relay = run(go())
        assert events[-1]["t"] == "closed" and "protocol error" in events[-1]["error"]
        assert relay.error.startswith("protocol error")

    def test_the_hosts_own_frames_pass_the_guests_table(self, sock):
        async def go():
            guest = FakeGuest(sock)
            await guest.start()
            relay = await channel.Relay.connect(sock, wait_s=5, hello_s=5)
            await relay.send_user_turn("x")
            await relay.answer_question("q9", {"Q?": "a"})
            await relay.end_session("done")
            await asyncio.sleep(0.2)
            await relay.close()
            await guest.stop()
            return guest
        guest = run(go())
        assert guest.violations == []
        assert [f["t"] for f in guest.received] == ["hello", "user_turn", "answer", "end_session"]


# ------------------------------------------------------------------------------- tunnel


async def _http_server(body: bytes):
    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.0 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        await writer.drain()
        writer.close()
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _get(port: int) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"GET / HTTP/1.0\r\nHost: x\r\n\r\n")
    await writer.drain()
    data = await reader.read()
    writer.close()
    return data


class TestTunnel:
    def test_a_browser_request_reaches_the_guest_app_and_back(self, sock):
        body = b"<h1>plant log</h1>" * 5000          # larger than one chunk
        async def go():
            http, http_port = await _http_server(body)
            guest = FakeGuest(sock, app_ports={8000: http_port})
            await guest.start()
            try:
                async with await channel.Relay.connect(sock, wait_s=5, hello_s=5) as relay:
                    host, port = await relay.listen_tunnel(8000)
                    first = await _get(port)
                    second = await _get(port)           # a second stream on the same tunnel
            finally:
                await guest.stop()
                http.close()
            return host, first, second, guest
        host, first, second, guest = run(go())
        assert host == "127.0.0.1"
        assert first.endswith(body) and second.endswith(body) and b"200 OK" in first
        opens = [f for f in guest.received if f["t"] == "tunnel_open"]
        assert [f["port"] for f in opens] == [8000, 8000]
        assert len({f["sid"] for f in opens}) == 2
        assert guest.violations == []

    def test_a_port_the_guest_does_not_serve_closes_the_stream(self, sock):
        async def go():
            guest = FakeGuest(sock, app_ports={})
            await guest.start()
            try:
                async with await channel.Relay.connect(sock, wait_s=5, hello_s=5) as relay:
                    _, port = await relay.listen_tunnel(9999)
                    data = await _get(port)
            finally:
                await guest.stop()
            return data, guest
        data, guest = run(go())
        assert data == b"" and guest.refused_ports == [9999]

    def test_the_listener_is_on_loopback_only(self, sock):
        async def go():
            guest = FakeGuest(sock)
            await guest.start()
            try:
                async with await channel.Relay.connect(sock, wait_s=5, hello_s=5) as relay:
                    await relay.listen_tunnel(8000)
                    return [s.getsockname()[0] for srv in relay._servers for s in srv.sockets]
            finally:
                await guest.stop()
        assert run(go()) == ["127.0.0.1"]

    def test_data_for_a_stream_the_host_never_opened_is_dropped(self, sock):
        async def go():
            guest = FakeGuest(sock)
            await guest.start()
            relay = await channel.Relay.connect(sock, wait_s=5, hello_s=5)
            await guest.send({"t": "tunnel_opened", "sid": 77})
            await guest.send({"t": "tunnel_data", "sid": 77,
                              "data": base64.b64encode(b"x").decode()})
            await guest.send({"t": "tunnel_close", "sid": 77})
            await asyncio.sleep(0.3)
            dropped, closed = relay.dropped, relay.closed
            await relay.close()
            await guest.stop()
            return dropped, closed
        assert run(go()) == (3, False)

    def test_the_stream_count_is_bounded(self, sock):
        async def go():
            guest = FakeGuest(sock, app_ports={})
            await guest.start()
            relay = await channel.Relay.connect(sock, wait_s=5, hello_s=5)
            _, port = await relay.listen_tunnel(8000)
            holders = []
            for _ in range(channel.MAX_STREAMS):
                r, w = await asyncio.open_connection("127.0.0.1", port)
                holders.append(w)
                relay._streams[1000 + len(holders)] = channel._Stream(w)   # occupy slots
            r, w = await asyncio.open_connection("127.0.0.1", port)
            data = await asyncio.wait_for(r.read(), 5)
            await relay.close()
            await guest.stop()
            return data
        assert run(go()) == b""


# ------------------------------------------------------------- driver thread and the CLI


class TestDriverThread:
    def test_a_driver_runs_against_the_socket_while_the_caller_blocks(self, sock):
        # the guest lives in its own loop, as the backend's broker does
        import threading
        ready, stop = threading.Event(), threading.Event()
        holder = {}

        def guest_thread():
            async def main():
                holder["guest"] = FakeGuest(sock, on_frame=scripted_worker(None))
                await holder["guest"].start()
                ready.set()
                while not stop.is_set():
                    await asyncio.sleep(0.05)
                await holder["guest"].stop()
            asyncio.run(main())
        t = threading.Thread(target=guest_thread)
        t.start()
        ready.wait(5)
        try:
            thread = channel.DriverThread(
                sock, channel.ScriptedPrompter(["go"]).run, wait_s=5, hello_s=5)
            thread.start()
            report = thread.finish(grace_s=15)
        finally:
            stop.set()
            t.join()
        assert report["driver_error"] is None
        assert report["driver_result"]["turns"] == 1 and report["driver_result"]["ended"]

    def test_a_cell_that_never_started_is_not_waited_for(self, sock):
        thread = channel.DriverThread(sock, channel.ScriptedPrompter(["go"]).run,
                                      wait_s=60, hello_s=5)
        thread.start()
        report = thread.finish(grace_s=0.5)
        assert report["driver_result"] is None
        assert report["driver_error"]


class TestCli:
    def test_a_script_prints_the_events_and_exits_clean(self, sock, tmp_path, capsys):
        async def go():
            guest = FakeGuest(sock, on_frame=scripted_worker(None))
            await guest.start()
            try:
                return await channel.run_cli(
                    str(sock), tunnels=[], script={"turns": ["a"]}, wait_s=5, hello_s=5)
            finally:
                await guest.stop()
        assert run(go()) == 0
        lines = [json.loads(row) for row in capsys.readouterr().out.splitlines()]
        kinds = [row["t"] for row in lines]
        assert kinds[0] == "connected" and "turn_result" in kinds and kinds[-1] == "summary"

    def test_an_unreachable_socket_exits_2(self, sock, capsys):
        assert run(channel.run_cli(str(sock), tunnels=[], script=None,
                                   wait_s=0.2, hello_s=1)) == 2


# ------------------------------------------------------------------------- the seam


class TestSeam:
    def test_job_spec_accepts_a_conversation_session_with_a_socket(self, tmp_path):
        session = vm.conversation_session(tmp_path / "r.sock", app_ports=[8000],
                                          options={"model": "opus"})
        spec = vm.job_spec(worktree=tmp_path, home=tmp_path, prompt="p", argv=["claude"],
                           env={}, network_policy="isolated", timeout=60, session=session)
        assert spec["session"] == {"kind": "sdk-conversation",
                                   "relay_socket": str(tmp_path / "r.sock"),
                                   "app_ports": [8000], "options": {"model": "opus"}}

    def test_a_conversation_without_a_socket_is_refused(self, tmp_path):
        kw = dict(worktree=tmp_path, home=tmp_path, prompt="p", argv=["claude"], env={},
                  network_policy="isolated", timeout=60)
        with pytest.raises(vm.VmBackendError, match="relay_socket"):
            vm.job_spec(session={"kind": "sdk-conversation"}, **kw)

"""A stand-in for the guest end of the conversation channel, speaking the wire protocol.

It listens on a unix socket the way the backend's broker does, answers the relay's hello,
plays a script of frames, and forwards tunnel streams to real local TCP ports — the part
of the guest that is plain sockets. It judges what the relay sends by the host->guest
table, so a relay that sent a frame the guest end would refuse fails here.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from pathlib import Path

from thalamus_eval import channel


class FakeGuest:
    def __init__(self, path: Path, *, app_ports: dict[int, int] | None = None,
                 on_frame=None, hello: bool = True) -> None:
        self.path = Path(path)
        #: guest app port -> real local port it forwards to
        self.app_ports = dict(app_ports or {})
        self.on_frame = on_frame
        self.hello = hello
        self.received: list[dict] = []
        self.refused_ports: list[int] = []
        self.violations: list[str] = []
        self.writer: asyncio.StreamWriter | None = None
        self._server: asyncio.Server | None = None
        self._streams: dict[int, asyncio.StreamWriter] = {}
        self.connected = asyncio.Event()

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(
            self._serve, str(self.path), limit=channel.MAX_LINE + 1)

    async def stop(self) -> None:
        if self.writer is not None:
            with contextlib.suppress(OSError):
                self.writer.close()
        for writer in self._streams.values():
            writer.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def send(self, frame: dict) -> None:
        self.writer.write(channel.encode(frame))
        await self.writer.drain()

    async def send_raw(self, data: bytes) -> None:
        self.writer.write(data)
        await self.writer.drain()

    async def _serve(self, reader, writer) -> None:
        self.writer = writer
        self.connected.set()
        if self.hello:
            await self.send({"t": "hello", "role": "guest", "runner": "fake",
                             "app_ports": sorted(self.app_ports)})
        try:
            while True:
                try:
                    line = await reader.readuntil(b"\n")
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
                try:
                    frame = channel.decode(line, channel.HOST_TO_GUEST)
                except channel.ProtocolError as exc:
                    self.violations.append(str(exc))
                    continue
                self.received.append(frame)
                await self._tunnel(frame)
                if self.on_frame is not None:
                    await self.on_frame(self, frame)
        finally:
            writer.close()

    async def _tunnel(self, frame: dict) -> None:
        kind = frame["t"]
        if kind == "tunnel_open":
            sid, port = frame["sid"], frame["port"]
            if port not in self.app_ports:
                self.refused_ports.append(port)
                await self.send({"t": "tunnel_close", "sid": sid, "reason": "undeclared"})
                return
            try:
                r, w = await asyncio.open_connection("127.0.0.1", self.app_ports[port])
            except OSError:
                await self.send({"t": "tunnel_close", "sid": sid, "reason": "refused"})
                return
            self._streams[sid] = w
            asyncio.create_task(self._pump(sid, r))
            await self.send({"t": "tunnel_opened", "sid": sid})
        elif kind == "tunnel_data":
            w = self._streams.get(frame["sid"])
            if w is not None:
                w.write(base64.b64decode(frame["data"]))
                await w.drain()
        elif kind == "tunnel_close":
            w = self._streams.pop(frame["sid"], None)
            if w is not None:
                w.close()

    async def _pump(self, sid: int, reader) -> None:
        while True:
            chunk = await reader.read(channel.MAX_CHUNK)
            if not chunk:
                break
            await self.send({"t": "tunnel_data", "sid": sid,
                             "data": base64.b64encode(chunk).decode()})
        if self._streams.pop(sid, None) is not None:
            await self.send({"t": "tunnel_close", "sid": sid, "reason": "eof"})


def scripted_worker(turn_frames, *, ask=None):
    """An `on_frame` for a worker: one `turn_result` per user_turn, optionally a question first.

    `ask` is (questions) to ask on the first turn; the turn completes after the answer.
    """
    state = {"turn": 0, "asked": False}

    async def on_frame(guest: FakeGuest, frame: dict) -> None:
        kind = frame["t"]
        if kind == "user_turn":
            state["turn"] += 1
            n = state["turn"]
            if ask and not state["asked"]:
                state["asked"] = True
                await guest.send({"t": "question", "qid": "q1", "turn": n, "questions": ask})
                return
            await _finish_turn(guest, n, turn_frames)
        elif kind == "answer":
            await _finish_turn(guest, state["turn"], turn_frames)
        elif kind == "end_session":
            await guest.send({"t": "ended", "reason": frame.get("reason", "host"),
                              "turns": state["turn"], "total_cost_usd": 0.01 * state["turn"]})
    return on_frame


async def _finish_turn(guest: FakeGuest, n: int, turn_frames) -> None:
    await guest.send({"t": "assistant", "turn": n, "text": f"working on turn {n}",
                      "tool_uses": [], "usage": {"input_tokens": 10}, "context_tokens": 1000 * n})
    await guest.send({"t": "turn_result", "turn": n, "subtype": "success", "is_error": False,
                      "total_cost_usd": 0.01 * n, "turn_cost_usd": 0.01,
                      "context_tokens": 1000 * n, "usage": {"output_tokens": 5 * n},
                      "turn_usage": {"output_tokens": 5}})

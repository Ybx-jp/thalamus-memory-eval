"""The host end of a campaign cell's conversation channel.

A campaign session is a conversation: a prompter on the host sends the worker (Claude
Code, inside a cell) one turn at a time, answers the questions the worker asks through
`AskUserQuestion`, and may drive the worker's app from a browser. No host port reaches a
cell, so all of it rides one virtio-serial byte channel the backend declares. The
backend listens for the cell and exposes a unix socket (`session.relay_socket` in the job,
`vm.job_spec`); this module is what connects to it.

    async with await Relay.connect(socket_path) as relay:
        await relay.send_user_turn("build the app")
        async for event in relay.events():
            ...                       # assistant, question, turn_result, ended ...
        # a question arrives as {"t": "question", "qid": ..., "questions": [...]}
        await relay.answer_question(qid, {question_text: "free text"})
        host, port = await relay.listen_tunnel(app_port=8000)   # 127.0.0.1:<port>
        await relay.end_session()

THE WIRE (version 1). Newline-delimited JSON, one object per line, every object carrying
`v` and `t`. A line is at most `MAX_LINE` bytes. The two directions have separate type
tables and a frame is judged by who may send it; an unknown type, a foreign version, a
non-object or an over-long line is a protocol error and closes the channel. The
in-guest end is `ops/cell-producer/sdk_conversation.py`, and the two tables are kept
equal by a test that loads both.

    host -> guest   hello, user_turn {id, text}, answer {qid, answers}, end_session,
                    tunnel_open {sid, port}, tunnel_data {sid, data}, tunnel_close {sid}
    guest -> host   hello, ready, assistant, tool_result, question {qid, questions},
                    turn_result, notice, ended, error,
                    tunnel_opened {sid}, tunnel_data, tunnel_close {sid, reason}

`assistant` and `turn_result` carry usage and `context_tokens` (the size of the context
the message was generated from); `turn_result` also carries `total_cost_usd`,
`turn_cost_usd`, `turn_usage` and `model_usage`. Tunnel data is base64 in chunks of at
most `MAX_CHUNK` bytes.

THE HOST IS THE ONLY SIDE THAT OPENS A TUNNEL, and only to a port the job declared in
`app_ports`. The local listener binds host loopback and nothing else. Everything that
arrives from the guest is untrusted: sizes are bounded, tunnel frames for a stream this
side did not open are dropped, and a malformed frame ends the channel rather than being
skipped.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

PROTOCOL_VERSION = 1
MAX_LINE = 256 * 1024
MAX_CHUNK = 32 * 1024
MAX_STREAMS = 16
MAX_TURN_TEXT = 100_000
MAX_ID = 64

#: What each side may send. Field values are checked by `_check_field`; extra fields
#: are allowed so a minor addition does not need a version bump.
HOST_TO_GUEST = {
    "hello": {"role": "host"},
    "user_turn": {"id": "id", "text": "turn_text"},
    "answer": {"qid": "id", "answers": "answers"},
    "end_session": {},
    "tunnel_open": {"sid": "sid", "port": "port"},
    "tunnel_data": {"sid": "sid", "data": "b64"},
    "tunnel_close": {"sid": "sid"},
}
GUEST_TO_HOST = {
    "hello": {"role": "guest"},
    "ready": {},
    "assistant": {"turn": "int"},
    "tool_result": {"turn": "int"},
    "question": {"qid": "id", "questions": "list"},
    "turn_result": {"turn": "int"},
    "notice": {"kind": "str"},
    "ended": {"reason": "str"},
    "error": {"message": "str"},
    "tunnel_opened": {"sid": "sid"},
    "tunnel_data": {"sid": "sid", "data": "b64"},
    "tunnel_close": {"sid": "sid"},
}

#: Frames the relay consumes itself rather than yielding to a driver.
INTERNAL = {"hello", "tunnel_opened", "tunnel_data", "tunnel_close"}


class ProtocolError(ValueError):
    """A frame the protocol does not allow. The channel is closed on it."""


class ChannelClosed(RuntimeError):
    """The relay is closed, or the guest end went away."""


def _check_field(kind: str, value: object) -> bool:
    if kind == "id":
        return isinstance(value, str) and 0 < len(value) <= MAX_ID
    if kind == "str":
        return isinstance(value, str)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "list":
        return isinstance(value, list)
    if kind == "turn_text":
        return isinstance(value, str) and len(value) <= MAX_TURN_TEXT
    if kind == "answers":
        return (isinstance(value, dict)
                and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()))
    if kind == "sid":
        return isinstance(value, int) and not isinstance(value, bool) and 0 < value < 2 ** 31
    if kind == "port":
        return isinstance(value, int) and not isinstance(value, bool) and 0 < value < 65536
    if kind == "b64":
        if not isinstance(value, str) or len(value) > (MAX_CHUNK * 4) // 3 + 8:
            return False
        try:
            return len(base64.b64decode(value, validate=True)) <= MAX_CHUNK
        except (binascii.Error, ValueError):
            return False
    return value == kind


def encode(frame: dict) -> bytes:
    frame = {"v": PROTOCOL_VERSION, **frame}
    line = json.dumps(frame, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
    if len(line) > MAX_LINE:
        raise ProtocolError(f"frame {frame.get('t')!r} is {len(line)} bytes; the limit is {MAX_LINE}")
    return line


def decode(line: bytes, table: dict) -> dict:
    if len(line) > MAX_LINE:
        raise ProtocolError(f"line of {len(line)} bytes exceeds {MAX_LINE}")
    try:
        frame = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"not JSON: {exc}") from exc
    if not isinstance(frame, dict):
        raise ProtocolError("a frame is a JSON object")
    if frame.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version {frame.get('v')!r}; this side speaks {PROTOCOL_VERSION}")
    kind = frame.get("t")
    if not isinstance(kind, str) or kind not in table:
        raise ProtocolError(f"frame type {kind!r} is not allowed in this direction")
    for name, check in table[kind].items():
        if name not in frame or not _check_field(check, frame[name]):
            raise ProtocolError(f"{kind}: field {name!r} is missing or malformed")
    return frame


# ------------------------------------------------------------------------------ relay


@dataclass
class _Stream:
    writer: asyncio.StreamWriter
    opened: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False


class Relay:
    """One connection to a cell's conversation channel."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer
        self._events: asyncio.Queue = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._turn_count = 0
        self._next_sid = 0
        self._streams: dict[int, _Stream] = {}
        self._servers: list[asyncio.Server] = []
        self._tasks: set[asyncio.Task] = set()
        self._reader_task: asyncio.Task | None = None
        self.guest_hello: dict | None = None
        self._hello = asyncio.Event()
        self.closed = False
        self.error: str | None = None
        #: Frames that failed validation or were dropped, for a record of the cell.
        self.dropped = 0

    # -- connecting --------------------------------------------------------------------
    @classmethod
    async def connect(cls, socket_path: str | Path, *, wait_s: float = 600.0,
                      hello_s: float = 900.0) -> "Relay":
        """Connect to the relay socket, waiting for the backend to create it.

        `wait_s` bounds the wait for the socket file (the backend creates it when the
        cell starts); `hello_s` bounds the wait for the guest's hello, which arrives
        when the guest has booted and its runner has started.
        """
        deadline = time.monotonic() + wait_s
        path = str(socket_path)
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(path, limit=MAX_LINE + 1)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise ChannelClosed(f"relay socket {path} did not appear in {wait_s:.0f}s")
                await asyncio.sleep(0.2)
        relay = cls(reader, writer)
        relay._reader_task = asyncio.create_task(relay._read_loop())
        await relay._send({"t": "hello", "role": "host"})
        try:
            await asyncio.wait_for(relay._hello.wait(), hello_s)
        except asyncio.TimeoutError:
            await relay.close()
            raise ChannelClosed(f"the guest sent no hello within {hello_s:.0f}s") from None
        if relay.error:
            await relay.close()
            raise ChannelClosed(relay.error)
        return relay

    async def __aenter__(self) -> "Relay":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    # -- conversation ------------------------------------------------------------------
    async def send_user_turn(self, text: str, *, turn_id: str | None = None) -> str:
        self._turn_count += 1
        turn_id = turn_id or f"t{self._turn_count}"
        await self._send({"t": "user_turn", "id": turn_id, "text": text})
        return turn_id

    async def answer_question(self, qid: str, answers: dict[str, str]) -> None:
        await self._send({"t": "answer", "qid": qid, "answers": dict(answers)})

    async def end_session(self, reason: str = "host") -> None:
        await self._send({"t": "end_session", "reason": reason})

    async def events(self) -> AsyncIterator[dict]:
        """The worker's frames in order, ending after `ended` (or the channel's end).

        A final `{"t": "closed", "error": ...}` is yielded when the channel ended
        without an `ended` frame, so a driver sees a vanished guest as an event rather
        than as silence.
        """
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event
            if event.get("t") in ("ended", "closed"):
                return

    # -- tunnel ------------------------------------------------------------------------
    async def listen_tunnel(self, app_port: int, host_port: int = 0) -> tuple[str, int]:
        """Listen on host loopback and tunnel each connection to the guest's `app_port`."""
        if not 0 < app_port < 65536:
            raise ValueError("app_port is a TCP port")
        server = await asyncio.start_server(
            lambda r, w: self._serve_local(r, w, app_port), "127.0.0.1", host_port)
        self._servers.append(server)
        host, port = server.sockets[0].getsockname()[:2]
        return host, port

    async def _serve_local(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                           app_port: int) -> None:
        if self.closed or len(self._streams) >= MAX_STREAMS:
            writer.close()
            return
        self._next_sid += 1
        sid = self._next_sid
        stream = _Stream(writer)
        self._streams[sid] = stream
        try:
            await self._send({"t": "tunnel_open", "sid": sid, "port": app_port})
            try:
                await asyncio.wait_for(stream.opened.wait(), 15)
            except asyncio.TimeoutError:
                return
            if stream.closed:
                return
            while True:
                chunk = await reader.read(MAX_CHUNK)
                if not chunk:
                    break
                if stream.closed:
                    return
                await self._send({"t": "tunnel_data", "sid": sid,
                                  "data": base64.b64encode(chunk).decode()})
        except (ChannelClosed, OSError, ConnectionError):
            pass
        finally:
            if not stream.closed:
                stream.closed = True
                with contextlib.suppress(ChannelClosed, OSError):
                    await self._send({"t": "tunnel_close", "sid": sid})
            self._streams.pop(sid, None)
            with contextlib.suppress(OSError):
                writer.close()

    # -- plumbing ----------------------------------------------------------------------
    async def _send(self, frame: dict) -> None:
        if self.closed:
            raise ChannelClosed(self.error or "the relay is closed")
        data = encode(frame)
        try:
            async with self._write_lock:
                self._writer.write(data)
                await self._writer.drain()
        except (OSError, ConnectionError) as exc:
            await self._finish(f"write failed: {exc}")
            raise ChannelClosed(str(exc)) from exc

    async def _read_loop(self) -> None:
        reason = "the guest end closed"
        try:
            while True:
                try:
                    line = await self._reader.readuntil(b"\n")
                except asyncio.IncompleteReadError:
                    break
                except asyncio.LimitOverrunError:
                    raise ProtocolError(f"a line longer than {MAX_LINE} bytes") from None
                if not line.strip():
                    continue
                frame = decode(line, GUEST_TO_HOST)
                await self._dispatch(frame)
        except ProtocolError as exc:
            reason = f"protocol error: {exc}"
            self.error = reason
        except (OSError, ConnectionError) as exc:
            reason = f"read failed: {exc}"
        await self._finish(reason)

    async def _dispatch(self, frame: dict) -> None:
        kind = frame["t"]
        if kind == "hello":
            self.guest_hello = frame
            self._hello.set()
        elif kind == "tunnel_opened":
            stream = self._streams.get(frame["sid"])
            if stream is None or stream.opened.is_set():
                self.dropped += 1          # opened a stream this side never asked for
            else:
                stream.opened.set()
        elif kind == "tunnel_data":
            stream = self._streams.get(frame["sid"])
            if stream is None or stream.closed or not stream.opened.is_set():
                self.dropped += 1
                return
            try:
                stream.writer.write(base64.b64decode(frame["data"]))
                await stream.writer.drain()
            except (OSError, ConnectionError):
                stream.closed = True
        elif kind == "tunnel_close":
            stream = self._streams.get(frame["sid"])
            if stream is None:
                self.dropped += 1
                return
            stream.closed = True
            stream.opened.set()
            with contextlib.suppress(OSError):
                stream.writer.close()
        else:
            await self._events.put(frame)

    async def _finish(self, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        if self.error is None:
            self.error = reason
        self._hello.set()
        await self._events.put({"t": "closed", "error": reason})
        await self._events.put(None)
        for stream in list(self._streams.values()):
            stream.closed = True
            with contextlib.suppress(OSError):
                stream.writer.close()
        for server in self._servers:
            server.close()
        with contextlib.suppress(OSError):
            self._writer.close()

    async def close(self) -> None:
        await self._finish("closed by the host")
        task = self._reader_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        for server in self._servers:
            with contextlib.suppress(Exception):
                await server.wait_closed()


# ------------------------------------------------------------------- the scripted prompter


AnswerPolicy = dict[str, str] | str | Callable[[list], dict[str, str]]


@dataclass
class ScriptedPrompter:
    """A prompter that follows a script, for tests and for dry runs of a campaign.

    `turns` are sent one at a time, each after the previous turn's `turn_result`;
    questions are answered by `answers` (a dict from question text to answer, a single
    string used for every question, or a callable taking the question list), and the
    session is ended after the last turn. The run returns every event it saw.
    """

    turns: list[str]
    answers: AnswerPolicy = "go with the first option"
    #: Called with each event as it arrives (a driver's logging hook).
    on_event: Callable[[dict], Awaitable[None] | None] | None = None
    #: Sent after the last turn's result. False leaves the session open for the caller.
    end_when_done: bool = True

    def answer_for(self, questions: list) -> dict[str, str]:
        texts = [q.get("question", "") for q in questions if isinstance(q, dict)]
        if callable(self.answers):
            return self.answers(questions)
        if isinstance(self.answers, str):
            return {text: self.answers for text in texts}
        return {text: self.answers.get(text, next(iter(self.answers.values()), ""))
                for text in texts}

    async def run(self, relay: Relay) -> dict:
        events: list[dict] = []
        pending = list(self.turns)
        if pending:
            await relay.send_user_turn(pending.pop(0))
        async for event in relay.events():
            events.append(event)
            if self.on_event is not None:
                outcome = self.on_event(event)
                if asyncio.iscoroutine(outcome):
                    await outcome
            kind = event["t"]
            if kind == "question":
                await relay.answer_question(event["qid"], self.answer_for(event["questions"]))
            elif kind == "turn_result":
                if pending:
                    await relay.send_user_turn(pending.pop(0))
                elif self.end_when_done:
                    await relay.end_session("script finished")
        results = [e for e in events if e["t"] == "turn_result"]
        return {"events": events, "turns": len(results),
                "total_cost_usd": results[-1].get("total_cost_usd") if results else 0.0,
                "context_tokens": results[-1].get("context_tokens") if results else 0,
                "ended": any(e["t"] == "ended" for e in events)}


# ----------------------------------------------------------------------- driver thread


class DriverThread:
    """Runs a prompter against a cell's relay socket while the backend blocks the caller.

    `vm.run_backend` is a blocking subprocess call, and the conversation has to happen
    during it. This runs an event loop in a thread: it connects when the backend
    creates the socket, opens the requested tunnel listeners, hands the relay to
    `driver`, and keeps the outcome for `finish`. If the cell never starts (a refusal
    before boot) `finish` cancels the wait.
    """

    def __init__(self, socket_path: str | Path,
                 driver: Callable[[Relay], Awaitable[object]], *,
                 tunnels: list[tuple[int, int]] | None = None,
                 wait_s: float = 600.0, hello_s: float = 900.0) -> None:
        self.socket_path = Path(socket_path)
        self.driver = driver
        self.tunnels = list(tunnels or [])
        self.wait_s = wait_s
        self.hello_s = hello_s
        self.result: object = None
        self.error: str | None = None
        self.listening: list[dict] = []
        self.relay_error: str | None = None
        self.dropped = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._thread = None

    def start(self) -> None:
        import threading
        self._thread = threading.Thread(target=self._main, name="conversation-driver",
                                        daemon=True)
        self._thread.start()

    def _main(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        try:
            relay = await Relay.connect(self.socket_path, wait_s=self.wait_s,
                                        hello_s=self.hello_s)
        except ChannelClosed as exc:
            self.error = str(exc)
            return
        except asyncio.CancelledError:
            self.error = "cancelled before the cell started"
            return
        try:
            for app_port, host_port in self.tunnels:
                host, port = await relay.listen_tunnel(app_port, host_port)
                self.listening.append({"app_port": app_port, "host": host, "port": port})
            self.result = await self.driver(relay)
        except asyncio.CancelledError:
            self.error = "cancelled"
        except Exception as exc:                  # noqa: BLE001 — the driver is caller code
            self.error = f"{exc.__class__.__name__}: {exc}"
        finally:
            self.relay_error = relay.error
            self.dropped = relay.dropped
            await relay.close()

    def finish(self, *, grace_s: float = 30.0) -> dict:
        """Wait for the driver to wind down (cancelling it after `grace_s`) and report."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout=grace_s)
            if thread.is_alive() and self._loop is not None and self._task is not None:
                self._loop.call_soon_threadsafe(self._task.cancel)
                thread.join(timeout=10)
        return {"driver_result": self.result, "driver_error": self.error,
                "relay_error": self.relay_error, "tunnels": self.listening,
                "frames_dropped": self.dropped}


# ----------------------------------------------------------------------------------- CLI


async def _stdin_lines() -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        yield line


def _emit(obj: dict) -> None:
    print(json.dumps(obj, separators=(",", ":")), flush=True)


async def run_cli(socket_path: str, *, tunnels: list[tuple[int, int]], script: dict | None,
                  wait_s: float, hello_s: float) -> int:
    """`thalamus-eval relay`: events as JSON lines on stdout, commands on stdin.

    Commands (one JSON object per line): `{"cmd": "turn", "text": ...}`,
    `{"cmd": "answer", "qid": ..., "answers": {...}}`, `{"cmd": "end"}`. With a
    script (`{"turns": [...], "answers": ...}`) nothing is read from stdin. Every
    event is printed; the tunnel listeners print one `tunnel_listening` line each.
    """
    try:
        relay = await Relay.connect(socket_path, wait_s=wait_s, hello_s=hello_s)
    except ChannelClosed as exc:
        _emit({"t": "closed", "error": str(exc)})
        return 2
    async with relay:
        _emit({"t": "connected", "guest": relay.guest_hello})
        for app_port, host_port in tunnels:
            host, port = await relay.listen_tunnel(app_port, host_port)
            _emit({"t": "tunnel_listening", "app_port": app_port, "host": host, "port": port})
        if script is not None:
            prompter = ScriptedPrompter(turns=list(script.get("turns") or []),
                                        answers=script.get("answers", "go with the first option"),
                                        on_event=_emit)
            summary = await prompter.run(relay)
            _emit({"t": "summary", **{k: v for k, v in summary.items() if k != "events"}})
            return 0 if summary["ended"] else 1

        async def commands() -> None:
            async for line in _stdin_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                    kind = cmd["cmd"]
                    if kind == "turn":
                        await relay.send_user_turn(cmd["text"])
                    elif kind == "answer":
                        await relay.answer_question(cmd["qid"], cmd["answers"])
                    elif kind == "end":
                        await relay.end_session(cmd.get("reason", "host"))
                    else:
                        _emit({"t": "cli_error", "error": f"unknown command {kind!r}"})
                except (json.JSONDecodeError, KeyError, TypeError, ProtocolError) as exc:
                    _emit({"t": "cli_error", "error": f"{exc.__class__.__name__}: {exc}"})
                except ChannelClosed:
                    return

        reader = asyncio.create_task(commands())
        last = {}
        async for event in relay.events():
            _emit(event)
            last = event
        reader.cancel()
        return 0 if last.get("t") == "ended" else 1

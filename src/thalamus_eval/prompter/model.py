"""The prompter's model: stateless, tool-free calls.

Every call gets one constructed prompt and nothing else, so what the prompter knows is
exactly what the controller put in it. `CodexModel` runs `codex exec` non-interactively
under a private CODEX_HOME. What was measured on codex-cli 0.154.0 (see
`CodexModel.isolation_notes`):

  * Without `--ephemeral` and a private CODEX_HOME, `codex exec` loads the operator's own
    config: MCP servers (the Thalamus graph), hooks and thread state. A private
    CODEX_HOME holding only `auth.json` and a generated `config.toml` removes all three.
  * Its `read-only` sandbox restricts writes, not reads, and on a host where bubblewrap
    cannot create user namespaces it fails outright. So file reading is closed by
    removing the tools that can do it (`shell_tool`, `unified_exec`, `view_image`, web
    search, plugins, apps, hooks), and the call's environment is scrubbed (HOME and cwd
    are an empty scratch directory). The `exec` tool that remains is a JavaScript
    evaluator with no `fs`, `fetch` or imports.
  * stdin must be closed: with an open pipe the CLI waits for EOF before starting.
  * `-i FILE` attaches an image to the prompt with the tool lockdown unchanged (measured:
    a good and a deliberately broken page screenshot got opposite layout verdicts, 9k input
    tokens each). Images are the only files a call can see, and only those the caller passes.
  * `--output-schema` gives enforced JSON in the final message; usage comes from the
    `turn.completed` event.

Anything in the JSONL stream that is not a message is treated as a violation and fails
the call, so a codex release that adds a tool surfaces as an error, not as silent access.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class ModelError(RuntimeError):
    pass


class ModelTimeout(ModelError):
    """A call produced no finish within the timeout; `diagnostic` says where it was stuck."""

    def __init__(self, message: str, diagnostic: str = "", hangs: int = 1) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic
        self.hangs = hangs


class ToolUseViolation(ModelError):
    """The model's call used a tool; the prompter's calls must have none."""


@dataclass
class Reply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    seconds: float = 0.0
    purpose: str = ""
    #: The model that answered: named in the stream when codex says so, else the configured
    #: model, else "cli-default".
    model: str = ""
    #: Timed-out attempts that preceded this answer.
    hangs: int = 0
    hang_notes: list = field(default_factory=list)


class Model(Protocol):
    async def complete(self, prompt: str, *, schema: dict | None = None,
                       purpose: str = "", images: list[Path] | None = None) -> Reply: ...


#: Features turned off in the private config. Each closes a route to the host's files,
#: network or the operator's own memory.
DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "view_image", "apps", "plugins", "hooks", "multi_agent",
    "browser_use", "computer_use", "image_generation", "skill_search", "memories", "goals",
    "sleep_tool", "request_user_input")

ALLOWED_ITEM_TYPES = {"agent_message", "reasoning"}

#: Variables a call may inherit. Nothing else from the caller's environment reaches it.
PASS_ENV = ("PATH", "LANG", "LC_ALL", "TERM", "SSL_CERT_FILE", "SSL_CERT_DIR",
            "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy")


def build_config(model: str | None, effort: str) -> str:
    lines = []
    if model:
        lines.append(f'model = "{model}"')
    lines.append(f'model_reasoning_effort = "{effort}"')
    lines.append('web_search = "disabled"')
    lines.append("[features]")
    lines.extend(f"{name} = false" for name in DISABLED_FEATURES)
    return "\n".join(lines) + "\n"


DEFAULT_API_KEY_FILE = Path.home() / ".config" / "thalamus-eval" / "openai-api-key"
AUTH_MODES = ("api-key", "chatgpt")


@dataclass
class CodexModel:
    """`codex exec` as a stateless text-in, text-out function."""

    home: Path                       # private CODEX_HOME (created here)
    codex_bin: str = "codex"
    model: str | None = None         # None: the CLI's default model
    effort: str = "low"
    timeout_s: float = 90.0
    #: Attempts after a timeout: a hung call is killed and asked again once.
    retries: int = 1
    #: "api-key" (default): the private home is logged in with the key in `api_key_file`;
    #: "chatgpt": the operator's ChatGPT login is copied from `auth_source`.
    auth: str = "api-key"
    api_key_file: Path = field(default_factory=lambda: DEFAULT_API_KEY_FILE)
    auth_source: Path = field(default_factory=lambda: Path.home() / ".codex" / "auth.json")
    auth_mode: str = ""
    scratch_root: Path | None = None
    #: Directories the private home must not sit inside (anything copied into a cell).
    forbidden_parents: tuple[Path, ...] = ()
    calls: int = 0
    hangs: int = 0

    def __post_init__(self) -> None:
        self.home = Path(self.home)
        resolved = self.home.resolve()
        for bad in self.forbidden_parents:
            if resolved == Path(bad).resolve() or Path(bad).resolve() in resolved.parents:
                raise ModelError(f"codex home {self.home} is inside {bad}, which is copied "
                                 "into a cell")
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.home.chmod(0o700)
        if self.auth not in AUTH_MODES:
            raise ModelError(f"auth must be one of {AUTH_MODES}")
        self._login()
        (self.home / "config.toml").write_text(build_config(self.model, self.effort))

    def _login(self) -> None:
        """Log the private home in, then read the mode back from `codex login status`.

        The key is piped on stdin to `codex login --with-api-key`: never on argv, never in
        the environment, never in a message. The environment of these commands is the same
        scrubbed one calls get, so `OPENAI_API_KEY` in the caller's environment reaches
        nothing.
        """
        auth = self.home / "auth.json"
        env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
        env["HOME"] = str(self.home)
        env["CODEX_HOME"] = str(self.home)
        if self.auth == "api-key":
            path = Path(self.api_key_file)
            try:
                mode = path.stat().st_mode
            except OSError:
                raise ModelError(f"no OpenAI API key file at {path}; create it (mode 0600) "
                                 "or choose --auth chatgpt") from None
            if mode & 0o077:
                raise ModelError(f"{path} must be mode 0600 (it is {mode & 0o777:04o})")
            key = path.read_text().strip()
            if not key:
                raise ModelError(f"{path} is empty")
            auth.unlink(missing_ok=True)
            done = subprocess.run([self.codex_bin, "login", "--with-api-key"], input=key + "\n",
                                  capture_output=True, text=True, env=env, timeout=60)
            if done.returncode != 0:
                raise ModelError(f"codex login --with-api-key exited {done.returncode}")
            expect, self.auth_mode = "API key", "api-key"
        else:
            if not self.auth_source.exists():
                raise ModelError(f"no codex login at {self.auth_source}")
            shutil.copyfile(self.auth_source, auth)
            auth.chmod(0o600)
            expect, self.auth_mode = "ChatGPT", "chatgpt"
        status = subprocess.run([self.codex_bin, "login", "status"], capture_output=True,
                                text=True, env=env, timeout=60)
        said = (status.stdout + status.stderr).strip().splitlines()
        if status.returncode != 0 or not said or expect not in said[0]:
            raise ModelError(f"the private codex home is not logged in with {self.auth_mode} "
                             f"(codex login status: {said[0].split(' - ')[0] if said else ''})")

    def argv(self, workdir: Path, schema_path: Path | None, last: Path,
             images: list[Path] | None = None) -> list[str]:
        args = [self.codex_bin, "exec", "--ephemeral", "--skip-git-repo-check",
                "-s", "read-only", "-C", str(workdir), "--json", "-o", str(last)]
        if schema_path is not None:
            args += ["--output-schema", str(schema_path)]
        for image in images or []:
            args += ["-i", str(image)]
        return args + ["-"]

    def env(self, workdir: Path) -> dict[str, str]:
        env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
        env["HOME"] = str(workdir)
        env["CODEX_HOME"] = str(self.home)
        return env

    async def complete(self, prompt: str, *, schema: dict | None = None,
                       purpose: str = "", images: list[Path] | None = None) -> Reply:
        started = time.monotonic()
        hangs = 0
        diagnostics: list[str] = []
        while True:
            try:
                reply = await self._once(prompt, schema, purpose, images)
            except ModelTimeout as exc:
                hangs += 1
                self.hangs += 1
                diagnostics.append(exc.diagnostic)
                if hangs > self.retries:
                    raise ModelTimeout(str(exc), "\n".join(diagnostics), hangs) from None
                continue
            reply.hangs = hangs
            reply.model = reply.model or self.model or "cli-default"
            reply.hang_notes = diagnostics
            reply.seconds = round(time.monotonic() - started, 2)
            return reply

    async def _once(self, prompt: str, schema, purpose: str, images) -> Reply:
        self.calls += 1
        with tempfile.TemporaryDirectory(prefix="prompter-", dir=self.scratch_root) as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            schema_path = None
            if schema is not None:
                schema_path = Path(tmp) / "schema.json"
                schema_path.write_text(json.dumps(schema))
            last = Path(tmp) / "last.txt"
            proc = await asyncio.create_subprocess_exec(
                *self.argv(workdir, schema_path, last, images), cwd=workdir, env=self.env(workdir),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=True)
            out, err = bytearray(), bytearray()

            async def pump(stream, buf):
                while chunk := await stream.read(65536):
                    buf += chunk

            async def talk():
                proc.stdin.write(prompt.encode())
                await proc.stdin.drain()
                proc.stdin.close()
                await asyncio.gather(pump(proc.stdout, out), pump(proc.stderr, err), proc.wait())
            try:
                await asyncio.wait_for(talk(), self.timeout_s)
            except asyncio.TimeoutError:
                diagnostic = describe_hang(proc.pid, bytes(out), bytes(err), last, images,
                                           len(prompt))
                # the CLI is a launcher around a native binary: kill the whole group, or the
                # child keeps the pipes open and the wait never returns
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
                raise ModelTimeout(f"codex exec timed out after {self.timeout_s:.0f}s",
                                   diagnostic) from None
            if proc.returncode != 0:
                said = [ln for ln in bytes(out).decode(errors="replace").splitlines()
                        if '"error"' in ln or '"turn.failed"' in ln]
                raise ModelError(f"codex exec exited {proc.returncode}: "
                                 f"{bytes(err).decode(errors='replace')[-300:]} "
                                 f"{(said[-1] if said else '')[:300]}".strip())
            reply = parse_stream(bytes(out).decode(errors="replace"), purpose=purpose)
            if last.exists() and last.read_text().strip():
                reply.text = last.read_text().strip()
        return reply


def describe_hang(pgid: int, out: bytes, err: bytes, last: Path, images, prompt_chars: int) -> str:
    """Where a stuck call is: its process group, what each holds open, what it has said."""
    lines = [f"prompt_chars={prompt_chars} images={len(images or [])} "
             f"stdout_bytes={len(out)} stderr_bytes={len(err)} answer_file={last.exists()}"]
    try:
        ps = subprocess.run(["ps", "-o", "pid,ppid,stat,wchan:24,etimes,cmd", "-g", str(pgid)],
                            capture_output=True, text=True, timeout=5).stdout
        lines.append(ps[-1500:])
        for pid in [int(r.split()[0]) for r in ps.splitlines()[1:] if r.split()]:
            fds = []
            for fd in sorted(Path(f"/proc/{pid}/fd").iterdir(), key=lambda p: int(p.name)):
                try:
                    fds.append(f"{fd.name}->{os.readlink(fd)}")
                except OSError:
                    pass
            lines.append(f"fds {pid}: " + " ".join(fds)[:600])
    except Exception as exc:                          # noqa: BLE001
        lines.append(f"(diagnosis failed: {exc})")
    events = [ln[:160] for ln in out.decode(errors="replace").splitlines() if ln.startswith("{")]
    lines.append("last events: " + " | ".join(events[-4:]))
    lines.append("stderr tail: " + err.decode(errors="replace")[-300:])
    return "\n".join(lines)


def parse_stream(stream: str, *, purpose: str = "") -> Reply:
    """The final message and usage from `codex exec --json`; a tool item is a violation."""
    texts: list[str] = []
    usage: dict = {}
    named = ""
    for line in stream.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if isinstance(event.get("model"), str):
            named = event["model"]
        if kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") not in ALLOWED_ITEM_TYPES:
                raise ToolUseViolation(f"codex call used item type {item.get('type')!r}")
            if item.get("type") == "agent_message":
                texts.append(item.get("text", ""))
        elif kind == "turn.completed":
            usage = event.get("usage") or {}
        elif kind in ("turn.failed", "error"):
            raise ModelError(f"codex reported {kind}: {json.dumps(event)[:300]}")
    if not texts:
        raise ModelError("codex returned no message")
    return Reply(text=texts[-1].strip(), purpose=purpose,
                 input_tokens=int(usage.get("input_tokens") or 0),
                 cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
                 output_tokens=int(usage.get("output_tokens") or 0), model=named)


def parse_json(text: str) -> dict:
    """A JSON object from a reply, tolerating a code fence around it."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ModelError(f"reply is not JSON: {exc}: {text[:200]!r}") from exc
    if not isinstance(value, dict):
        raise ModelError("reply is not a JSON object")
    return value


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


class Metered:
    """Wraps a model and tallies calls and tokens by purpose."""

    def __init__(self, inner: Model) -> None:
        self.inner = inner
        self.by_purpose: dict[str, Usage] = {}
        self.log: list[dict] = []
        self.hangs = 0
        self.auth_mode = getattr(inner, "auth_mode", None)

    async def complete(self, prompt: str, *, schema: dict | None = None,
                       purpose: str = "", images: list[Path] | None = None) -> Reply:
        try:
            reply = await self.inner.complete(prompt, schema=schema, purpose=purpose,
                                              images=images)
        except ModelTimeout as exc:
            self.hangs += exc.hangs
            self.log.append({"purpose": purpose, "input_tokens": 0, "output_tokens": 0,
                             "seconds": 0.0, "hangs": exc.hangs, "failed": True,
                             "diagnostic": exc.diagnostic})
            raise
        self.hangs += reply.hangs
        row = self.by_purpose.setdefault(purpose, Usage())
        row.calls += 1
        row.input_tokens += reply.input_tokens
        row.cached_input_tokens += reply.cached_input_tokens
        row.output_tokens += reply.output_tokens
        self.log.append({"purpose": purpose, "input_tokens": reply.input_tokens,
                         "cached_input_tokens": reply.cached_input_tokens,
                         "output_tokens": reply.output_tokens, "model": reply.model,
                         "seconds": reply.seconds,
                         "hangs": reply.hangs, **({"diagnostics": reply.hang_notes}
                                                  if reply.hang_notes else {})})
        return reply

    async def json(self, prompt: str, schema: dict, purpose: str, *, retries: int = 1,
                   images: list[Path] | None = None) -> dict:
        """A schema-shaped reply; one retry on an unparseable one."""
        last: Exception | None = None
        for _ in range(retries + 1):
            reply = await self.complete(prompt, schema=schema, purpose=purpose, images=images)
            try:
                return parse_json(reply.text)
            except ModelError as exc:
                last = exc
        raise ModelError(str(last))

    def totals(self) -> dict:
        return {p: {"calls": u.calls, "input_tokens": u.input_tokens,
                    "cached_input_tokens": u.cached_input_tokens,
                    "output_tokens": u.output_tokens} for p, u in self.by_purpose.items()}

    def usage(self, since: int = 0) -> dict:
        """Token counts of the calls logged from index `since`, in total and by model.

        Counts only; cost is computed from them later. A timed-out attempt is killed
        before codex reports usage, so it adds no tokens.
        """
        out = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
               "models": {}, "auth_mode": self.auth_mode}
        for row in self.log[since:]:
            if row.get("failed"):
                continue
            bucket = out["models"].setdefault(row.get("model") or "unknown", {
                "calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0})
            for target in (out, bucket):
                target["calls"] += 1
                for k in ("input_tokens", "cached_input_tokens", "output_tokens"):
                    target[k] += row.get(k, 0)
        return out


class FakeModel:
    """A scripted model for tests: `handler(purpose, prompt, schema)` returns text or a dict."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.prompts: list[tuple[str, str]] = []
        self.images: list[tuple[str, list[Path]]] = []

    async def complete(self, prompt: str, *, schema: dict | None = None,
                       purpose: str = "", images: list[Path] | None = None) -> Reply:
        self.prompts.append((purpose, prompt))
        self.images.append((purpose, list(images or [])))
        out = self.handler(purpose, prompt, schema)
        if asyncio.iscoroutine(out):
            out = await out
        text = out if isinstance(out, str) else json.dumps(out)
        return Reply(text=text, purpose=purpose, input_tokens=len(prompt) // 4,
                     output_tokens=max(1, len(text) // 4))

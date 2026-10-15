"""Coarse, human-like recollection of earlier sessions.

The prompter does not remember what it said. After each session a separate call turns
the session's visible conversation (the prompter's messages and the worker's text, not
tool calls or transcripts) into a short recollection of the kind a person keeps: what the
app does now in broad terms, what was asked for, what was left for later, how it felt.
No filenames, line references, function names or other code identifiers. A
deterministic lint enforces that; a failing text is rewritten up to `retries` times with
the offending spans named, and what still fails is scrubbed span by span, so the stored
text always lints clean. The lint result of every attempt is stored with the text.

Older recollections get coarser: a recollection is kept at full length for
`fine_sessions` sessions, as a short gist for the next `gist_sessions`, and as a single
line after that. Re-summarizing starts from the current (already coarse) text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from thalamus_eval.prompter.model import Metered

EXTENSIONS = (
    "py|pyi|js|jsx|mjs|cjs|ts|tsx|json|jsonl|yaml|yml|toml|md|mdx|html|htm|css|scss|sql|sh|"
    "txt|csv|db|sqlite|sqlite3|lock|cfg|ini|env|go|rs|java|rb|php|c|h|cpp|vue|svelte")

#: (kind, pattern). Order is not significant; overlapping matches are merged.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("backticked", re.compile(r"`[^`\n]+`|```.*?```", re.S)),
    ("path", re.compile(r"(?<![\w/])(?:\.{0,2}/)?(?:[\w.-]+/){2,}[\w.-]*")),
    ("path", re.compile(r"(?<![\w/])/[\w.{}:-]+(?:/[\w.{}:-]+)*")),
    ("filename", re.compile(rf"\b[\w.-]+\.(?:{EXTENSIONS})\b", re.I)),
    ("line-ref", re.compile(r"\blines?\s+\d+(?:\s*(?:-|to|–)\s*\d+)?\b|\b\w+\.\w+:\d+(?:-\d+)?\b|\bL\d+(?:-L?\d+)?\b", re.I)),
    ("snake_case", re.compile(r"\b[A-Za-z][a-z0-9]*(?:_[A-Za-z0-9]+)+\b")),
    ("camelCase", re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b")),
    ("PascalCase", re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b")),
    ("dotted", re.compile(r"\b[A-Za-z_]\w+(?:\.[A-Za-z_]\w+)+\b")),
    ("call", re.compile(r"\b\w+\(\)")),
    ("header", re.compile(r"\bX-[A-Z][\w-]+\b")),
]


@dataclass
class Violation:
    kind: str
    text: str
    start: int
    end: int

    def as_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text}


def lint(text: str) -> list[Violation]:
    """Code identifiers in `text`: non-overlapping, in order of position."""
    found: list[Violation] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            found.append(Violation(kind, m.group(0), m.start(), m.end()))
    found.sort(key=lambda v: (v.start, -(v.end - v.start)))
    merged: list[Violation] = []
    for v in found:
        if merged and v.start < merged[-1].end:
            continue
        merged.append(v)
    return merged


def scrub(text: str) -> str:
    """Replace every violation with a vague phrase, then lint again until clean."""
    for _ in range(5):
        hits = lint(text)
        if not hits:
            return text
        for v in reversed(hits):
            text = text[:v.start] + "one of the files" + text[v.end:] \
                if v.kind in ("path", "filename") else text[:v.start] + "that part" + text[v.end:]
    return text


@dataclass
class Recollection:
    session: int
    level: int                       # 0 as written, 1 gist, 2 one line
    text: str
    lint: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)   # earlier texts, for audit only

    def as_dict(self) -> dict:
        return {"session": self.session, "level": self.level, "text": self.text,
                "lint": self.lint, "history": self.history}


class RecollectionStore:
    """`<dir>/recollections.json`: the prompter's accumulated coarse memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.items: list[Recollection] = []
        if self.path.exists():
            for row in json.loads(self.path.read_text()):
                self.items.append(Recollection(**row))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([r.as_dict() for r in self.items], indent=2))
        tmp.replace(self.path)

    def texts_before(self, session: int) -> list[Recollection]:
        return sorted((r for r in self.items if r.session < session), key=lambda r: r.session)

    def render(self, session: int) -> str:
        rows = self.texts_before(session)
        if not rows:
            return "(nothing: this is the first session)"
        return "\n\n".join(f"After session {r.session}:\n{r.text}" for r in rows)

    def put(self, rec: Recollection) -> None:
        self.items = [r for r in self.items if r.session != rec.session] + [rec]
        self.items.sort(key=lambda r: r.session)


SUMMARY_SYSTEM = """\
You are writing down, as the person who directed a coding assistant, what you remember of \
a work session that just ended. This is a human's recollection: coarse, partial, in your \
own words. Say what you were after, roughly what the app does now, what you asked to \
change, what you told the assistant to leave for later, and how it felt (smooth, \
fiddly, something still bothering you). Plain prose, first person, no lists.

HARD RULE: you do not remember any code identifiers. No file names or paths, no file \
extensions, no line numbers, no function, class, variable, table or field names (nothing in \
snake_case, camelCase or with dots), no API paths, nothing in backticks. Say "the page that \
lists the groups", "the balances panel", "the rule about duplicate names", never the name \
in the code. Ordinary product words and visible button or heading text are fine."""

LEVEL_LIMITS = {0: "4 to 7 sentences (at most 140 words)", 1: "2 to 3 sentences (at most 50 words)",
                2: "one sentence (at most 25 words)"}


def render_transcript(entries: list[dict], *, per_message: int = 1800, total: int = 30000) -> str:
    """The visible conversation of a session: who said what, nothing else."""
    lines = []
    for e in entries:
        who = "ME" if e["role"] == "operator" else "ASSISTANT"
        text = e["text"].strip()
        if len(text) > per_message:
            text = text[:per_message] + " [...]"
        lines.append(f"{who}: {text}")
    out = "\n\n".join(lines)
    if len(out) > total:
        half = total // 2
        out = out[:half] + "\n\n[... middle of the session omitted ...]\n\n" + out[-half:]
    return out


class Recollector:
    def __init__(self, model: Metered, *, retries: int = 2, fine_sessions: int = 1,
                 gist_sessions: int = 2) -> None:
        self.model = model
        self.retries = retries
        self.fine_sessions = fine_sessions
        self.gist_sessions = gist_sessions

    def target_level(self, age: int) -> int:
        if age < self.fine_sessions:
            return 0
        if age < self.fine_sessions + self.gist_sessions:
            return 1
        return 2

    async def _clean(self, draft: str, rewrite_prompt) -> tuple[str, dict]:
        attempts = []
        text = draft.strip()
        for attempt in range(self.retries + 1):
            hits = lint(text)
            attempts.append({"attempt": attempt, "violations": [v.as_dict() for v in hits]})
            if not hits:
                return text, {"passed": True, "scrubbed": False, "attempts": attempts}
            if attempt == self.retries:
                break
            reply = await self.model.complete(rewrite_prompt(text, hits), purpose="recollect_rewrite")
            text = reply.text.strip()
        text = scrub(text)
        final = lint(text)
        return text, {"passed": not final, "scrubbed": True, "attempts": attempts}

    async def summarize(self, session: int, entries: list[dict], context: str = "") -> Recollection:
        prompt = (f"{SUMMARY_SYSTEM}\n\n{context}\nHere is the conversation, as you and the "
                  f"assistant saw it:\n\n{render_transcript(entries)}\n\n"
                  f"Write your recollection in {LEVEL_LIMITS[0]}. Output only the recollection.")
        reply = await self.model.complete(prompt, purpose="recollect")

        def rewrite(text: str, hits: list[Violation]) -> str:
            named = ", ".join(sorted({h.text for h in hits}))
            return (f"{SUMMARY_SYSTEM}\n\nThis draft contains code identifiers ({named}). "
                    f"Rewrite it with the same content, replacing each with a plain-language "
                    f"description. Output only the rewrite.\n\nDRAFT:\n{text}")

        text, result = await self._clean(reply.text, rewrite)
        return Recollection(session=session, level=0, text=text, lint=result)

    async def coarsen_one(self, rec: Recollection, level: int) -> Recollection:
        def prompt(text: str) -> str:
            return (f"{SUMMARY_SYSTEM}\n\nYou wrote this a while ago:\n\n{text}\n\n"
                    f"Time has passed. Now you only keep the gist: write it again in "
                    f"{LEVEL_LIMITS[level]}, keeping what a person would still remember "
                    f"(what the app is for, what was left undone, anything still "
                    f"nagging) and dropping fine detail. Output only the text.")

        reply = await self.model.complete(prompt(rec.text), purpose="recollect_coarsen")

        def rewrite(text: str, hits: list[Violation]) -> str:
            named = ", ".join(sorted({h.text for h in hits}))
            return (f"{SUMMARY_SYSTEM}\n\nThis text contains code identifiers ({named}). Rewrite "
                    f"it in {LEVEL_LIMITS[level]} without them, in plain language. Output only "
                    f"the text.\n\nTEXT:\n{text}")

        text, result = await self._clean(reply.text, rewrite)
        return Recollection(
            session=rec.session, level=level, text=text, lint=result,
            history=[*rec.history, {"level": rec.level, "text": rec.text}])

    async def update(self, store: RecollectionStore, newest: int) -> list[int]:
        """Coarsen every recollection to the level its age calls for. Returns sessions changed."""
        changed = []
        for rec in list(store.items):
            want = self.target_level(newest - rec.session)
            if rec.level < want:
                store.put(await self.coarsen_one(rec, want))
                changed.append(rec.session)
        return changed

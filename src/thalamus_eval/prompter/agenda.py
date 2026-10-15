"""The campaign agenda as the prompter reads it, and the campaign state it writes.

A campaign's `agenda.yaml` lists key ideas in build order
and facts with the key idea at which each is stated (`revealed_at`), changed
(`changed_at`), reported (`reported_at`, deferred bugs) or fixed (`fix_at`). This module
turns that into the items that are *due* in a session and keeps the persisted state:
which key ideas are finished, which items were stated and when, which deferred bugs
were reported or marked not-triggered.

A fact is stated once. The state records each statement as `<fact>:<kind>`, so a
session that carries a key idea over does not repeat what an earlier session said.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class AgendaError(ValueError):
    pass


class AuthModeError(RuntimeError):
    """A store holds calls made under one login mode and is offered another."""


class DryRunError(RuntimeError):
    """A dry-run record was offered where a real run is required (or the reverse)."""


@dataclass(frozen=True)
class KeyIdea:
    id: str
    title: str
    build: str
    acceptance: tuple[str, ...]
    fixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fact:
    id: str
    cls: str
    statement: str = ""
    v1: str = ""
    v2: str = ""
    revealed_at: str | None = None
    changed_at: str | None = None
    reported_at: str | None = None
    fix_at: str | None = None
    conditional: bool = False
    locator_added: str = ""
    matters_at: tuple[str, ...] = ()

    @property
    def deferred(self) -> bool:
        return self.cls == "deferred-bug"


@dataclass(frozen=True)
class Due:
    """Something the operator owes the worker in this session."""

    fact: str
    kind: str           # reveal | change | fix
    text: str           # what is to be conveyed, as the agenda words it
    locator_added: str = ""

    @property
    def key(self) -> str:
        return f"{self.fact}:{self.kind}"


@dataclass
class Agenda:
    key_ideas: list[KeyIdea]
    facts: dict[str, Fact]

    @classmethod
    def load(cls, path: str | Path) -> "Agenda":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict) or not raw.get("key_ideas"):
            raise AgendaError(f"{path}: no key_ideas")
        ideas = []
        for item in raw["key_ideas"]:
            ideas.append(KeyIdea(
                id=str(item["id"]), title=str(item.get("title", item["id"])),
                build=str(item.get("build", "")).strip(),
                acceptance=tuple(str(a) for a in item.get("acceptance") or ()),
                fixes=tuple(str(f) for f in item.get("fixes") or ())))
        facts = {}
        for item in raw.get("facts") or []:
            fact = Fact(
                id=str(item["id"]), cls=str(item.get("class", "")),
                statement=" ".join(str(item.get("statement", "")).split()),
                v1=" ".join(str(item.get("v1", "")).split()),
                v2=" ".join(str(item.get("v2", "")).split()),
                revealed_at=item.get("revealed_at"), changed_at=item.get("changed_at"),
                reported_at=item.get("reported_at"), fix_at=item.get("fix_at"),
                conditional=bool(item.get("conditional", False)),
                locator_added=str(item.get("locator_added", "")),
                matters_at=tuple(str(m) for m in item.get("matters_at") or ()))
            facts[fact.id] = fact
        known = {k.id for k in ideas}
        for fact in facts.values():
            for where in (fact.revealed_at, fact.changed_at, fact.reported_at, fact.fix_at):
                if where is not None and where not in known:
                    raise AgendaError(f"fact {fact.id} names unknown key idea {where}")
        return cls(ideas, facts)

    def key_idea(self, key_id: str) -> KeyIdea:
        for idea in self.key_ideas:
            if idea.id == key_id:
                return idea
        raise AgendaError(f"unknown key idea {key_id}")

    def due(self, key_id: str) -> list[Due]:
        """Everything to state, change or fix while working on `key_id`."""
        out: list[Due] = []
        for fact in self.facts.values():
            if fact.revealed_at == key_id:
                out.append(Due(fact.id, "reveal", fact.v1 or fact.statement, fact.locator_added))
            if fact.changed_at == key_id and fact.changed_at != fact.revealed_at:
                out.append(Due(fact.id, "change", fact.v2 or fact.statement, fact.locator_added))
            if fact.fix_at == key_id:
                out.append(Due(fact.id, "fix", fact.statement, fact.locator_added))
        return out

    def requirements(self, key_id: str, stated: dict) -> list[tuple[str, str]]:
        """What the operator has required of the worker that bears on `key_id`.

        Stated rules, changes and fixes whose `matters_at` names the key idea (or that were
        stated during it), at the latest version stated: a change replaces its rule.
        """
        out = []
        for fact in self.facts.values():
            if fact.deferred:
                continue
            changed = f"{fact.id}:change" in stated
            revealed = f"{fact.id}:reveal" in stated
            if not (changed or revealed):
                continue
            here = key_id in fact.matters_at or any(
                stated.get(f"{fact.id}:{k}", {}).get("key_idea") == key_id
                for k in ("reveal", "change"))
            if here:
                out.append((fact.id, (fact.v2 or fact.statement) if changed
                            else (fact.v1 or fact.statement)))
        return out

    def deferred_here(self, key_id: str) -> list[Fact]:
        return [f for f in self.facts.values() if f.deferred and f.reported_at == key_id]


@dataclass
class CampaignState:
    """What persists between sessions of one campaign run (`<store>/state.json`)."""

    completed: list[str] = field(default_factory=list)
    #: "<fact>:<kind>" -> {"session": n, "turn": n, "key_idea": id}
    stated: dict[str, dict] = field(default_factory=dict)
    #: fact id -> "reported" | "not-triggered"
    deferred: dict[str, str] = field(default_factory=dict)
    sessions: list[dict] = field(default_factory=list)
    #: True once a session ran without a browser (tests, dry runs); None while empty.
    dry_run: bool | None = None
    #: How the prompter's Codex was logged in ("api-key" or "chatgpt"); one store, one mode.
    auth_mode: str | None = None
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "CampaignState":
        path = Path(path)
        state = cls(path=path)
        if path.exists():
            raw = json.loads(path.read_text())
            state.completed = list(raw.get("completed", []))
            state.stated = dict(raw.get("stated", {}))
            state.deferred = dict(raw.get("deferred", {}))
            state.sessions = list(raw.get("sessions", []))
            state.dry_run = raw.get("dry_run")
            state.auth_mode = raw.get("auth_mode")
        return state

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "completed": self.completed, "stated": self.stated, "deferred": self.deferred,
            "sessions": self.sessions, "dry_run": self.dry_run,
            "auth_mode": self.auth_mode}, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def require_real(self) -> None:
        """Refuse a store whose sessions were accepted without a browser."""
        if self.dry_run:
            raise DryRunError(f"{self.path}: report-only (dry-run) sessions are not a real run")

    def current(self, agenda: Agenda) -> KeyIdea | None:
        """The first key idea not completed: an unfinished one carries over."""
        for idea in agenda.key_ideas:
            if idea.id not in self.completed:
                return idea
        return None

    def outstanding(self, agenda: Agenda, key_id: str) -> list[Due]:
        return [d for d in agenda.due(key_id) if d.key not in self.stated]

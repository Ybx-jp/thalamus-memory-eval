"""One command runs a campaign: the prompter plays the operator of every chained session.

`plan_factory` returns the `chain.PlanFactory` that `chain.run_chain` calls before each
cell boots. For session k it reads the persisted prompter state, picks the first key idea
not yet accepted (one key idea per session; an unfinished one carries over), builds the
`SessionDriver` for it and wraps it as a conversation plan. The driver writes the opener
(policy `continue` or `self-contained`), relays the conversation to the in-guest worker,
runs browser acceptance through the tunnel when the worker reports the key idea done,
sends the fixed wrap-up, and ends the session; a token count past the hard ceiling ends
it mid-turn through `end_session`. State, recollections and the per-session records live
under `<root>/prompter/`, between cells, outside the worktree and HOME that cross.

A real run needs the browser. `dry_run` is the only way to run without one, and it
stamps the store so it can never be mistaken for a real run.
"""

from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from thalamus_eval import chain as chain_mod
from thalamus_eval.prompter import PrompterConfig, conversation_plan, default_model
from thalamus_eval.prompter.agenda import Agenda, CampaignState
from thalamus_eval.prompter.browser import PlaywrightBrowser

#: The `prompt` of a conversation session: the worker's first words come from the driver.
CONVERSATION_PROMPT = "(driven over the conversation channel)"

#: Left out of the worktree copy the self-contained opener may look through.
COPY_IGNORE = (".venv", "node_modules", ".claude", ".codex", "__pycache__")


@dataclass
class CampaignSpec:
    """Where the campaign's files are and how the prompter is set."""

    agenda: Path
    spec: Path
    brief: Path
    config: PrompterConfig
    #: Prompter state and records; default `<chain root>/prompter`.
    store_dir: Path | None = None
    codex_home: Path | None = None
    codex_model: str | None = None
    effort: str = "low"
    auth: str = "api-key"
    api_key_file: Path | None = None
    chromium: str | None = None
    #: Builds the model; tests supply a fake.
    model_factory: Callable[[Path], object] | None = None
    #: Builds the browser for a tunnel URL; tests supply a fake. Ignored in a dry run.
    browser_factory: Callable[[str], object] | None = None


def browser_missing() -> str | None:
    if importlib.util.find_spec("playwright") is None:
        return "playwright is not installed"
    return None


def check_ready(spec: CampaignSpec) -> None:
    """Refuse before any spend: unreadable inputs, or a real run with no browser."""
    for label, path in (("agenda", spec.agenda), ("spec", spec.spec), ("brief", spec.brief)):
        if not Path(path).is_file():
            raise chain_mod.ChainError(f"campaign {label} {path} is not a file")
    Agenda.load(spec.agenda)
    if not spec.config.dry_run and spec.browser_factory is None:
        reason = browser_missing()
        if reason:
            raise chain_mod.ChainError(
                f"a real campaign run needs the browser click-through ({reason}); "
                "use --dry-run for a run without one")


def _copy_worktree(src: Path, dest: Path) -> Path:
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest, symlinks=True, ignore=shutil.ignore_patterns(*COPY_IGNORE))
    return dest


def plan_factory(chain: chain_mod.Chain, spec: CampaignSpec) -> chain_mod.PlanFactory:
    store_dir = Path(spec.store_dir or chain.root / "prompter")
    agenda = Agenda.load(spec.agenda)
    if spec.config.dry_run:
        browser_factory = None
    elif spec.browser_factory is not None:
        browser_factory = spec.browser_factory
    else:
        def browser_factory(url: str):
            return PlaywrightBrowser(url, executable_path=spec.chromium)

    def make(index: int, prior: dict | None) -> chain_mod.SessionPlan | None:
        if CampaignState.load(store_dir / "state.json").current(agenda) is None:
            return None
        home = Path(spec.codex_home or store_dir / "codex-home")
        if spec.model_factory:
            model = spec.model_factory(home)
        else:
            model = default_model(home, model=spec.codex_model, effort=spec.effort,
                                  auth=spec.auth, api_key_file=spec.api_key_file,
                                  forbidden_parents=[chain.worktree, chain.home])
        copy = None
        if spec.config.policy == "self-contained" and index > 1:
            copy = _copy_worktree(chain.worktree, chain.session_dir(index) / "worktree-copy")
        plan = conversation_plan(
            config=spec.config, agenda_path=spec.agenda, spec_path=spec.spec,
            brief_path=spec.brief, store_dir=store_dir, session=index, model=model,
            worktree_copy=copy, browser_factory=browser_factory)
        return chain_mod.SessionPlan(prompt=CONVERSATION_PROMPT, label=f"s{index:02d}",
                                     conversation=plan)
    return make

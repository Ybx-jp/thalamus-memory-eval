"""`thalamus-eval chain --campaign`: the prompter wired into the chain, with fakes.

The VM is replaced by a fake `agent_runner` that plays the guest end of the relay socket
(`FakeGuest`) for as long as the host driver talks to it; the operator's model and the
browser are fakes too. What is checked is the wiring: one key idea per session, state
carried between sessions through the campaign store, the policy reaching the opener, the
real-run refusal without a browser, and the early end when the agenda is done.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from channel_fakes import FakeGuest
from test_prompter import BRIEF, FakeBrowser, SPEC, make_handler, verdict, worker
from thalamus_eval import arms, campaign, chain, vm
from thalamus_eval.prompter import PrompterConfig
from thalamus_eval.prompter.agenda import Agenda, CampaignState
from thalamus_eval.prompter.model import FakeModel

AGENDA = """\
key_ideas:
  - {id: K1, title: First, acceptance: ["see the first thing"]}
  - {id: K2, title: Second, acceptance: ["see the second thing"]}
facts: []
"""


class FakeVm:
    """`run_agent` stand-in: a worker answering over the relay socket the job declares."""

    def __init__(self):
        self.calls = []
        self.guests = []

    def __call__(self, worktree, prompt, **kw):
        n = len(self.calls) + 1
        self.calls.append({"prompt": prompt, **kw})
        (worktree / f"file{n}.txt").write_text(f"session {n}")
        sock = Path(kw["session"]["relay_socket"])
        done = asyncio.Event()
        inner = worker([("did the thing", 50_000), ("committed", 60_000)])

        async def on_frame(guest, frame):
            await inner(guest, frame)
            if frame["t"] == "end_session":
                done.set()

        async def go():
            guest = FakeGuest(sock, on_frame=on_frame)
            await guest.start()
            self.guests.append(guest)
            try:
                await asyncio.wait_for(done.wait(), 20)
                await asyncio.sleep(0.2)
            finally:
                await guest.stop()
        asyncio.run(go())
        return arms.AgentRun(
            session_id=f"s{n}", result="done", cost_usd=1.0, duration_ms=10, num_turns=2,
            is_error=False, probe={"pre": {"ok": True}, "post": {"ok": True}},
            vm_confinement={"cell": f"c{n}"}, delivery={"delivered": True, "check": {}},
            prep=[], cost_envelope={"total_cost_usd": 1.0, "models": {}})


def handler_for(agenda, seen):
    base = make_handler(states=["reports_done"] * 6)

    def handler(purpose, prompt, schema):
        seen.append((purpose, prompt))
        if purpose in ("opener", "opener_fix"):
            return {"worker_state": "working", "message": "start on the next part please",
                    "reveals": []}
        if purpose == "accept":
            key = next(k for k in agenda.key_ideas if k.acceptance[0] in prompt)
            return verdict([(i, "pass", "") for i in key.acceptance])
        return base(purpose, prompt, schema)
    return handler


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "refresh_credentials", lambda home: None)
    monkeypatch.setattr(chain, "check_token_lifetime", lambda home, timeout: 10**6)
    monkeypatch.setattr(arms, "seed_arm_home", lambda wt, mcp, home: (home.mkdir(parents=True, exist_ok=True), {}))
    monkeypatch.setattr(vm, "resolve_backend", lambda: Path("/bin/true"))
    for name, text in (("agenda.yaml", AGENDA), ("spec.md", SPEC), ("brief.md", BRIEF)):
        (tmp_path / name).write_text(text)
    agenda = Agenda.load(tmp_path / "agenda.yaml")
    fake_vm = FakeVm()
    seen: list = []
    browsers: list = []

    def browser(url):
        browsers.append(url)
        return FakeBrowser()

    def make(*, dry_run=False, policy="continue", browser_factory=browser):
        c = chain.Chain(arm=arms.parse_arm("campaign-bare", []), root=tmp_path / "run" / "r1",
                        name="camp", agent_runner=fake_vm)
        spec = campaign.CampaignSpec(
            agenda=tmp_path / "agenda.yaml", spec=tmp_path / "spec.md",
            brief=tmp_path / "brief.md", config=PrompterConfig(policy=policy, dry_run=dry_run),
            model_factory=lambda home: FakeModel(handler_for(agenda, seen)),
            browser_factory=browser_factory)
        return c, spec
    return make, fake_vm, seen, browsers


def test_one_key_idea_per_session_and_the_state_crosses_between_them(env):
    make, fake_vm, seen, browsers = env
    c, spec = make()
    campaign.check_ready(spec)
    summary = chain.run_chain(c, [campaign.plan_factory(c, spec)] * 5)
    assert summary["status"] == "complete" and len(summary["sessions"]) == 2, \
        "the third session was never made: the agenda was done"
    assert "nothing left" in summary["ended_early"]
    state = CampaignState.load(c.root / "prompter" / "state.json")
    assert state.completed == ["K1", "K2"] and state.dry_run is False
    recs = [r["conversation"]["driver_result"] for r in summary["sessions"]]
    assert [r["key_idea"] for r in recs] == ["K1", "K2"]
    assert all(r["key_idea_done"] and r["real_run"] for r in recs)
    assert len(browsers) == 2, "acceptance ran in a browser at the end of each key idea"
    first_user = [[f for f in g.received if f["t"] == "user_turn"] for g in fake_vm.guests]
    assert "port 8000" in first_user[0][0]["text"] and BRIEF in first_user[0][0]["text"]
    assert BRIEF not in first_user[1][0]["text"]
    assert first_user[0][-1]["text"].startswith("that covers it")
    assert fake_vm.guests[0].received[-1]["t"] == "end_session"
    assert fake_vm.calls[0]["session"]["kind"] == "sdk-conversation"
    assert (c.root / "prompter" / "sessions" / "s02.json").exists()
    assert (c.root / "prompter" / "recollections.json").exists()
    # session 2's operator was shown session 1's recollection
    assert any(p == "opener" and "I wanted the plant page first" in t for p, t in seen)


def test_an_unfinished_key_idea_carries_over_to_the_next_session(env, tmp_path):
    make, fake_vm, seen, _ = env
    c, spec = make()
    store = c.root / "prompter"
    state = CampaignState.load(store / "state.json")
    state.completed.append("K1")
    state.save()
    summary = chain.run_chain(c, [campaign.plan_factory(c, spec)] * 3)
    assert [s["conversation"]["driver_result"]["key_idea"] for s in summary["sessions"]] == ["K2"]


def test_self_contained_sessions_get_a_transcript_free_copy_of_the_worktree(env):
    make, fake_vm, seen, _ = env
    c, spec = make(policy="self-contained")
    chain.run_chain(c, [campaign.plan_factory(c, spec)] * 2)
    copy = c.session_dir(2) / "worktree-copy"
    assert (copy / "file1.txt").read_text() == "session 1"
    assert not (c.session_dir(1) / "worktree-copy").exists()
    assert any(p == "opener" and "REPOSITORY VIEW" in t and "file1.txt" in t
               for p, t in seen if "THIS SESSION'S WORK ===\nSecond" in t)


def test_a_real_run_without_the_browser_is_refused_before_spend(env, monkeypatch):
    make, fake_vm, *_ = env
    c, spec = make(browser_factory=None)
    monkeypatch.setattr(campaign, "browser_missing", lambda: "playwright is not installed")
    with pytest.raises(chain.ChainError, match="--dry-run"):
        campaign.check_ready(spec)
    assert not fake_vm.calls


def test_a_dry_run_has_no_browser_and_stamps_the_store(env):
    make, fake_vm, _, browsers = env
    c, spec = make(dry_run=True, browser_factory=None)
    campaign.check_ready(spec)
    summary = chain.run_chain(c, [campaign.plan_factory(c, spec)] * 1)
    assert summary["status"] == "complete" and not browsers
    rec = summary["sessions"][0]["conversation"]["driver_result"]
    assert rec["real_run"] is False and rec["stamp"].startswith("DRY RUN")
    assert CampaignState.load(c.root / "prompter" / "state.json").dry_run is True


def test_a_factory_that_raises_is_a_chain_error_before_any_cell(tmp_path, env):
    make, fake_vm, *_ = env
    c, _ = make()

    def broken(index, prior):
        raise ValueError("agenda unreadable")
    summary = chain.run_chain(c, [broken])
    assert summary["status"] == "halted" and summary["sessions"][0]["status"] == "chain_error"
    assert "agenda unreadable" in summary["reason"] and not fake_vm.calls

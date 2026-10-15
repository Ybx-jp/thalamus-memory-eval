"""The operator prompter: agenda, recollection lint, repo view, the session driver, codex plumbing.

No model spend: the model is `FakeModel` (or a fake `codex` executable for the subprocess
plumbing) and the worker is `channel_fakes.FakeGuest`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import stat
from pathlib import Path

import pytest

from channel_fakes import FakeGuest
from thalamus_eval import channel
from thalamus_eval.prompter import PrompterConfig, build_driver, conversation_plan
from thalamus_eval.prompter.acceptance import reconcile
from thalamus_eval.prompter.agenda import Agenda, CampaignState, DryRunError
from thalamus_eval.prompter.driver import hard_tokens_of
from thalamus_eval.prompter.model import (CodexModel, FakeModel, Metered, ModelError,
                                          ToolUseViolation, build_config, parse_stream)
from thalamus_eval.prompter.recollection import (Recollection, Recollector, RecollectionStore,
                                                 lint, scrub)
from thalamus_eval.prompter.repo import RepoRefused, RepoView

FIX = Path(__file__).parent / "fixtures" / "prompter"
AGENDA = FIX / "agenda.yaml"


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


# ------------------------------------------------------------------------------ agenda


class TestAgenda:
    def test_due_items_follow_the_reveal_protocol(self):
        a = Agenda.load(AGENDA)
        k2 = {(d.fact, d.kind) for d in a.due("K2")}
        assert ("P-AMOUNT", "reveal") in k2 and ("RC-UNITS", "reveal") in k2
        assert ("RC-UNITS", "change") not in k2
        k5 = {(d.fact, d.kind) for d in a.due("K5")}
        assert ("RC-UNITS", "change") in k5
        assert {(d.fact, d.kind) for d in a.due("K8")} >= {
            ("DB-PHOTO-ROTATE", "fix"), ("DB-CHART-AXIS", "fix")}
        change = next(d for d in a.due("K5") if d.kind == "change")
        assert "cups when under a litre" in change.text           # v2, not v1
        assert [f.id for f in a.deferred_here("K3")] == ["DB-PHOTO-ROTATE"]

    def test_a_carried_over_key_idea_does_not_restate_what_was_said(self, tmp_path):
        a = Agenda.load(AGENDA)
        st = CampaignState.load(tmp_path / "state.json")
        assert st.current(a).id == "K1"
        st.stated["P-SPECIES:reveal"] = {"session": 1, "turn": 1}
        assert {d.fact for d in st.outstanding(a, "K1")} == {"PREF-NICKNAME"}
        st.completed.append("K1")
        st.save()
        assert CampaignState.load(tmp_path / "state.json").current(a).id == "K2"

    def test_a_fact_naming_an_unknown_key_idea_is_refused(self, tmp_path):
        bad = tmp_path / "a.yaml"
        bad.write_text("key_ideas: [{id: K1, title: t}]\nfacts: [{id: F, class: x, revealed_at: K9}]")
        with pytest.raises(Exception, match="unknown key idea"):
            Agenda.load(bad)


# --------------------------------------------------------------------------- the lint


class TestLint:
    @pytest.mark.parametrize("text", [
        "it lives in src/app/server.py", "see `plants` table", "check app.js", "line 42",
        "at routes.py:88", "the create_plant handler", "calls addPlant", "PlantService class",
        "uses express.Router", "the /api/plants endpoint", "call render()", "X-Trace-Id header",
        "lines 10-20 look off", "returns nickname_too_long", "stored in data.db",
    ])
    def test_code_identifiers_are_caught(self, text):
        assert lint(text), text

    @pytest.mark.parametrize("text", [
        "I wanted the plants page to list everything, e.g. the newest first.",
        "The log looked right after the second watering; the delete button was fiddly.",
        "We agreed on 7 days for ferns, i.e. no surprises. and/or later",
        "It felt smooth. Mia and Leo were the test names.",
    ])
    def test_ordinary_prose_passes(self, text):
        assert lint(text) == []

    def test_scrub_leaves_nothing_the_lint_flags(self):
        text = "It broke in `server.py` line 12 where add_plant calls db.insert in src/x/y.js."
        assert lint(scrub(text)) == []


class TestRecollector:
    def test_a_draft_with_identifiers_is_rewritten_and_the_attempts_are_stored(self):
        calls = []

        def handler(purpose, prompt, schema):
            calls.append(purpose)
            if purpose == "recollect":
                return "we built the plant page in plants.py with add_plant"
            return "we built the page that lists plants and lets you add plants"
        rec = run(Recollector(Metered(FakeModel(handler))).summarize(
            1, [{"role": "operator", "text": "build"}, {"role": "worker", "text": "done"}]))
        assert calls == ["recollect", "recollect_rewrite"]
        assert rec.lint["passed"] and not rec.lint["scrubbed"]
        assert len(rec.lint["attempts"]) == 2
        assert rec.lint["attempts"][0]["violations"] and not rec.lint["attempts"][1]["violations"]
        assert lint(rec.text) == []

    def test_a_rewrite_that_keeps_failing_is_scrubbed_after_the_bound(self):
        model = FakeModel(lambda p, prompt, s: "it is all in `app.py` and do_thing")
        rec = run(Recollector(Metered(model), retries=2).summarize(1, []))
        assert model.prompts.__len__() == 3                 # draft + 2 rewrites, then stop
        assert rec.lint["passed"] and rec.lint["scrubbed"]
        assert lint(rec.text) == []

    def test_older_recollections_are_coarsened_by_age(self, tmp_path):
        store = RecollectionStore(tmp_path / "r.json")
        for n in (1, 2, 3, 4):
            store.put(Recollection(session=n, level=0, text=f"long text {n} " * 10))
        model = FakeModel(lambda p, prompt, s: "short gist")
        changed = run(Recollector(Metered(model), fine_sessions=1, gist_sessions=1)
                      .update(store, 4))
        assert changed == [1, 2, 3]
        assert [r.level for r in store.items] == [2, 2, 1, 0]
        assert store.items[0].history[-1]["level"] == 0      # the earlier text is kept for audit
        store.save()
        assert [r.level for r in RecollectionStore(tmp_path / "r.json").items] == [2, 2, 1, 0]


# ------------------------------------------------------------------------- repo view


class TestRepoView:
    def make(self, tmp_path):
        root = tmp_path / "wt"
        (root / "src").mkdir(parents=True)
        (root / "src" / "server.js").write_text("x")
        (root / "README.md").write_text("# app\nrun it\n")
        (root / ".claude").mkdir()
        (root / ".claude" / "settings.json").write_text("{}")
        return root

    def test_lists_files_and_docs_but_not_agent_state(self, tmp_path):
        view = RepoView(self.make(tmp_path))
        assert "src/server.js" in view.files and "README.md" in view.files
        assert not any(f.startswith(".claude") for f in view.files)
        assert "# app" in view.render()

    def test_a_tree_with_a_transcript_is_refused(self, tmp_path):
        root = self.make(tmp_path)
        (root / "notes").mkdir()
        (root / "notes" / "session.jsonl").write_text("{}")
        with pytest.raises(RepoRefused):
            RepoView(root)
        root2 = self.make(tmp_path / "b")
        (root2 / ".claude" / "projects").mkdir()
        with pytest.raises(RepoRefused):
            RepoView(root2)

    def test_unknown_paths_are_found(self, tmp_path):
        view = RepoView(self.make(tmp_path))
        assert view.unknown_paths("look at README.md and src/server.js") == []
        assert view.unknown_paths("see docs/design.md and `lib/db.py`") == ["docs/design.md", "lib/db.py"]
        assert view.unknown_paths("the /api/plants route, e.g. nothing") == []


# --------------------------------------------------------------- codex model plumbing


LOGIN_PRELUDE = """
import json, os, sys
if len(sys.argv) > 1 and sys.argv[1] == "login":
    home = os.environ["CODEX_HOME"]
    here = os.path.dirname(os.path.abspath(__file__))
    if sys.argv[2:3] == ["status"]:
        mode = json.load(open(home + "/auth.json")).get("auth_mode")
        print("Logged in using an API key - sk-***" if mode == "apikey" else "Logged in using ChatGPT")
    else:
        key = sys.stdin.read().strip()
        open(here + "/login.jsonl", "a").write(json.dumps(
            {"argv": sys.argv, "env": dict(os.environ), "stdin": key}) + "\\n")
        json.dump({"auth_mode": "apikey", "OPENAI_API_KEY": key}, open(home + "/auth.json", "w"))
        print("Successfully logged in")
    sys.exit(0)
"""

FAKE_CODEX = """#!PYTHON
""" + LOGIN_PRELUDE + """
args = sys.argv[1:]
prompt = sys.stdin.read()
last = args[args.index("-o") + 1]
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.jsonl"), "a").write(
    json.dumps({"args": args, "env": dict(os.environ), "cwd": os.getcwd(),
                "prompt": prompt}) + "\\n")
open(last, "w").write("answer: " + prompt[:12])
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "x"}}))
print(json.dumps({"type": "turn.completed", "model": "fake-model-1", "usage": {
    "input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 7}}))
"""


class TestCodexModel:
    def fake(self, tmp_path, monkeypatch):
        import sys
        exe = tmp_path / "codex"
        exe.write_text(FAKE_CODEX.replace("PYTHON", sys.executable))
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        auth = tmp_path / "auth.json"
        auth.write_text("{}")
        log = tmp_path / "log.jsonl"
        monkeypatch.setenv("SECRET_FROM_PARENT", "leak")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/worker/home")
        return CodexModel(home=tmp_path / "ch", codex_bin=str(exe), auth="chatgpt",
                          auth_source=auth, scratch_root=tmp_path), log

    def test_a_call_is_stateless_tool_free_and_environment_scrubbed(self, tmp_path, monkeypatch):
        model, log = self.fake(tmp_path, monkeypatch)
        reply = run(model.complete("hello world, prompt"))
        assert reply.text == "answer: hello world," and reply.output_tokens == 7
        row = json.loads(log.read_text().splitlines()[0])
        for flag in ("--ephemeral", "--skip-git-repo-check", "--json"):
            assert flag in row["args"]
        assert row["args"][row["args"].index("-s") + 1] == "read-only"
        assert row["args"][-1] == "-" and row["prompt"] == "hello world, prompt"
        assert "SECRET_FROM_PARENT" not in row["env"] and "CLAUDE_CONFIG_DIR" not in row["env"]
        assert row["env"]["CODEX_HOME"] == str(tmp_path / "ch")
        assert row["env"]["HOME"] == row["cwd"] or row["env"]["HOME"].endswith("/work")
        assert not Path(row["cwd"]).exists()                # the scratch dir is gone afterwards
        cfg = (tmp_path / "ch" / "config.toml").read_text()
        for feature in ("shell_tool", "unified_exec", "view_image", "hooks", "plugins"):
            assert f"{feature} = false" in cfg
        assert "mcp_servers" not in cfg and 'web_search = "disabled"' in cfg

    def test_two_calls_get_two_scratch_dirs(self, tmp_path, monkeypatch):
        model, log = self.fake(tmp_path, monkeypatch)
        run(model.complete("a"))
        run(model.complete("b"))
        cwds = [json.loads(ln)["cwd"] for ln in log.read_text().splitlines()]
        assert len(set(cwds)) == 2

    def test_a_timeout_kills_the_launcher_and_its_child(self, tmp_path, monkeypatch):
        import sys
        import time
        exe = tmp_path / "codex"
        exe.write_text(f"#!{sys.executable}\n" + LOGIN_PRELUDE + "import subprocess, time\n"
                       "subprocess.Popen(['sleep', '60'])\ntime.sleep(60)\n")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        auth = tmp_path / "auth.json"
        auth.write_text("{}")
        model = CodexModel(home=tmp_path / "ch", codex_bin=str(exe), auth="chatgpt",
                           auth_source=auth, scratch_root=tmp_path, timeout_s=1.0)
        started = time.monotonic()
        with pytest.raises(ModelError, match="timed out"):
            run(model.complete("x"))
        assert time.monotonic() - started < 10, "the wait returned once the group was killed"

    def test_a_tool_item_in_the_stream_fails_the_call(self):
        stream = json.dumps({"type": "item.completed", "item": {"type": "command_execution"}})
        with pytest.raises(ToolUseViolation):
            parse_stream(stream)
        with pytest.raises(ModelError):
            parse_stream('{"type":"turn.completed","usage":{}}')

    def test_config_has_no_tools_that_read_the_host(self):
        cfg = build_config("m", "low")
        assert 'model = "m"' in cfg and "shell_tool = false" in cfg


# ----------------------------------------------------------------------- the driver

SPEC = "Spec: a plant log."
BRIEF = "# Plant Log brief\nBuild it."

#: What the opener of K1 must carry to state its two rules.
K1_OPENER = ("start with plants. each plant has a light need, LOW, MEDIUM or BRIGHT, and is "
             "watered every 1 to 60 days. a nickname is optional, up to 24 characters, in "
             "italics after the species; a longer one is a 422 nickname_too_long. ask me if "
             "anything is unclear.")


class FakeBrowser:
    def __init__(self, reachable=True):
        self._reachable = reachable
        self.acts = []
        self.closed = False

    async def reachable(self):
        return self._reachable

    async def observe(self):
        return "URL: http://x/\nTITLE: app\nVISIBLE TEXT:\nPlants\nCONTROLS:\n  button \"Create\""

    async def act(self, action):
        self.acts.append(action)
        return "ok"

    async def screenshot(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG fake")
        self.shots = getattr(self, "shots", []) + [path]
        return path

    async def close(self):
        self.closed = True


def verdict(rows, flows=None, reqs=()):
    """An acceptance verdict: rows are (item, result, matches_deferred)."""
    return {"kind": "verdict", "action": {"type": "none", "target": "", "value": ""},
            "items": [{"item": i, "result": r, "evidence": "saw it", "matches_deferred": b}
                      for i, r, b in rows],
            "flows": [{"flow": "create a plant and open it", "status": "worked", "performed_action": True, "evidence": "ok"}]
            if flows is None else flows,
            "requirements": [{"id": i, "result": r, "evidence": "e", "matches_deferred": b}
                             for i, r, b in reqs]}


PLAN = {"lines": [], "flows": [{"name": "create a plant and open it", "goal": "make a plant"}]}


def make_handler(*, states, verdicts=None, answer=None, opener=K1_OPENER, log=None):
    """A fake operator model. `states` is the worker_state it judges on each `next` call."""
    states = list(states)
    verdicts = list(verdicts or [])

    def handler(purpose, prompt, schema):
        if log is not None:
            log.append((purpose, prompt))
        if purpose in ("opener", "opener_fix"):
            return {"worker_state": "working", "message": opener,
                    "reveals": ["P-SPECIES", "PREF-NICKNAME"]}
        if purpose.startswith("next"):
            s = states.pop(0) if states else "working"
            return {"worker_state": s, "message": "" if s == "reports_done" else "ok, keep going",
                    "reveals": []}
        if purpose == "accept":
            if verdicts:
                return verdicts.pop(0)
            return verdict([(i, "pass", "") for i in ACCEPT_K1])
        if purpose == "visual":
            return {"layout_ok": True, "reasons": "fine"}
        if purpose == "plan":
            return PLAN
        if purpose == "answer":
            return answer or {"answers": [{"question": "Which colour?", "answer": "blue"}],
                              "reveals": []}
        if purpose in ("feedback", "report_bug", "reveal_pass", "ask_where"):
            return {"worker_state": "working", "message": "the add plant thing is off",
                    "reveals": []}
        if purpose == "recollect":
            return "I wanted the plant page first. It went fine, I think."
        return "gist"
    return handler


ACCEPT_K1 = list(Agenda.load(AGENDA).key_idea("K1").acceptance)


def worker(turns, *, ask=None, junk="TOOLSECRET-9f3"):
    """A worker that answers each user_turn with the next (text, context_tokens) pair."""
    state = {"n": 0}
    turns = list(turns)

    async def on_frame(guest, frame):
        kind = frame["t"]
        if kind in ("user_turn", "answer"):
            if kind == "user_turn":
                state["n"] += 1
                if ask and state["n"] == 1:
                    await guest.send({"t": "question", "qid": "q1", "turn": 1, "questions": ask})
                    return
            n = state["n"]
            text, ctx = turns[min(n, len(turns)) - 1]
            await guest.send({"t": "tool_result", "turn": n, "content": junk, "tool_use_id": "t"})
            await guest.send({"t": "assistant", "turn": n, "text": text, "main_thread": True,
                              "tool_uses": [{"name": "Bash", "input": {"command": junk}}],
                              "context_tokens": ctx})
            await guest.send({"t": "turn_result", "turn": n, "subtype": "success",
                              "is_error": False, "total_cost_usd": 0.5 * n,
                              "context_tokens": ctx, "usage": {}})
        elif kind == "end_session":
            await guest.send({"t": "ended", "reason": "host", "turns": state["n"]})
    return on_frame


def drive(tmp_path, turns, handler, *, config=None, ask=None, browser=None, session=1,
          state=None, worktree=None, app_server=None):
    store = tmp_path / "store"
    sock = tmp_path / "r.sock"
    cfg = config or PrompterConfig(policy="continue", soft_tokens=400_000)
    fb = browser or FakeBrowser()
    model = FakeModel(handler)
    drv = build_driver(config=cfg, agenda_path=AGENDA, spec_path=_write(tmp_path, "spec.md", SPEC),
                       brief_path=_write(tmp_path, "brief.md", BRIEF), store_dir=store,
                       session=session, model=model, worktree_copy=worktree,
                       browser_factory=lambda url: fb)

    async def go():
        guest = FakeGuest(sock, on_frame=worker(turns, ask=ask),
                          app_ports={cfg.app_port: app_server} if app_server else None)
        await guest.start()
        try:
            async with await channel.Relay.connect(sock, wait_s=5, hello_s=5) as relay:
                return guest, await drv.run(relay)
        finally:
            await guest.stop()
    guest, record = run(go())
    return guest, record, drv, model, fb


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


class TestSessionDriver:
    def test_a_finished_key_idea_ends_on_the_acceptance_pass(self, tmp_path):
        guest, rec, drv, model, fb = drive(
            tmp_path, [("started plants", 50_000), ("plants are done", 90_000),
                       ("committed", 91_000)],
            make_handler(states=["working", "reports_done"]))
        sent = [f["text"] for f in guest.received if f["t"] == "user_turn"]
        assert sent[0].startswith("start with plants") and BRIEF in sent[0]
        assert "port 8000" in sent[0]
        assert sent[-1].startswith("that covers it") and guest.received[-1]["t"] == "end_session"
        assert rec["ending"]["rule"] == "key_idea_done" and not rec["ending"]["censored"]
        assert rec["key_idea_done"] and rec["acceptance"][0]["passed"]
        assert fb.closed
        state = CampaignState.load(tmp_path / "store" / "state.json")
        assert state.completed == ["K1"] and "P-SPECIES:reveal" in state.stated
        assert [(r["fact"], r["session"], r["turn"]) for r in rec["reveals"]] == [
            ("P-SPECIES", 1, 1), ("PREF-NICKNAME", 1, 1)]
        eff = rec["effort"]
        assert eff["operator_turns"] == 3 and eff["operator_output_tokens"] > 0
        assert rec["recollection"]["lint"]["passed"]
        stored = json.loads((tmp_path / "store" / "recollections.json").read_text())
        assert stored[0]["session"] == 1 and stored[0]["lint"]["passed"]
        assert (tmp_path / "store" / "sessions" / "s01.json").exists()

    def test_the_prompter_never_sees_tool_calls_or_results(self, tmp_path):
        seen = []
        drive(tmp_path, [("working", 1000), ("done", 2000), ("ok", 3000)],
              make_handler(states=["reports_done"], log=seen))
        assert seen and not any("TOOLSECRET" in p for _, p in seen)
        assert any("working" in p for _, p in seen)         # the worker's visible text is seen

    def test_soft_threshold_steers_to_a_break_and_the_key_idea_carries_over(self, tmp_path):
        log = []
        guest, rec, drv, *_ = drive(
            tmp_path, [("step one", 100_000), ("step two", 410_000), ("committed", 412_000)],
            make_handler(states=["working", "at_break"], log=log))
        assert rec["ending"]["rule"] == "soft_break" and not rec["ending"]["censored"]
        assert not rec["key_idea_done"]
        nxt = [p for purpose, p in log if purpose == "next"]
        assert "past the point where you stop adding work" not in nxt[0]
        assert "past the point where you stop adding work" in nxt[1]
        sent = [f["text"] for f in guest.received if f["t"] == "user_turn"]
        assert sent[-1].startswith("let's stop here")
        assert CampaignState.load(tmp_path / "store" / "state.json").completed == []
        assert rec["unstated_due"] == []

    def test_the_hard_ceiling_ends_the_session_and_marks_it_censored(self, tmp_path):
        guest, rec, *_ = drive(tmp_path, [("big", 850_000)],
                               make_handler(states=["working"]))
        assert rec["ending"]["rule"] == "hard_ceiling" and rec["ending"]["censored"]
        assert rec["thresholds"]["hard_tokens"] == 800_000     # default: 2 x soft
        assert [f["t"] for f in guest.received].count("user_turn") == 1

    def test_thresholds_are_configuration(self, tmp_path):
        cfg = PrompterConfig(soft_tokens=1000, hard_tokens=5000)
        guest, rec, *_ = drive(tmp_path, [("a", 6000)], make_handler(states=[]), config=cfg)
        assert rec["ending"]["rule"] == "hard_ceiling"
        with pytest.raises(ValueError):
            PrompterConfig(soft_tokens=10, hard_tokens=5)

    def test_the_turn_cap_ends_a_session_that_never_finishes(self, tmp_path):
        cfg = PrompterConfig(max_turns=2)
        _, rec, *_ = drive(tmp_path, [("a", 10), ("b", 20)], make_handler(states=[]), config=cfg)
        assert rec["ending"]["rule"] == "turn_cap" and rec["ending"]["censored"]

    def test_a_failed_checklist_goes_back_to_the_worker_and_a_pass_ends_it(self, tmp_path):
        fail = verdict([(i, "fail" if n == 1 else "pass", "") for n, i in enumerate(ACCEPT_K1)])
        guest, rec, *_ = drive(
            tmp_path, [("a", 1), ("done", 2), ("fixed, done", 3), ("committed", 4)],
            make_handler(states=["reports_done", "reports_done"], verdicts=[fail]))
        assert [a["passed"] for a in rec["acceptance"]] == [False, True]
        assert rec["acceptance"][0]["failures"][0]["result"] == "fail"
        assert rec["ending"]["rule"] == "key_idea_done" and rec["acceptance_rounds"] == 2

    def test_unreachable_app_is_reported_to_the_worker_not_failed(self, tmp_path):
        fb = FakeBrowser(reachable=False)
        _, rec, *_ = drive(tmp_path, [("a", 1), ("done", 2), ("up", 3), ("b", 4)],
                           make_handler(states=["reports_done", "reports_done", "reports_done",
                                                "reports_done"]),
                           browser=fb, config=PrompterConfig(max_unreachable=1))
        assert rec["ending"]["rule"] == "acceptance_exhausted" and rec["ending"]["censored"]

    def test_questions_are_answered_and_counted(self, tmp_path):
        ask = [{"question": "Which colour?", "options": [{"label": "red"}, {"label": "blue"}]}]
        guest, rec, *_ = drive(tmp_path, [("a", 1), ("done", 2), ("c", 3)],
                               make_handler(states=["reports_done"]), ask=ask)
        answer = next(f for f in guest.received if f["t"] == "answer")
        assert answer["answers"] == {"Which colour?": "blue"}
        assert rec["effort"]["askuserquestion_answers"] == 1
        assert rec["effort"]["askuserquestion_output_tokens"] > 0

    def test_a_reveal_missing_its_details_is_redrafted_once(self, tmp_path):
        log = []
        h = make_handler(states=["reports_done"], opener="start with plants, light matters",
                         log=log)

        def two_step(purpose, prompt, schema):
            if purpose == "opener":
                return {"worker_state": "working", "message": "start with plants",
                        "reveals": ["P-SPECIES"]}
            if purpose == "opener_fix":
                assert "leaves out" in prompt and "BRIGHT" in prompt
                return {"worker_state": "working", "message": K1_OPENER,
                        "reveals": ["P-SPECIES", "PREF-NICKNAME"]}
            return h(purpose, prompt, schema)
        _, rec, *_ = drive(tmp_path, [("a", 1), ("b", 2)], two_step)
        assert {r["fact"] for r in rec["reveals"]} == {"P-SPECIES", "PREF-NICKNAME"}

    def test_facts_still_owed_are_stated_before_the_key_idea_is_judged(self, tmp_path):
        def h(purpose, prompt, schema):
            if purpose in ("opener", "opener_fix"):
                return {"worker_state": "working", "message": "start with plants",
                        "reveals": []}
            if purpose == "reveal_pass":
                return {"worker_state": "working", "message": K1_OPENER,
                        "reveals": ["P-SPECIES", "PREF-NICKNAME"]}
            return make_handler(states=["reports_done", "reports_done"])(purpose, prompt, schema)
        guest, rec, *_ = drive(tmp_path, [("a", 1), ("done", 2), ("done again", 3), ("c", 4)], h)
        kinds = [(r["fact"], r["via"]) for r in rec["reveals"]]
        assert ("P-SPECIES", "reveal_pass") in kinds
        assert rec["ending"]["rule"] == "key_idea_done"

    def test_a_deferred_bug_found_in_click_through_is_reported_once_and_left(self, tmp_path):
        agenda_state = CampaignState.load(tmp_path / "store" / "state.json")
        agenda_state.completed = ["K1", "K2"]
        agenda_state.stated = {f"{d.fact}:{d.kind}": {"session": 1, "turn": 1}
                               for d in Agenda.load(AGENDA).due("K3")}
        agenda_state.save()
        k3 = Agenda.load(AGENDA).key_idea("K3")
        v = verdict([(i, "fail" if "right way up" in i else "pass",
                      "DB-PHOTO-ROTATE" if "right way up" in i else "") for i in k3.acceptance])
        h = make_handler(states=["reports_done"], verdicts=[v])
        _, rec, *_ = drive(tmp_path, [("a", 1), ("done", 2), ("c", 3)], h, session=3)
        assert rec["key_idea"] == "K3" and rec["key_idea_done"]
        assert rec["acceptance"][0]["deferred_matched"] == ["DB-PHOTO-ROTATE"]
        state = CampaignState.load(tmp_path / "store" / "state.json")
        assert state.deferred["DB-PHOTO-ROTATE"] == "reported" and "K3" in state.completed
        assert ("DB-PHOTO-ROTATE", "report") in [(r["fact"], r["kind"]) for r in rec["reveals"]]

    def test_a_conditional_bug_that_did_not_occur_is_marked_not_triggered(self, tmp_path):
        state = CampaignState.load(tmp_path / "store" / "state.json")
        state.completed = ["K1", "K2"]
        state.stated = {f"{d.fact}:{d.kind}": {} for d in Agenda.load(AGENDA).due("K3")}
        state.save()
        k3 = Agenda.load(AGENDA).key_idea("K3")
        v = verdict([(i, "pass", "") for i in k3.acceptance])
        drive(tmp_path, [("a", 1), ("done", 2), ("c", 3)],
              make_handler(states=["reports_done"], verdicts=[v]), session=3)
        assert CampaignState.load(tmp_path / "store" / "state.json").deferred == {
            "DB-PHOTO-ROTATE": "not-triggered"}

    def test_report_only_mode_is_recorded_when_there_is_no_browser(self, tmp_path):
        store = tmp_path / "store"
        sock = tmp_path / "r.sock"
        drv = build_driver(config=PrompterConfig(dry_run=True), agenda_path=AGENDA,
                           spec_path=_write(tmp_path, "s.md", SPEC),
                           brief_path=_write(tmp_path, "b.md", BRIEF), store_dir=store,
                           session=1, model=FakeModel(make_handler(states=["reports_done"])))

        async def go():
            guest = FakeGuest(sock, on_frame=worker([("a", 1), ("done", 2), ("c", 3)]))
            await guest.start()
            try:
                async with await channel.Relay.connect(sock, wait_s=5, hello_s=5) as relay:
                    return await drv.run(relay)
            finally:
                await guest.stop()
        rec = run(go())
        assert rec["acceptance"][0]["mode"].startswith("report-only") and rec["key_idea_done"]
        assert rec["real_run"] is False and "DRY RUN" in rec["stamp"]
        state = CampaignState.load(store / "state.json")
        with pytest.raises(DryRunError):
            state.require_real()


class TestPolicies:
    def test_prompt_carries_the_policy_and_only_self_contained_sees_the_repo(self, tmp_path):
        root = tmp_path / "wt"
        root.mkdir()
        (root / "README.md").write_text("# the app\n")
        for policy in ("continue", "self-contained"):
            log = []
            sub = tmp_path / policy
            sub.mkdir()
            drive(sub, [("a", 1), ("done", 2), ("c", 3)],
                  make_handler(states=["reports_done"], log=log),
                  config=PrompterConfig(policy=policy), session=2 if False else 1)
        # session 2 so a worktree copy applies
        logs = {}
        for policy in ("continue", "self-contained"):
            sub = tmp_path / f"s2-{policy}"
            sub.mkdir()
            lg = []
            st = CampaignState.load(sub / "store" / "state.json")
            st.completed = ["K1"]
            st.save()
            drive(sub, [("a", 1), ("done", 2), ("c", 3)],
                  make_handler(states=["reports_done"], log=lg, opener="let's go on with it"),
                  config=PrompterConfig(policy=policy), session=2, worktree=root)
            logs[policy] = next(p for purpose, p in lg if purpose == "opener")
        assert "BRIEFING STYLE (continue)" in logs["continue"]
        assert "REPOSITORY VIEW" not in logs["continue"]
        assert "BRIEFING STYLE (self-contained)" in logs["self-contained"]
        assert "REPOSITORY VIEW" in logs["self-contained"] and "README.md" in logs["self-contained"]

    def test_the_ask_invitation_is_optional_and_can_be_switched_off(self, tmp_path):
        on, off = [], []
        drive(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path,
              [("a", 1), ("d", 2), ("c", 3)], make_handler(states=["reports_done"], log=on))
        (tmp_path / "y").mkdir()
        drive(tmp_path / "y", [("a", 1), ("d", 2), ("c", 3)],
              make_handler(states=["reports_done"], log=off),
              config=PrompterConfig(invite_ask=False))
        assert "AskUserQuestion" in on[0][1] and "AskUserQuestion" not in off[0][1]

    def test_a_self_contained_opener_citing_a_missing_file_is_redrafted(self, tmp_path):
        root = tmp_path / "wt"
        root.mkdir()
        (root / "README.md").write_text("# the app\n")
        st = CampaignState.load(tmp_path / "store" / "state.json")
        st.completed = ["K1"]
        st.save()
        seen = []

        def h(purpose, prompt, schema):
            seen.append(purpose)
            if purpose == "opener":
                return {"worker_state": "working", "message": "see docs/plan.md", "reveals": []}
            if purpose == "opener_fix":
                assert "docs/plan.md" in prompt
                return {"worker_state": "working", "message": "see README.md", "reveals": []}
            return make_handler(states=["reports_done"])(purpose, prompt, schema)
        guest, *_ = drive(tmp_path, [("a", 1), ("d", 2), ("c", 3)], h,
                          config=PrompterConfig(policy="self-contained"), session=2, worktree=root)
        assert "opener_fix" in seen
        assert [f["text"] for f in guest.received if f["t"] == "user_turn"][0].startswith("see README.md")


class TestIntegration:
    def test_hard_token_details_of_a_rule(self):
        toks = hard_tokens_of('LOW, MEDIUM or BRIGHT; 1.5 means 1500; a 422 nickname_too_long; "Log watering"')
        assert {"LOW", "BRIGHT", "1.5", "1500", "422", "nickname_too_long", "Log watering"} <= set(toks)

    def test_conversation_plan_declares_the_app_port(self, tmp_path):
        plan = conversation_plan(
            config=PrompterConfig(app_port=8123, dry_run=True), agenda_path=AGENDA,
            spec_path=_write(tmp_path, "s.md", SPEC), brief_path=_write(tmp_path, "b.md", BRIEF),
            store_dir=tmp_path / "store", session=1, model=FakeModel(lambda *a: ""))
        assert plan.app_ports == [8123] and plan.tunnels == []
        assert inspect.iscoroutinefunction(plan.driver.run)

    def test_conversation_plan_refuses_a_worktree_with_transcripts_before_the_cell(self, tmp_path):
        root = tmp_path / "wt"
        root.mkdir()
        (root / "a.jsonl").write_text("{}")
        with pytest.raises(RepoRefused):
            conversation_plan(
                config=PrompterConfig(policy="self-contained"), agenda_path=AGENDA,
                spec_path=_write(tmp_path, "s.md", SPEC),
                brief_path=_write(tmp_path, "b.md", BRIEF), store_dir=tmp_path / "store",
                session=2, model=FakeModel(lambda *a: ""), worktree_copy=root)

    def test_reconcile_marks_skipped_items_unverifiable(self):
        rows, matched = reconcile(("a", "b"), [{"item": "a", "result": "pass", "evidence": "",
                                                "matches_deferred": ""}], set())
        assert [r["result"] for r in rows] == ["pass", "unverifiable"] and matched == []


def scripted_accept(steps):
    """A handler for `accept` that plays the given replies in order, then a clean verdict."""
    steps = list(steps)

    def handler(purpose, prompt, schema):
        if purpose == "plan":
            return PLAN
        if steps:
            return steps.pop(0)
        return verdict([(i, "pass", "") for i in ACCEPT_K1])
    return handler


def act(kind, target="", value=""):
    return {"kind": "act", "action": {"type": kind, "target": target, "value": value},
            "items": [], "flows": [], "requirements": []}


class TestAcceptanceStandard:
    def accept(self, tmp_path, steps, visual=None, reqs=(), browser=None, plan=None, **kw):
        from thalamus_eval.prompter.acceptance import run_acceptance
        key = Agenda.load(AGENDA).key_idea("K1")
        handler = scripted_accept(steps)
        vis = visual or {"layout_ok": True, "reasons": "fine"}
        plan = plan or PLAN
        model = FakeModel(lambda p, pr, sc: vis if p == "visual" else plan if p == "plan"
                          else handler(p, pr, sc))
        b = browser or FakeBrowser()
        res = run(run_acceptance(Metered(model), b, key, [], requirements=list(reqs),
                                 shot_dir=tmp_path / "shots", **kw))
        return res, model, b

    def test_a_round_needs_a_flow_a_screenshot_and_a_good_look(self, tmp_path):
        res, model, b = self.accept(tmp_path, [act("click", "Create"), act("look")])
        assert res.passed and res.flows and len(res.screenshots) == 1
        assert Path(res.screenshots[0]).exists()
        # the screenshot is what the next acceptance call and the visual call are given
        assert [imgs for p, imgs in model.images if p == "accept" and imgs]
        assert [imgs for p, imgs in model.images if p == "visual"][0] == [Path(res.screenshots[0])]

    def test_a_verdict_with_no_look_gets_a_screenshot_taken_anyway(self, tmp_path):
        res, _, b = self.accept(tmp_path, [])
        assert res.passed and len(res.screenshots) == 1 and res.screenshots[0].endswith("auto.png")

    def test_no_flow_exercised_is_inconclusive_not_a_failure(self, tmp_path):
        res, *_ = self.accept(tmp_path, [verdict([(i, "pass", "") for i in ACCEPT_K1], flows=[])])
        assert not res.passed and res.inconclusive and not res.failures

    def test_a_flow_that_did_not_work_fails_the_round_even_if_the_checklist_passes(self, tmp_path):
        bad = verdict([(i, "pass", "") for i in ACCEPT_K1],
                      flows=[{"flow": "open the plant", "status": "failed", "evidence": "404"}])
        res, *_ = self.accept(tmp_path, [bad])
        assert not res.passed and res.failures[0]["item"] == "flow: create a plant and open it"

    def test_lines_a_browser_cannot_check_and_lines_never_seen_do_not_fail_the_round(self, tmp_path):
        rows = [(ACCEPT_K1[0], "unverifiable", ""), (ACCEPT_K1[1], "pass", ""),
                (ACCEPT_K1[2], "unverifiable", "")]
        plan = {**PLAN, "lines": [{"n": 1, "browser": False}]}
        res, *_ = self.accept(tmp_path, [verdict(rows)], plan=plan)
        assert res.passed and not res.failures and not res.inconclusive
        assert res.not_browser == [ACCEPT_K1[0]] and res.unverified == [ACCEPT_K1[2], ACCEPT_K1[3]]
        assert res.items[0]["result"] == "not_checked_in_browser"

    def test_an_unreachable_flow_makes_the_round_inconclusive(self, tmp_path):
        v = verdict([(i, "unverifiable", "") for i in ACCEPT_K1],
                    flows=[{"flow": "create a plant and open it", "status": "unreachable",
                            "evidence": "no button for it"}])
        res, *_ = self.accept(tmp_path, [v])
        assert not res.passed and res.inconclusive and res.unreachable == [
            "create a plant and open it"]

    def test_each_flow_has_its_own_step_budget(self, tmp_path):
        plan = {"lines": [], "flows": [{"name": "hard", "goal": "g"}, {"name": "easy", "goal": "g"}]}
        calls = []

        def handler(purpose, prompt, schema):
            calls.append(prompt)
            if "THIS FLOW: hard" in prompt:
                return act("click", "Create")                 # never settles
            return verdict([(i, "pass", "") for i in ACCEPT_K1],
                           flows=[{"flow": "easy", "status": "worked", "evidence": "ok"}])
        from thalamus_eval.prompter.acceptance import run_acceptance
        key = Agenda.load(AGENDA).key_idea("K1")
        model = FakeModel(lambda p, pr, sc: {"layout_ok": True, "reasons": ""} if p == "visual"
                          else plan if p == "plan" else handler(p, pr, sc))
        res = run(run_acceptance(Metered(model), FakeBrowser(), key, [], max_steps=3,
                                 shot_dir=tmp_path / "s"))
        hard = [c for c in calls if "THIS FLOW: hard" in c]
        assert len(hard) == 4 and "out of steps for this flow" in hard[-1]
        assert [f["status"] for f in res.flows] == ["unreachable", "worked"] and res.inconclusive

    def test_a_broken_layout_fails_the_round_with_the_reasons(self, tmp_path):
        res, *_ = self.accept(tmp_path, [], visual={"layout_ok": False,
                                                    "reasons": "button text is invisible"})
        assert not res.passed
        assert res.failures[0] == {"item": "how it looks", "result": "fail",
                                   "evidence": "button text is invisible", "matches_deferred": ""}
        assert res.visual["layout_ok"] is False

    def test_stated_requirements_are_checked_and_a_failed_one_fails_the_round(self, tmp_path):
        v = verdict([(i, "pass", "") for i in ACCEPT_K1],
                    reqs=[("PREF-NICKNAME", "fail", ""), ("P-SPECIES", "not_observable", "")])
        res, model, _ = self.accept(tmp_path, [v], reqs=[("PREF-NICKNAME", "names are trimmed")])
        assert not res.passed and res.failures[0]["item"] == "requirement PREF-NICKNAME"
        assert "id=PREF-NICKNAME: names are trimmed" in model.prompts[1][1]

    def test_requirements_come_from_what_was_stated_at_the_latest_version(self):
        a = Agenda.load(AGENDA)
        stated = {"RC-UNITS:reveal": {"key_idea": "K2"}, "RC-UNITS:change": {"key_idea": "K5"},
                  "P-AMOUNT:reveal": {"key_idea": "K2"}, "DB-PHOTO-ROTATE:report": {}}
        reqs = dict(a.requirements("K5", stated))
        assert "cups when under a litre" in reqs["RC-UNITS"] and "P-AMOUNT" in reqs
        assert "DB-PHOTO-ROTATE" not in reqs
        assert "P-AMOUNT" not in dict(a.requirements("K7", stated))

    def test_the_driver_records_flows_screenshots_and_reasons_per_round(self, tmp_path):
        guest, rec, drv, model, fb = drive(
            tmp_path, [("a", 1), ("done", 2), ("c", 3)], make_handler(states=["reports_done"]))
        r = rec["acceptance"][0]
        assert r["flows"] and r["screenshots"] and r["visual"]["layout_ok"] is True
        assert "shots/s01-r1" in r["screenshots"][0] and Path(r["screenshots"][0]).exists()

    def test_a_failed_visual_is_taken_back_to_the_worker_until_the_rounds_run_out(self, tmp_path):
        def h(purpose, prompt, schema):
            if purpose == "visual":
                return {"layout_ok": False, "reasons": "text overlaps the buttons"}
            return make_handler(states=["reports_done"] * 5)(purpose, prompt, schema)
        guest, rec, *_ = drive(tmp_path, [("a", 1), ("done", 2), ("fixed", 3), ("again", 4)],
                               h, config=PrompterConfig(max_acceptance_rounds=2))
        assert rec["ending"]["rule"] == "acceptance_exhausted" and rec["ending"]["censored"]
        assert rec["acceptance_rounds"] == 2 and not rec["key_idea_done"]
        assert all("overlaps" in r["failures"][0]["evidence"] for r in rec["acceptance"])


class TestRealRunStandard:
    def kwargs(self, tmp_path):
        return dict(agenda_path=AGENDA, spec_path=_write(tmp_path, "s.md", SPEC),
                    brief_path=_write(tmp_path, "b.md", BRIEF), store_dir=tmp_path / "store",
                    session=1, model=FakeModel(lambda *a: ""))

    def test_a_real_run_without_a_browser_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="browser"):
            build_driver(config=PrompterConfig(), **self.kwargs(tmp_path))

    def test_dry_run_and_a_browser_are_contradictory(self, tmp_path):
        with pytest.raises(ValueError):
            build_driver(config=PrompterConfig(dry_run=True), browser_factory=lambda u: None,
                         **self.kwargs(tmp_path))

    def test_a_store_is_never_both_kinds_of_run(self, tmp_path):
        guest, rec, *_ = drive(tmp_path, [("a", 1), ("d", 2), ("c", 3)],
                               make_handler(states=["reports_done"]))
        assert rec["real_run"] is True and rec["stamp"] is None
        with pytest.raises(DryRunError):
            build_driver(config=PrompterConfig(dry_run=True), **self.kwargs(tmp_path))


class TestMidTurnCeiling:
    def test_the_hard_ceiling_interrupts_a_turn_in_flight(self, tmp_path):
        async def on_frame(g, f):
            if f["t"] == "user_turn":
                await g.send({"t": "assistant", "turn": 1, "text": "grinding", "main_thread": True,
                              "tool_uses": [], "context_tokens": 300_000})
                await g.send({"t": "assistant", "turn": 1, "text": "still", "main_thread": True,
                              "tool_uses": [], "context_tokens": 820_000})
                # no turn_result: the turn is still running until the host interrupts it
            elif f["t"] == "end_session":
                await g.send({"t": "ended", "reason": "host", "turns": 0})

        store, sock = tmp_path / "store", tmp_path / "r.sock"
        drv = build_driver(config=PrompterConfig(), agenda_path=AGENDA,
                           spec_path=_write(tmp_path, "s.md", SPEC),
                           brief_path=_write(tmp_path, "b.md", BRIEF), store_dir=store,
                           session=1, model=FakeModel(make_handler(states=[])),
                           browser_factory=lambda u: FakeBrowser())

        async def go():
            guest = FakeGuest(sock, on_frame=on_frame)
            await guest.start()
            try:
                async with await channel.Relay.connect(sock, wait_s=5, hello_s=5) as relay:
                    return guest, await drv.run(relay)
            finally:
                await guest.stop()
        guest, rec = run(go())
        assert rec["ending"]["rule"] == "hard_ceiling" and rec["ending"]["censored"]
        assert rec["ending"]["context_tokens"] == 820_000 and rec["ending"]["worker_turns"] == 0
        assert [f["t"] for f in guest.received].count("end_session") == 1

    def test_the_soft_break_still_waits_for_a_turn_boundary(self, tmp_path):
        guest, rec, *_ = drive(
            tmp_path, [("a", 100_000), ("b", 450_000), ("c", 451_000)],
            make_handler(states=["working", "at_break"]))
        assert rec["ending"]["rule"] == "soft_break"


class TestCodexImagesAndHome:
    def test_images_are_attached_and_the_home_is_private(self, tmp_path, monkeypatch):
        t = TestCodexModel()
        model, log = t.fake(tmp_path, monkeypatch)
        img = tmp_path / "shot.png"
        img.write_bytes(b"x")
        run(model.complete("look", images=[img]))
        row = json.loads(log.read_text().splitlines()[0])
        assert row["args"][row["args"].index("-i") + 1] == str(img)
        assert stat.S_IMODE((tmp_path / "ch").stat().st_mode) == 0o700
        assert stat.S_IMODE((tmp_path / "ch" / "auth.json").stat().st_mode) == 0o600

    def test_a_home_inside_a_copied_directory_is_refused(self, tmp_path):
        wt = tmp_path / "worktree"
        wt.mkdir()
        (tmp_path / "a.json").write_text("{}")
        with pytest.raises(ModelError, match="copied into a cell"):
            CodexModel(home=wt / "codex-home", auth="chatgpt", auth_source=tmp_path / "a.json",
                       forbidden_parents=(wt,))



def test_acceptance_is_shown_what_the_worker_last_said(tmp_path):
    seen = []
    drive(tmp_path, [("started plants", 50_000), ("open the fern's page", 90_000),
                     ("committed", 91_000)],
          make_handler(states=["working", "reports_done"], log=seen))
    accept = [p for purpose, p in seen if purpose == "accept"]
    assert accept and "WHAT THE ASSISTANT JUST TOLD YOU" in accept[0]
    assert "open the fern's page" in accept[0]


def test_observe_looks_again_when_a_click_navigates_mid_read():
    from thalamus_eval.prompter.browser import PlaywrightBrowser

    class Page:
        url = "http://x/"
        reads = 0

        async def wait_for_load_state(self, *a, **k): ...
        async def wait_for_timeout(self, ms): ...
        async def title(self): return "t"

        async def inner_text(self, sel):
            self.reads += 1
            if self.reads == 1:
                raise RuntimeError("Page.evaluate: Execution context was destroyed, most likely "
                                   "because of a navigation")
            return "Hello"

        async def evaluate(self, js): return []

    b = PlaywrightBrowser("http://x")
    b._page = Page()
    assert "Hello" in run(b.observe()) and b._page.reads == 2


def test_an_unreachable_flow_asks_the_worker_where_and_reports_no_defect(tmp_path):
    seen = []
    unreachable = verdict([(i, "unverifiable", "") for i in ACCEPT_K1],
                          flows=[{"flow": "x", "status": "unreachable", "evidence": "no button"}])
    guest, rec, *_ = drive(
        tmp_path, [("a", 1), ("done", 2), ("it is under Settings", 3), ("ok", 4)],
        make_handler(states=["reports_done", "reports_done"], verdicts=[unreachable], log=seen))
    ask = [p for purpose, p in seen if purpose == "ask_where"]
    assert ask and "no button" in ask[0]
    assert not [1 for purpose, _ in seen if purpose == "feedback"]
    assert rec["acceptance"][0]["inconclusive"] and rec["acceptance"][1]["passed"]
    assert rec["key_idea_done"]


class TestFlowOutcomes:
    def flow_verdict(self, status, performed, evidence="x"):
        return verdict([(i, "unverifiable", "") for i in ACCEPT_K1],
                       flows=[{"flow": "f", "status": status, "performed_action": performed,
                               "evidence": evidence}])

    def accept(self, tmp_path, v):
        return TestAcceptanceStandard().accept(tmp_path, [v])[0]

    def test_a_control_that_could_not_be_found_is_inconclusive(self, tmp_path):
        res = self.accept(tmp_path, self.flow_verdict("unreachable", False))
        assert res.inconclusive and not res.failures and not res.passed

    def test_an_action_with_no_visible_result_is_a_failure_even_if_the_page_looks_fine(self, tmp_path):
        res = self.accept(tmp_path, self.flow_verdict("unreachable", True, "clicked Create, nothing"))
        assert not res.inconclusive and not res.passed and res.visual["layout_ok"]
        assert res.failures[0]["item"].startswith("flow:") and "nothing" in res.failures[0]["evidence"]
        assert res.flows[0]["status"] == "failed"

    def test_a_flow_the_app_got_wrong_is_a_failure(self, tmp_path):
        res = self.accept(tmp_path, self.flow_verdict("failed", True))
        assert not res.inconclusive and res.failures

    def test_a_dead_button_goes_back_to_the_worker_as_a_defect_not_as_a_question(self, tmp_path):
        seen = []
        dead = self.flow_verdict("unreachable", True, "clicked Create, nothing happened")
        drive(tmp_path, [("a", 1), ("done", 2), ("fixed", 3), ("ok", 4)],
              make_handler(states=["reports_done", "reports_done"], verdicts=[dead], log=seen))
        purposes = [p for p, _ in seen]
        assert "feedback" in purposes and "ask_where" not in purposes


class TestHangs:
    def fake(self, tmp_path, script):
        import sys
        exe = tmp_path / "codex"
        exe.write_text(f"#!{sys.executable}\n" + LOGIN_PRELUDE + script)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        auth = tmp_path / "auth.json"
        auth.write_text("{}")
        return CodexModel(home=tmp_path / "ch", codex_bin=str(exe), auth="chatgpt",
                          auth_source=auth, scratch_root=tmp_path, timeout_s=1.5)

    def test_a_hung_call_is_killed_asked_again_and_counted(self, tmp_path):
        marker = tmp_path / "first"
        script = (f"import json, os, sys, time\nsys.stdin.read()\n"
                  f"if not os.path.exists({str(marker)!r}):\n    open({str(marker)!r}, 'w').write('x')\n"
                  "    print(json.dumps({'type': 'thread.started'}), flush=True)\n    time.sleep(60)\n"
                  "last = sys.argv[sys.argv.index('-o') + 1]\nopen(last, 'w').write('answer')\n"
                  "print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'answer'}}))\n"
                  "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 5, 'output_tokens': 1}}))\n")
        model = Metered(self.fake(tmp_path, script))
        reply = run(model.complete("p"))
        assert reply.text == "answer" and reply.hangs == 1 and model.hangs == 1
        assert "thread.started" in reply.hang_notes[0] and "stdout_bytes" in reply.hang_notes[0]

    def test_a_call_that_hangs_twice_fails_with_the_diagnosis(self, tmp_path):
        from thalamus_eval.prompter.model import ModelTimeout
        model = Metered(self.fake(tmp_path, "import sys, time\nsys.stdin.read()\ntime.sleep(60)\n"))
        with pytest.raises(ModelTimeout) as exc:
            run(model.complete("p"))
        assert exc.value.hangs == 2 and model.hangs == 2 and "prompt_chars" in exc.value.diagnostic


class TestApiKeyLogin:
    KEY = "sk-test-0123456789"

    def make(self, tmp_path, mode=0o600, key=None, **kw):
        import sys
        exe = tmp_path / "codex"
        exe.write_text(FAKE_CODEX.replace("PYTHON", sys.executable))
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        keyfile = tmp_path / "key"
        if key is not False:
            keyfile.write_text((key or self.KEY) + "\n")
            keyfile.chmod(mode)
        return CodexModel(home=tmp_path / "ch", codex_bin=str(exe), api_key_file=keyfile,
                          scratch_root=tmp_path, **kw)

    def test_the_key_goes_in_on_stdin_only_and_the_mode_is_read_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-caller-env")
        model = self.make(tmp_path)
        assert model.auth_mode == "api-key"
        row = json.loads((tmp_path / "login.jsonl").read_text().splitlines()[0])
        assert row["stdin"] == self.KEY and row["argv"][1:] == ["login", "--with-api-key"]
        assert self.KEY not in " ".join(row["argv"]) and self.KEY not in json.dumps(row["env"])
        assert "sk-caller-env" not in json.dumps(row["env"])

    def test_a_missing_or_loose_key_file_refuses_to_start(self, tmp_path):
        with pytest.raises(ModelError, match="no OpenAI API key file"):
            self.make(tmp_path, key=False)
        with pytest.raises(ModelError, match="0600"):
            self.make(tmp_path, mode=0o644)

    def test_chatgpt_is_an_explicit_option_and_is_recorded_as_such(self, tmp_path):
        import sys
        exe = tmp_path / "codex"
        exe.write_text(FAKE_CODEX.replace("PYTHON", sys.executable))
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        (tmp_path / "a.json").write_text("{}")
        model = CodexModel(home=tmp_path / "ch", codex_bin=str(exe), auth="chatgpt",
                           auth_source=tmp_path / "a.json")
        assert model.auth_mode == "chatgpt" and not (tmp_path / "login.jsonl").exists()

    def test_a_login_that_is_not_the_requested_mode_is_refused(self, tmp_path):
        import sys
        exe = tmp_path / "codex"
        exe.write_text(f"#!{sys.executable}\nimport sys\nif sys.argv[2:3] == ['status']:\n"
                       "    print('Logged in using ChatGPT')\nelse:\n    sys.stdin.read()\n")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        key = tmp_path / "key"
        key.write_text("sk-x\n")
        key.chmod(0o600)
        with pytest.raises(ModelError, match="not logged in with api-key"):
            CodexModel(home=tmp_path / "ch", codex_bin=str(exe), api_key_file=key)

    def test_tokens_cached_tokens_and_model_are_recorded_per_call_and_summed(self, tmp_path):
        model = Metered(self.make(tmp_path))
        run(model.complete("one", purpose="plan"))
        run(model.complete("two", purpose="accept"))
        assert [(r["input_tokens"], r["cached_input_tokens"], r["output_tokens"], r["model"])
                for r in model.log] == [(100, 40, 7, "fake-model-1")] * 2
        use = model.usage()
        assert (use["calls"], use["input_tokens"], use["cached_input_tokens"],
                use["output_tokens"]) == (2, 200, 80, 14)
        assert use["models"]["fake-model-1"]["calls"] == 2 and use["auth_mode"] == "api-key"
        assert model.usage(since=1)["calls"] == 1
        assert model.totals()["plan"]["cached_input_tokens"] == 40

    def test_a_store_does_not_mix_login_modes(self, tmp_path):
        class Modal(FakeModel):
            auth_mode = "api-key"

        def build(mode):
            m = Modal(lambda *a: "")
            m.auth_mode = mode
            return build_driver(
                config=PrompterConfig(dry_run=True), agenda_path=AGENDA,
                spec_path=_write(tmp_path, "s.md", SPEC), brief_path=_write(tmp_path, "b.md", BRIEF),
                store_dir=tmp_path / "store", session=1, model=m)
        build("api-key").state.save()
        assert CampaignState.load(tmp_path / "store" / "state.json").auth_mode == "api-key"
        from thalamus_eval.prompter.agenda import AuthModeError
        with pytest.raises(AuthModeError, match="one store is one mode"):
            build("chatgpt")

    def test_the_session_record_and_the_chain_summary_carry_the_counts(self, tmp_path):
        from thalamus_eval.chain import prompter_usage
        guest, rec, drv, *_ = drive(tmp_path, [("a", 1), ("done", 2), ("c", 3)],
                                    make_handler(states=["reports_done"]))
        assert rec["usage"]["calls"] > 0 and "input_tokens" in rec["usage"]
        assert rec["acceptance"][0]["usage"]["calls"] > 0
        total = prompter_usage([{"conversation": {"driver_result": rec}}] * 2)
        assert total["calls"] == 2 * rec["usage"]["calls"]

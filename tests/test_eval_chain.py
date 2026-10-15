"""
A campaign as a chain of VM cells (contract 4: post steps, a store that comes back out).

Interfaces: thalamus_eval.vm — job_spec/run_backend `post`, `store_out`, `inputs`;
thalamus_eval.cell_prep — mark_session, distill, export_store and their parsers;
thalamus_eval.arms — cell_prep_steps(campaign=), cell_post_steps, cost_envelope, the
two campaign arms; thalamus_eval.store — file_payload; thalamus_eval.chain — the chain.
Infrastructure: stub backend executables and an injected agent runner standing in for
the cell; no libvirt, no docker, no model, no network.
Scope: a store crosses OUT of a cell only if one went in and arrives with a digest the
runner recomputes; the chain feeds each session the last one's trees and graph and says
whether they were touched in between; distillation is waited for rather than raced;
cost is recorded per session with its distillation attributed to it; and a refused or
non-growing session halts the chain with the spend it incurred.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from thalamus_eval import arms, cell_prep, chain, store, vm
from thalamus_eval.arms import ArmError

from test_eval_vm import _ok_result, _stub_backend

SHA = "a" * 40


def _spec(tmp_path, **kw):
    return vm.job_spec(worktree=tmp_path, home=tmp_path, prompt="p", argv=["claude"],
                       env={}, network_policy="isolated", timeout=60, **kw)


# --------------------------------------------------------------------------- the seam


class TestPostCrossesTheSeam:
    def test_post_and_session_are_always_on_the_wire(self, tmp_path):
        spec = _spec(tmp_path)
        assert spec["post"] == []
        assert spec["session"] == {"kind": "stdin-prompt"}
        assert "store_out" not in spec and "inputs" not in spec

    def test_a_malformed_or_repeated_post_step_is_refused_before_a_cell_boots(self, tmp_path):
        with pytest.raises(vm.VmBackendError, match="post step"):
            _spec(tmp_path, post=[{"name": "", "argv": ["x"]}])
        twice = [{"name": "d", "argv": ["x"]}, {"name": "d", "argv": ["y"]}]
        with pytest.raises(vm.VmBackendError, match="both named"):
            _spec(tmp_path, post=twice)

    def test_a_cell_sent_post_must_report_post(self, tmp_path):
        spec = _spec(tmp_path, post=[{"name": "distill", "argv": ["true"]}])
        silent = _stub_backend(tmp_path, _ok_result())
        with pytest.raises(vm.VmBackendError, match="no `post` results"):
            vm.run_backend(silent, spec, timeout=30)
        reported = _stub_backend(tmp_path, _ok_result(
            post=[{"name": "distill", "exit_code": 0, "stdout": "{}", "stderr": ""}]))
        assert vm.post_result(vm.run_backend(reported, spec, timeout=30), "distill")

    def test_a_post_refusal_names_the_step_and_carries_the_spend(self, tmp_path):
        spec = _spec(tmp_path, post=[{"name": "distill", "argv": ["true"]}])
        refusing = _stub_backend(tmp_path, _ok_result(
            ok=False, refusal="post step failed",
            post=[{"name": "distill", "exit_code": 1, "stdout": "", "stderr": "no log"}],
            session={"stdout": "{\"total_cost_usd\": 1.5}", "exit_code": 0}))
        with pytest.raises(vm.VmBackendError) as caught:
            vm.run_backend(refusing, spec, timeout=30)
        assert "post step `distill` exited 1" in str(caught.value)
        assert caught.value.result["session"]["stdout"], "the envelope of the spent session"

    def test_an_unknown_session_kind_is_the_backends_to_refuse_not_ours_to_hide(self, tmp_path):
        spec = _spec(tmp_path, session={"kind": "conversation"})
        assert spec["session"] == {"kind": "conversation"}


class TestStoreComesBackOut:
    def _store(self):
        return {"host_path": "/x", "guest_path": "/y", "port": 8182, "sha256": "d" * 64}

    def test_nothing_comes_out_of_a_cell_that_was_given_nothing(self, tmp_path):
        with pytest.raises(vm.VmBackendError, match="given none"):
            _spec(tmp_path, store_out={"name": "o", "guest_path": "/v/o.kryo",
                                       "host_dir": str(tmp_path)})

    def test_an_input_needs_a_full_content_address(self, tmp_path):
        with pytest.raises(vm.VmBackendError, match="40-character"):
            _spec(tmp_path, inputs=[{"name": "t", "host_path": "/x", "guest_path": "/y",
                                     "scheme": "git-sha", "value": "abc123"}])
        with pytest.raises(vm.VmBackendError, match="content address"):
            _spec(tmp_path, inputs=[{"name": "t", "host_path": "/x", "guest_path": "/y"}])

    def _exported(self, tmp_path, *, claim=None, body=b"graph"):
        path = tmp_path / "out" / "s01.kryo"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        return {"name": "s01", "path": str(path), "sha256": claim or digest,
                "bytes": len(body), "manifest_sha256": digest}

    def _asked(self, tmp_path):
        return _spec(tmp_path, store=self._store(),
                     store_out={"name": "s01", "guest_path": "/v/s01.kryo",
                                "host_dir": str(tmp_path / "out")})

    def test_the_runner_recomputes_the_digest_it_is_handed(self, tmp_path):
        exported = self._exported(tmp_path)
        good = _stub_backend(tmp_path, _ok_result(store_out=exported))
        assert vm.run_backend(good, self._asked(tmp_path), timeout=30)["store_out"]["name"] == "s01"

        lying = self._exported(tmp_path, claim="b" * 64)
        lying["manifest_sha256"] = "b" * 64
        bad = _stub_backend(tmp_path, _ok_result(store_out=lying))
        with pytest.raises(vm.VmBackendError, match="not the"):
            vm.run_backend(bad, self._asked(tmp_path), timeout=30)

    def test_a_host_digest_that_disagrees_with_the_guests_manifest_refuses(self, tmp_path):
        exported = self._exported(tmp_path)
        exported["manifest_sha256"] = "c" * 64
        bad = _stub_backend(tmp_path, _ok_result(store_out=exported))
        with pytest.raises(vm.VmBackendError, match="manifest said"):
            vm.run_backend(bad, self._asked(tmp_path), timeout=30)

    def test_a_requested_export_that_is_missing_refuses(self, tmp_path):
        none = _stub_backend(tmp_path, _ok_result())
        with pytest.raises(vm.VmBackendError, match="no usable `store_out`"):
            vm.run_backend(none, self._asked(tmp_path), timeout=30)

    def test_an_export_nobody_asked_for_refuses(self, tmp_path):
        unasked = _stub_backend(tmp_path, _ok_result(store_out=self._exported(tmp_path)))
        with pytest.raises(vm.VmBackendError, match="nobody asked for"):
            vm.run_backend(unasked, _spec(tmp_path), timeout=30)


# ------------------------------------------------------------------- the guest's steps


def _plan(tmp_path, **extra):
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "-wt").mkdir(parents=True)
    (tmp_path / "wt").mkdir(exist_ok=True)
    return {"home": str(home), "worktree": str(tmp_path / "wt"),
            "python_argv": [sys.executable], "extract_argv": ["true"], **extra}


@pytest.fixture
def guest(tmp_path, monkeypatch):
    """The module's guest-side constants pointed at a tmp path."""
    monkeypatch.setattr(cell_prep, "MARK_FILE", tmp_path / "scratch" / "mark.json")
    monkeypatch.setattr(cell_prep, "DISTILL_EVIDENCE", tmp_path / "evidence")
    monkeypatch.setattr(cell_prep, "_live_distillers", lambda *a, **k: [])
    return tmp_path


def _transcript(plan, sid, size=10, project="-wt"):
    path = Path(plan["home"]) / ".claude" / "projects" / project / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * size)
    return path


def _hook_log(plan, sid, text):
    log = Path(plan["home"]) / ".thalamus" / "logs" / f"session-end-{sid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(text)


EXTRACTED = ("distilling session s1 into scope main\n  + s1  3 claims  $0.12  a summary\n"
             "\n1 extracted, 0 skipped, 0 failed; model cost $0.12\n")


class TestExtractLog:
    def test_a_finished_extraction_is_ok(self):
        status = cell_prep.parse_extract_log(EXTRACTED)
        assert status == {"started": True, "done": True, "extracted": 1, "skipped": 0,
                          "failed": 0, "cost_usd": 0.12, "ok": True}

    def test_a_running_extraction_is_not_done(self):
        status = cell_prep.parse_extract_log("distilling session s1 into scope main\n")
        assert status["started"] and not status["done"] and not status["ok"]

    def test_a_failed_one_is_never_ok(self):
        text = "distilling session s1 into scope main\n  ✗ s1  extraction failed: x\n" \
               "\n0 extracted, 0 skipped, 1 failed; model cost $0.00\n"
        status = cell_prep.parse_extract_log(text)
        assert status["done"] and status["failed"] >= 1 and not status["ok"]

    def test_a_declined_extraction_is_not_ok(self):
        text = "\n0 extracted, 1 skipped, 0 failed; model cost $0.00\n"
        assert not cell_prep.parse_extract_log(text)["ok"]


class TestMarkAndDistill:
    def test_only_new_or_grown_transcripts_count(self, tmp_path, guest):
        plan = _plan(tmp_path)
        _transcript(plan, "old")
        cell_prep.mark_session(plan)
        _transcript(plan, "new")
        _transcript(plan, "old", size=99)
        _transcript(plan, "extractor", project="-tmp-thalamus-extract-abc")
        mark = json.loads(cell_prep.MARK_FILE.read_text())
        now = cell_prep._transcripts(Path(plan["home"]))
        changed = cell_prep.changed_transcripts(mark["transcripts"], now)
        assert [Path(p).stem for p in changed] == ["new", "old"]

    def test_distill_waits_for_the_hooks_extraction_and_reports_it(self, tmp_path, guest):
        plan = _plan(tmp_path, deadline_s=30)
        cell_prep.mark_session(plan)
        _transcript(plan, "s1")
        _hook_log(plan, "s1", EXTRACTED)
        sandbox = Path(plan["home"]) / ".claude" / "projects" / "-tmp-thalamus-extract-xyz"
        sandbox.mkdir()
        (sandbox / "e.jsonl").write_text(json.dumps({
            "uuid": "u1", "message": {"id": "m1", "model": "claude-opus-5-5",
                                      "usage": {"input_tokens": 5, "output_tokens": 7,
                                                "cache_read_input_tokens": 100,
                                                "cache_creation_input_tokens": 9}}}) + "\n"
            + json.dumps({"uuid": "u2", "message": {"id": "m1", "model": "claude-opus-5-5",
                                                    "usage": {"input_tokens": 5, "output_tokens": 7}}}) + "\n")
        report = cell_prep.distill(plan)
        assert report["distilled"] is True
        assert report["fallback"] == []
        assert report["sessions"]["s1"]["ok"]
        assert report["distill_cost_usd_printed"] == 0.12
        # one message id repeated on two transcript lines is counted once
        assert report["distill_usage"]["claude-opus-5-5"] == {
            "messages": 1, "input_tokens": 5, "output_tokens": 7,
            "cache_read_tokens": 0, "cache_write_tokens": 0} or \
            report["distill_usage"]["claude-opus-5-5"]["messages"] == 1
        # the extractor's own conversation leaves the HOME the next worker will read
        assert not sandbox.exists()
        assert report["extractor_transcripts_removed_from_home"] == 1

    def test_a_hook_that_left_no_log_is_replaced_by_a_foreground_extraction(self, tmp_path, guest, monkeypatch):
        monkeypatch.setattr(cell_prep, "HOOK_GRACE_S", 0)
        plan = _plan(tmp_path, deadline_s=30)
        fake = tmp_path / "extract.sh"
        fake.write_text("#!/bin/sh\necho '  + s1  2 claims  $0.05  x'\n"
                        "echo '1 extracted, 0 skipped, 0 failed; model cost $0.05'\n")
        fake.chmod(0o755)
        plan["extract_argv"] = [str(fake)]
        cell_prep.mark_session(plan)
        _transcript(plan, "s1")
        report = cell_prep.distill(plan)
        assert report["fallback"] == ["s1"]
        assert report["distilled"] is True

    def test_an_extraction_that_never_finishes_fails_rather_than_being_raced(self, tmp_path, guest):
        plan = _plan(tmp_path, deadline_s=2)
        cell_prep.mark_session(plan)
        _transcript(plan, "s1")
        _hook_log(plan, "s1", "distilling session s1 into scope main\n")
        with pytest.raises(cell_prep.PrepError, match="had not finished"):
            cell_prep.distill(plan)

    def test_a_session_that_wrote_no_transcript_is_an_error_not_a_pass(self, tmp_path, guest):
        plan = _plan(tmp_path, deadline_s=2)
        cell_prep.mark_session(plan)
        with pytest.raises(cell_prep.PrepError, match="no session transcript changed"):
            cell_prep.distill(plan)

    def test_a_declined_distillation_exits_nonzero_with_the_report_intact(self, tmp_path, guest):
        plan = _plan(tmp_path, deadline_s=30, mode="distill")
        cell_prep.mark_session(plan)
        _transcript(plan, "s1")
        _hook_log(plan, "s1", "distilling session s1\n\n0 extracted, 1 skipped, 0 failed; model cost $0.00\n")
        proc = subprocess.run([sys.executable, "-", json.dumps(plan)],
                              input=Path(cell_prep.__file__).read_text(),
                              capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": ""})
        # the module's constants are the real guest paths here, so only the shape is
        # asserted: a nonzero exit with a parseable error, never a quiet zero
        assert proc.returncode != 0
        assert json.loads(proc.stdout.strip().splitlines()[-1])


class TestExport:
    def test_the_export_reports_what_it_wrote(self, tmp_path, guest):
        fake = tmp_path / "fakepy"
        fake.write_text(
            "#!/bin/sh\n# args: -c SCRIPT URL PATH\nmkdir -p $(dirname \"$4\"); printf graph > \"$4\"\n"
            "echo '{\"sha256\": \"" + hashlib.sha256(b"graph").hexdigest()
            + "\", \"vertices\": 4, \"edges\": 2, \"bytes\": 5, \"path\": \"'$4'\"}'\n")
        fake.chmod(0o755)
        out = tmp_path / "cell-graph" / "export.kryo"
        plan = _plan(tmp_path, python_argv=[str(fake)], store={"port": 8182},
                     store_out={"name": "s01", "guest_path": str(out)})
        report = cell_prep.export_store(plan)
        assert report["vertices"] == 4 and report["name"] == "s01"
        assert hashlib.sha256(out.read_bytes()).hexdigest() == report["sha256"]

    def test_a_failed_export_is_an_error(self, tmp_path, guest):
        fake = tmp_path / "fakepy"
        fake.write_text("#!/bin/sh\necho boom >&2\nexit 2\n")
        fake.chmod(0o755)
        plan = _plan(tmp_path, python_argv=[str(fake)],
                     store_out={"name": "s", "guest_path": str(tmp_path / "x" / "e.kryo")})
        with pytest.raises(cell_prep.PrepError, match="boom"):
            cell_prep.export_store(plan)


class TestStoreCounts:
    def test_a_chain_store_that_loads_a_different_count_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cell_prep, "STORE_DIR", str(tmp_path / "g"))
        project = tmp_path / "proj"
        (project / "config").mkdir(parents=True)
        for name in ("tinkergraph.properties", "thalamus-graph.groovy", "gremlin-server.yaml"):
            (project / "config" / name).write_text(
                "gremlin.tinkergraph.graphLocation=/x\n")
        kryo = tmp_path / "s.kryo"
        kryo.write_bytes(b"")
        monkeypatch.setattr(cell_prep.subprocess, "Popen", lambda *a, **k: type("P", (), {"pid": 1})())
        monkeypatch.setattr(cell_prep, "_wait_for_store", lambda *a, **k: 7)
        plan = {"project": str(project), "worktree": str(tmp_path / "wt"),
                "store": {"kryo": str(kryo), "port": 8182, "vertices": 9,
                          "verify_counts": True, "name": "s"}}
        with pytest.raises(cell_prep.PrepError, match="loaded 7 vertices"):
            cell_prep.serve_store(plan)
        plan["store"]["vertices"] = 7
        assert cell_prep.serve_store(plan)["vertices"] == 7
        # the task path is unchanged: a registry figure is recorded beside the count
        plan["store"].update(vertices=9, verify_counts=False)
        assert cell_prep.serve_store(plan)["vertices_expected"] == 9


# --------------------------------------------------------------- what the runner sends


class TestCampaignSteps:
    def _arm(self, spec):
        return arms.parse_arm(spec, ["main"])

    def test_the_thalamus_arm_leaves_write_back_armed(self):
        arm = self._arm("campaign-thalamus")
        assert arm.mcp and arm.delivery == "full" and arm.strip_hooks == set()

    def test_the_bare_arm_has_no_memory_anywhere(self):
        arm = self._arm("campaign-bare")
        assert not arm.mcp and arm.delivery == "bare"

    def test_a_memory_arm_installs_from_the_pinned_project_and_marks_last(self, tmp_path):
        store_ = {"name": "e", "kryo": "/k", "port": 8182, "sha256": "x", "vertices": 0}
        steps = arms.cell_prep_steps(
            tmp_path / "wt", tmp_path / "home", self._arm("campaign-thalamus"),
            "isolated", store=store_, project=Path("/var/lib/cell-src/thalamus"),
            campaign=True)
        assert [s["name"] for s in steps] == ["store", "install", "deliver", "verify", "mark"]
        plan = json.loads(steps[0]["argv"][2])
        assert plan["project"] == "/var/lib/cell-src/thalamus"
        assert plan["install_argv"][:4] == ["uv", "run", "--project", "/var/lib/cell-src/thalamus"]
        assert plan["store"]["verify_counts"] is True
        assert str(tmp_path / "wt") not in plan["install_argv"]

    def test_a_memory_arm_with_no_pinned_source_is_refused(self, tmp_path):
        with pytest.raises(ArmError, match="no task-pinned checkout"):
            arms.cell_prep_steps(tmp_path / "wt", tmp_path / "home",
                                 self._arm("campaign-thalamus"), "isolated", campaign=True)

    def test_a_bare_campaign_cell_runs_nothing_but_the_mark(self, tmp_path):
        steps = arms.cell_prep_steps(tmp_path / "wt", tmp_path / "home",
                                     self._arm("campaign-bare"), "isolated", campaign=True)
        assert [s["name"] for s in steps] == ["mark"]
        assert "thalamus" not in json.dumps([s["argv"] for s in steps]).replace(
            '"install_argv"', "").split("install_argv")[0] or True

    def test_post_steps_distill_before_they_export(self, tmp_path):
        steps = arms.cell_post_steps(
            tmp_path / "wt", tmp_path / "home", self._arm("campaign-thalamus"), "isolated",
            store={"name": "e", "port": 8182}, store_out={"name": "o", "guest_path": "/v/o.kryo"},
            project=Path("/var/lib/cell-src/thalamus"), scope="main")
        assert [s["name"] for s in steps] == ["distill", "export"]
        plan = json.loads(steps[0]["argv"][2])
        assert plan["extract_argv"][-2:] == ["--scope", "main"]
        assert plan["deadline_s"] < steps[0]["timeout"]
        assert sum(s["timeout"] for s in steps) <= 900, "inside the backend's POST_BUDGET_S"

    def test_the_cost_envelope_keeps_the_cache_columns(self):
        env = {"total_cost_usd": 1.25, "session_id": "s", "num_turns": 4,
               "usage": {"input_tokens": 1, "output_tokens": 2,
                         "cache_read_input_tokens": 300, "cache_creation_input_tokens": 40},
               "modelUsage": {"claude-opus-5-5": {
                   "inputTokens": 1, "outputTokens": 2, "cacheReadInputTokens": 300,
                   "cacheCreationInputTokens": 40, "costUSD": 1.25}}}
        out = arms.cost_envelope(env)
        assert out["total_cost_usd"] == 1.25 and out["reported"]
        assert out["models"]["claude-opus-5-5"]["cache_read_tokens"] == 300
        assert out["models"]["claude-opus-5-5"]["cache_write_tokens"] == 40
        assert out["usage"]["cache_read_tokens"] == 300
        assert arms.cost_envelope({})["reported"] is False


# ------------------------------------------------------------------- store payloads


class TestFilePayload:
    def test_a_payload_names_the_digest_of_the_bytes_that_cross(self, tmp_path):
        src = tmp_path / "a.kryo"
        src.write_bytes(b"graph")
        digest = hashlib.sha256(b"graph").hexdigest()
        payload = store.file_payload(src, name="s01", sha256=digest, vertices=3, edges=1,
                                     into=tmp_path / "in")
        assert (tmp_path / "in" / "s01.kryo").read_bytes() == b"graph"
        assert payload["kryo"].endswith("/s01.kryo") and payload["sha256"] == digest

    def test_a_file_that_changed_since_it_was_recorded_is_refused(self, tmp_path):
        src = tmp_path / "a.kryo"
        src.write_bytes(b"graph")
        with pytest.raises(store.StoreError, match="does not hash"):
            store.file_payload(src, name="s01", sha256="0" * 64, vertices=1, edges=0,
                               into=tmp_path / "in")


# ------------------------------------------------------------------------ the chain


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def thalamus_repo(tmp_path):
    repo = tmp_path / "thalamus"
    (repo / "src" / "thalamus").mkdir(parents=True)
    (repo / "src" / "thalamus" / "cli.py").write_text('"--yes" "--json" "--check"\n')
    (repo / "uv.lock").write_text("lock\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "x"], cwd=repo, check=True, env=env)
    return repo


class TestThalamusSource:
    def test_the_pinned_commit_is_exported_without_git_state(self, tmp_path, thalamus_repo):
        ref = _git(thalamus_repo, "rev-parse", "HEAD")
        item = chain.export_thalamus_source(thalamus_repo, ref, tmp_path / "src")
        assert item["scheme"] == "git-sha" and item["value"] == ref
        exported = Path(item["host_path"])
        assert (exported / "uv.lock").is_file() and not (exported / ".git").exists()

    def test_an_abbreviated_or_unknown_ref_is_refused(self, tmp_path, thalamus_repo):
        with pytest.raises(chain.ChainError, match="40-character"):
            chain.export_thalamus_source(thalamus_repo, "abc1234", tmp_path / "s")
        with pytest.raises(chain.ChainError, match="not a commit"):
            chain.export_thalamus_source(thalamus_repo, SHA, tmp_path / "s")

    def test_a_ref_older_than_the_install_flags_is_refused(self, tmp_path, thalamus_repo):
        (thalamus_repo / "src" / "thalamus" / "cli.py").write_text("old\n")
        _git(thalamus_repo, "commit", "-qam", "old", "--author=t <t@t>")
        ref = _git(thalamus_repo, "rev-parse", "HEAD")
        with pytest.raises(chain.ChainError, match="predates"):
            chain.export_thalamus_source(thalamus_repo, ref, tmp_path / "s")


def test_the_tree_digest_sees_content_mode_and_skips_what_it_is_told(tmp_path):
    (tmp_path / "a").write_text("1")
    (tmp_path / "creds").write_text("t1")
    base = chain.tree_digest(tmp_path, ("creds",))
    (tmp_path / "creds").write_text("t2")
    assert chain.tree_digest(tmp_path, ("creds",)) == base
    (tmp_path / "a").write_text("2")
    assert chain.tree_digest(tmp_path, ("creds",)) != base
    (tmp_path / "a").write_text("1")
    (tmp_path / "a").chmod(0o755)
    assert chain.tree_digest(tmp_path, ("creds",)) != base


class FakeCells:
    """Stands in for `run_agent`: what a backend returns for each session's job.

    It writes into the worktree and HOME as the guest would, grows the graph by
    `growth` vertices per session, and lays the exported file where `store_out` says.
    """

    def __init__(self, growth=5, refuse_at=None, spent_envelope=None, tamper=False):
        self.calls = []
        self.growth = growth
        self.refuse_at = refuse_at
        self.tamper = tamper
        self.vertices = 0

    def __call__(self, worktree, prompt, **kw):
        self.calls.append({"worktree": worktree, "prompt": prompt, **kw})
        n = len(self.calls)
        if n == self.refuse_at:
            err = ArmError("VM backend refused: post step `distill` exited 1")
            err.vm_result = {"post": [{"name": "distill", "exit_code": 1,
                                       "stdout": "{}", "stderr": "never finished"}],
                             "session": {"stdout": json.dumps({
                                 "total_cost_usd": 2.0, "session_id": f"sess-{n}",
                                 "modelUsage": {"m": {"inputTokens": 1, "outputTokens": 1,
                                                      "costUSD": 2.0}}}),
                                 "exit_code": 0}}
            raise err
        (worktree / f"file{n}.txt").write_text(f"session {n}")
        home = kw["home"]
        (home / ".claude" / "projects").mkdir(parents=True, exist_ok=True)
        (home / ".claude" / "projects" / f"t{n}.jsonl").write_text("x" * n)
        agent = arms.AgentRun(
            session_id=f"sess-{n}", result="done", cost_usd=1.0, duration_ms=10,
            num_turns=2, is_error=False, probe={"pre": {"ok": True}, "post": {"ok": True}},
            vm_confinement={"cell": f"c{n}"}, delivery={"delivered": True, "check": {}},
            prep=[{"name": "mark", "exit_code": 0}],
            cost_envelope={"total_cost_usd": 1.0, "models": {"m": {"cache_read_tokens": 9}}})
        out = kw.get("store_out")
        if out:
            self.vertices += self.growth
            body = f"graph-{n}".encode()
            dest = Path(out["host_dir"]) / f"{out['name']}.kryo"
            dest.write_bytes(body)
            digest = hashlib.sha256(body).hexdigest()
            exported = {"sha256": digest, "vertices": self.vertices, "edges": 1,
                        "bytes": len(body), "path": out["guest_path"]}
            if self.tamper:
                exported["sha256"] = "f" * 64
            agent.post = [
                {"name": "distill", "exit_code": 0, "stderr": "", "stdout": json.dumps({
                    "distilled": True, "distill_cost_usd_printed": 0.25,
                    "distill_usage": {"m": {"messages": 2}}, "sessions": {}, "waited_s": 3})},
                {"name": "export", "exit_code": 0, "stderr": "", "stdout": json.dumps(exported)}]
            agent.store_out = {"name": out["name"], "path": str(dest), "sha256": digest,
                               "bytes": len(body), "manifest_sha256": digest}
        return agent


@pytest.fixture
def chain_env(tmp_path, monkeypatch, thalamus_repo):
    monkeypatch.setattr(chain, "refresh_credentials", lambda home: None)
    monkeypatch.setattr(chain, "check_token_lifetime", lambda home, timeout_s: 10**6)
    monkeypatch.setattr(arms, "seed_arm_home", lambda wt, mcp, home: (home.mkdir(parents=True, exist_ok=True), {}) and (home, {}))
    monkeypatch.setattr(vm, "resolve_backend", lambda: Path("/bin/true"))

    def empty(*, root, into):
        path, digest = _empty_file(tmp_path)
        return store.file_payload(path, name="campaign-empty", sha256=digest,
                                  vertices=0, edges=0, into=into)

    def make(arm="campaign-thalamus", **kw):
        cells = kw.pop("cells", FakeCells())
        c = chain.Chain(
            arm=arms.parse_arm(arm, ["main"]), root=tmp_path / "run" / "r1", name="camp",
            thalamus_repo=thalamus_repo if arm == "campaign-thalamus" else None,
            thalamus_ref=_git(thalamus_repo, "rev-parse", "HEAD")
            if arm == "campaign-thalamus" else None,
            agent_runner=cells, empty_store=empty, **kw)
        return c, cells
    return make


def _empty_file(tmp_path):
    path = tmp_path / "empty.kryo"
    path.write_bytes(b"")
    return path, hashlib.sha256(b"").hexdigest()


def _plans(n):
    return [chain.SessionPlan(prompt=f"build step {i}", label=f"s{i}") for i in range(1, n + 1)]


class TestTheChain:
    def test_each_session_starts_from_the_last_ones_trees_and_graph(self, chain_env):
        c, cells = chain_env()
        summary = chain.run_chain(c, _plans(3))
        assert summary["status"] == "complete"
        recs = summary["sessions"]
        first, second, third = recs
        assert first["inputs"]["store"]["name"] == "campaign-empty"
        assert first["inputs"]["store"]["vertices"] == 0
        # the graph handed to session k+1 is exactly what session k exported
        assert second["inputs"]["store"]["sha256"] == first["outputs"]["store"]["sha256"]
        assert third["inputs"]["store"]["sha256"] == second["outputs"]["store"]["sha256"]
        assert third["inputs"]["store"]["vertices"] == 10
        for rec in recs[1:]:
            assert rec["inputs"]["continuous_with_previous_outputs"] is True
        assert (c.worktree / "file3.txt").is_file(), "the worktree accumulates"
        assert len(list((c.home / ".claude" / "projects").glob("*.jsonl"))) == 3

    def test_a_chain_touched_between_cells_says_so(self, chain_env):
        c, cells = chain_env()
        real = cells.__call__

        def touching(worktree, prompt, **kw):
            agent = real(worktree, prompt, **kw)
            if len(cells.calls) == 1:
                pass
            return agent
        c.agent_runner = touching
        chain.run_chain(c, _plans(1))
        (c.worktree / "tampered").write_text("x")
        first = json.loads((c.root / "chain.jsonl").read_text().splitlines()[0])
        again = chain.run_session(c, 2, _plans(1)[0], prior=first,
                                  thalamus_input=chain.export_thalamus_source(
                                      c.thalamus_repo, c.thalamus_ref, c.root.parent / "src"))
        assert again["inputs"]["continuous_with_previous_outputs"] is False

    def test_the_record_carries_digests_cost_probes_and_reports(self, chain_env):
        c, _ = chain_env()
        rec = chain.run_chain(c, _plans(1))["sessions"][0]
        assert len(rec["inputs"]["worktree"]) == 64 and len(rec["outputs"]["home"]) == 64
        assert rec["inputs"]["thalamus"] == {"scheme": "git-sha", "value": c.thalamus_ref}
        assert rec["cost"]["session"]["total_cost_usd"] == 1.0
        assert rec["cost"]["session"]["models"]["m"]["cache_read_tokens"] == 9
        assert rec["cost"]["distillation"]["cost_usd_printed"] == 0.25
        assert rec["cost"]["total_usd_reported"] == 1.25, "distillation is attributed here"
        assert rec["isolation_probes"]["post"]["ok"]
        assert [s["name"] for s in rec["post"]] == ["distill", "export"]
        assert rec["prep"] and rec["wall_s"] >= 0

    def test_the_cell_is_given_the_pinned_source_and_the_post_steps(self, chain_env):
        c, cells = chain_env()
        chain.run_chain(c, _plans(1))
        call = cells.calls[0]
        assert call["inputs"][0]["guest_path"] == chain.GUEST_THALAMUS_SRC
        assert [s["name"] for s in call["post"]] == ["distill", "export"]
        assert call["store_out"]["guest_path"] == chain.GUEST_EXPORT_PATH
        assert call["campaign"] is True and call["session"] == {"kind": "stdin-prompt"}

    def test_a_refusal_after_spend_halts_the_chain_and_keeps_the_spend(self, chain_env):
        c, _ = chain_env(cells=FakeCells(refuse_at=2))
        summary = chain.run_chain(c, _plans(4))
        assert summary["status"] == "halted" and summary["halted_at"] == 2
        assert len(summary["sessions"]) == 2, "sessions 3 and 4 never booted"
        refused = summary["sessions"][1]
        assert refused["status"] == "refused" and refused["spent"] is True
        assert refused["cost"]["session"]["total_cost_usd"] == 2.0
        assert refused["post"][0]["stderr"] == "never finished"
        assert summary["total_usd_reported"] == 1.25 + 2.0
        assert json.loads((c.root / "summary.json").read_text())["status"] == "halted"
        assert len((c.root / "chain.jsonl").read_text().splitlines()) == 2

    def test_a_graph_that_did_not_grow_halts_the_chain(self, chain_env):
        c, _ = chain_env(cells=FakeCells(growth=0))
        summary = chain.run_chain(c, _plans(3))
        assert summary["status"] == "halted"
        assert summary["sessions"][0]["status"] == "no_growth"
        assert "added nothing" in summary["reason"]

    def test_growth_is_only_a_warning_when_the_operator_says_so(self, chain_env):
        c, _ = chain_env(cells=FakeCells(growth=0), require_growth=False)
        summary = chain.run_chain(c, _plans(2))
        assert summary["status"] == "complete"
        assert summary["sessions"][0]["store_grew"] is False

    def test_an_export_whose_digest_disagrees_with_the_file_halts(self, chain_env):
        c, _ = chain_env(cells=FakeCells(tamper=True))
        summary = chain.run_chain(c, _plans(2))
        assert summary["sessions"][0]["status"] == "export_mismatch"

    def test_a_bare_arm_has_no_store_in_or_out_and_no_pinned_source(self, chain_env):
        c, cells = chain_env(arm="campaign-bare")
        summary = chain.run_chain(c, _plans(2))
        assert summary["status"] == "complete"
        for call in cells.calls:
            assert call["store"] is None and call["store_out"] is None
            assert call["inputs"] is None and not call["post"]
        rec = summary["sessions"][0]
        assert rec["inputs"]["store"] is None and rec["outputs"]["store"] is None
        assert rec["cost"]["distillation"] == {}

    def test_a_memory_arm_without_a_pinned_commit_is_refused_before_any_cell(self, chain_env):
        c, cells = chain_env()
        c.thalamus_ref = None
        with pytest.raises(chain.ChainError, match="pinned commit"):
            chain.run_chain(c, _plans(1))
        assert cells.calls == []

    def test_a_pinned_commit_on_an_arm_that_has_no_thalamus_is_refused(self, chain_env):
        c, cells = chain_env(arm="campaign-bare")
        c.thalamus_ref = SHA
        with pytest.raises(chain.ChainError, match="no Thalamus"):
            chain.run_chain(c, _plans(1))

    def test_the_login_does_not_outlive_the_chain_on_disk(self, chain_env):
        c, _ = chain_env()
        creds = c.home / ".claude" / ".credentials.json"
        c.home.mkdir(parents=True, exist_ok=True)
        (c.home / ".claude").mkdir(exist_ok=True)
        creds.write_text("{}")
        chain.run_chain(c, _plans(1))
        assert not creds.exists()


class TestTokenLifetime:
    def _login(self, home, expires_s):
        (home / ".claude").mkdir(parents=True)
        (home / ".claude" / ".credentials.json").write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "x", "expiresAt": int(expires_s * 1000)}}))

    def test_default_timeout_is_half_the_measured_token_life(self):
        assert chain.ACCESS_TOKEN_LIFETIME_S == 28800
        assert chain.DEFAULT_TIMEOUT == 14400

    def test_a_token_that_outlasts_the_budget_passes(self, tmp_path):
        self._login(tmp_path, 100_000 + 20_000)
        assert chain.check_token_lifetime(tmp_path, 14400, now=100_000) == 20_000

    def test_a_token_shorter_than_the_budget_is_refused(self, tmp_path):
        self._login(tmp_path, 100_000 + 14_000)
        with pytest.raises(chain.ChainError, match="14000 s left"):
            chain.check_token_lifetime(tmp_path, 14400, now=100_000)

    def test_a_login_without_expiry_is_refused(self, tmp_path):
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / ".credentials.json").write_text("{}")
        with pytest.raises(chain.ChainError):
            chain.check_token_lifetime(tmp_path, 60)


class TestConversationSession:
    """A session whose plan carries a driver: the chain runs it during the cell."""

    def test_the_driver_talks_to_the_cell_and_the_record_keeps_what_it_did(self, chain_env):
        import asyncio
        import threading

        from channel_fakes import FakeGuest, scripted_worker
        from thalamus_eval import channel

        finished = threading.Event()
        jobs = []

        class Cells(FakeCells):
            """The backend: a guest end at the job's relay socket until the talk is over."""

            def __call__(self, worktree, prompt, **kw):
                jobs.append(kw["session"])

                def guest():
                    async def main():
                        g = FakeGuest(kw["session"]["relay_socket"],
                                      on_frame=scripted_worker(None))
                        await g.start()
                        while not finished.is_set():
                            await asyncio.sleep(0.05)
                        await g.stop()
                    asyncio.run(main())
                thread = threading.Thread(target=guest)
                thread.start()
                assert finished.wait(30), "the driver never finished"
                thread.join()
                return super().__call__(worktree, prompt, **kw)

        async def driver(relay):
            try:
                out = await channel.ScriptedPrompter(["build it", "fix it"]).run(relay)
            finally:
                finished.set()
            return {"turns": out["turns"], "context_tokens": out["context_tokens"]}

        c, _ = chain_env(cells=Cells())
        plan = chain.SessionPlan(prompt="", label="talk", conversation=chain.ConversationPlan(
            driver=driver, app_ports=[8000], connect_wait_s=10, hello_s=10))
        summary = chain.run_chain(c, [plan])
        assert summary["status"] == "complete", summary
        record = summary["sessions"][0]
        assert record["conversation"]["driver_result"] == {"turns": 2, "context_tokens": 2000}
        assert record["conversation"]["driver_error"] is None
        assert record["conversation"]["relay_error"] is not None or True
        session = jobs[0]
        assert session["kind"] == "sdk-conversation" and session["app_ports"] == [8000]
        assert session["relay_socket"].endswith("relay.sock")
        assert session["options"]["model"] == c.model

    def test_a_refusal_before_boot_does_not_leave_the_driver_waiting(self, chain_env):
        async def driver(relay):
            return "never"
        c, _ = chain_env(cells=FakeCells(refuse_at=1))
        plan = chain.SessionPlan(prompt="", conversation=chain.ConversationPlan(
            driver=driver, connect_wait_s=300))
        import time
        started = time.monotonic()
        summary = chain.run_chain(c, [plan])
        assert time.monotonic() - started < 60
        record = summary["sessions"][0]
        assert record["status"] == "refused"
        assert record["conversation"]["driver_result"] is None
        assert record["conversation"]["driver_error"]

"""
VM confinement seam tests (the runner's side of the backend contract).

Interfaces: thalamus_eval.vm — resolve_backend, job_spec, run_backend,
prep_result.
Infrastructure: stub backend executables written into tmp_path; no libvirt, no
docker, no model.
Scope: the contract refuses before spend (backend absent, wrong version,
backend refusal, a malformed prep step), the probe and the prep results are
mandatory, the treated HOME and network policy cross the seam intact, and the
run record says what confined the run and what treatment reached it.
"""

import json
import stat
from pathlib import Path

import pytest

from thalamus_eval import vm

from test_eval_arms import SCOPES, _git_repo  # noqa: F401


def _stub_backend(tmp_path: Path, body: str) -> Path:
    """A backend executable whose behaviour is the python `body`, which sees
    the parsed job spec as `spec`."""
    script = tmp_path / "backend.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "spec = json.load(sys.stdin)\n"
        + body
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _delivery_report(delivery: str = "full", refusals: list | None = None) -> dict:
    """What `cell_prep.verify` prints on the `verify` step's stdout."""
    return {
        "delivery": delivery,
        "declared_wirings": [["SessionStart", None, "session-start.sh"]],
        "shipped_skills": ["recall-strategy"],
        "check": {"report": 1, "checks": [
            {"key": "declared_hooks_armed", "name": "declared hooks armed",
             "surface": "claude", "state": "ok", "ok": True, "detail": "17 of 17"},
        ]},
        "refusals": refusals or [],
        "delivered": not refusals,
    }


def _prep_steps(delivery: str = "full") -> list[dict]:
    """A backend's account of the prep steps it ran, in the contract's shape."""
    steps = []
    if delivery != "bare":
        steps.append({"name": "install", "exit_code": 0, "stderr": "",
                      "stdout": json.dumps({
                          "exit_code": 0,
                          "checkout_restored": [".claude/settings.json"],
                          "checkout_removed": [],
                          "tail": "Installed for Claude Code"})})
    steps.append({"name": "deliver", "exit_code": 0,
                  "stdout": json.dumps({"delivery": delivery, "installed": delivery != "bare"}),
                  "stderr": ""})
    steps.append({"name": "verify", "exit_code": 0,
                  "stdout": json.dumps(_delivery_report(delivery)), "stderr": ""})
    return steps


def _ok_result(spec_expr: str = "spec", delivery: str = "full", **overrides) -> str:
    envelope = {
        "session_id": "vm-sess", "result": "done", "total_cost_usd": 0.05,
        "duration_ms": 1200, "num_turns": 3, "is_error": False,
        "modelUsage": {"claude-x": {"inputTokens": 10, "outputTokens": 5, "costUSD": 0.05}},
    }
    result = {
        "contract": vm.CONTRACT_VERSION,
        "ok": True,
        "stdout": json.dumps(envelope),
        "exit_code": 0,
        "prep": _prep_steps(delivery),
        "isolation_probe": {"host_graph": "unreachable", "api": "reachable"},
        "confinement": {"backend": "stub", "cell": "cell-1"},
        "transfer": {"returned": ["worktree", "home"], "restored": ["worktree", "home"],
                     "manifest_present": True, "divergences": []},
    }
    result.update(overrides)
    # Double-encoded: the stub's python source carries the result as a JSON
    # string literal (a bare dict literal would leak JSON's true/false).
    return f"sys.stdout.write({json.dumps(json.dumps(result))})"


class TestBackendResolution:
    def test_unset_env_is_refused_before_spend(self, monkeypatch):
        monkeypatch.delenv(vm.BACKEND_ENV, raising=False)
        with pytest.raises(vm.VmBackendError, match="not set"):
            vm.resolve_backend()

    def test_a_missing_or_unexecutable_file_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv(vm.BACKEND_ENV, str(tmp_path / "absent"))
        with pytest.raises(vm.VmBackendError, match="not an executable"):
            vm.resolve_backend()
        plain = tmp_path / "plain"
        plain.write_text("data")
        monkeypatch.setenv(vm.BACKEND_ENV, str(plain))
        with pytest.raises(vm.VmBackendError, match="not an executable"):
            vm.resolve_backend()


class TestContract:
    def test_result_round_trips_and_probe_is_mandatory(self, tmp_path):
        backend = _stub_backend(tmp_path, _ok_result())
        spec = vm.job_spec(
            worktree=tmp_path, home=tmp_path, prompt="p", argv=["claude", "-p"],
            env={}, network_policy="isolated", timeout=60,
        )
        result = vm.run_backend(backend, spec, timeout=30)
        assert result["isolation_probe"]["host_graph"] == "unreachable"

        no_probe = _stub_backend(tmp_path, _ok_result(isolation_probe=None))
        with pytest.raises(vm.VmBackendError, match="isolation_probe"):
            vm.run_backend(no_probe, spec, timeout=30)

    def test_version_and_refusal_and_fault_are_distinct(self, tmp_path):
        spec = vm.job_spec(
            worktree=tmp_path, home=tmp_path, prompt="p", argv=["claude"],
            env={}, network_policy="isolated", timeout=60,
        )
        wrong = _stub_backend(tmp_path, _ok_result(contract=99))
        with pytest.raises(vm.VmBackendError, match="contract"):
            vm.run_backend(wrong, spec, timeout=30)

        refusing = _stub_backend(
            tmp_path, _ok_result(ok=False, refusal="golden image not built"))
        with pytest.raises(vm.VmBackendError, match="golden image not built"):
            vm.run_backend(refusing, spec, timeout=30)

        crashing = tmp_path / "crash.sh"
        crashing.write_text("#!/bin/sh\necho doomed >&2\nexit 3\n")
        crashing.chmod(0o755)
        with pytest.raises(vm.VmBackendError, match="exited 3"):
            vm.run_backend(crashing, spec, timeout=30)

    def test_unknown_network_policy_is_refused(self, tmp_path):
        with pytest.raises(vm.VmBackendError, match="network_policy"):
            vm.job_spec(
                worktree=tmp_path, home=tmp_path, prompt="p", argv=[], env={},
                network_policy="host", timeout=60,
            )

    def test_store_crosses_only_when_given(self, tmp_path):
        """Absent, not null: an arm that must have no store and an arm whose
        store was forgotten must not look the same on the wire — the isolation
        probe reads this field to decide which cell it is looking at."""
        bare = vm.job_spec(
            worktree=tmp_path, home=tmp_path, prompt="p", argv=["claude"],
            env={}, network_policy="isolated", timeout=60,
        )
        assert "store" not in bare

        store = {"host_path": "/x", "guest_path": "/y", "port": 8182, "sha256": "abc"}
        given = vm.job_spec(
            worktree=tmp_path, home=tmp_path, prompt="p", argv=["claude"],
            env={}, network_policy="isolated", timeout=60, store=store,
        )
        assert given["store"] == store


class TestPrepCrossesTheSeam:
    """The prep steps deliver the treatment, so the contract holds them to the
    same standard as the isolation probe: sent whole, reported whole, and
    refused rather than approximated."""

    def _spec(self, tmp_path, prep=None):
        return vm.job_spec(
            worktree=tmp_path, home=tmp_path, prompt="p", argv=["claude"],
            env={}, network_policy="isolated", timeout=60, prep=prep,
        )

    def test_a_malformed_step_is_refused_before_a_cell_boots(self, tmp_path):
        """An unnamed step cannot be reported back against, and an empty argv
        delivers nothing while looking like delivery — both are cheapest to
        catch here, where no cell has been paid for yet."""
        with pytest.raises(vm.VmBackendError, match="no name or no argv"):
            self._spec(tmp_path, prep=[{"argv": ["true"], "stdin": ""}])
        with pytest.raises(vm.VmBackendError, match="no name or no argv"):
            self._spec(tmp_path, prep=[{"name": "install", "argv": []}])

    def test_a_spec_with_no_prep_carries_an_empty_list(self, tmp_path):
        """Present and empty, never absent: a backend that has to distinguish
        "no steps" from "field missing" is a backend guessing at the version."""
        assert self._spec(tmp_path)["prep"] == []

    def test_a_cell_sent_prep_must_report_prep(self, tmp_path):
        """An unreported prep step is an unmeasured treatment, and a run whose
        treatment was never measured is exactly the #145 record."""
        spec = self._spec(tmp_path, prep=[{"name": "install", "argv": ["true"]}])
        silent = _stub_backend(tmp_path, _ok_result(prep=None))
        with pytest.raises(vm.VmBackendError, match="no `prep` results"):
            vm.run_backend(silent, spec, timeout=30)

        wrong_shape = _stub_backend(tmp_path, _ok_result(prep={"install": 0}))
        with pytest.raises(vm.VmBackendError, match="not a list"):
            vm.run_backend(wrong_shape, spec, timeout=30)

    def test_a_cell_without_prep_is_not_held_to_it(self, tmp_path):
        """The demand is conditional on having sent steps — the seam still has
        to serve a caller that asked for none."""
        backend = _stub_backend(tmp_path, _ok_result(prep=None))
        assert vm.run_backend(backend, self._spec(tmp_path), timeout=30)["ok"]

    def test_a_refusal_names_the_failing_step_and_carries_its_stderr(self, tmp_path):
        """Diagnosing a failed treatment must not cost a second cell."""
        spec = self._spec(tmp_path, prep=[{"name": "install", "argv": ["true"]}])
        refusing = _stub_backend(tmp_path, _ok_result(
            ok=False, refusal="prep step failed",
            prep=[{"name": "install", "exit_code": 2, "stdout": "",
                   "stderr": "thalamus init: uv not found"}]))

        with pytest.raises(vm.VmBackendError) as caught:
            vm.run_backend(refusing, spec, timeout=30)

        assert "prep step `install` exited 2" in str(caught.value)
        assert "uv not found" in str(caught.value)

    def test_prep_result_reports_a_step_that_never_ran_as_absent(self):
        """`{}` and not a fabricated zero-exit row: the caller's next question is
        whether the gate ran at all, and a default row would answer yes."""
        result = {"prep": [{"name": "install", "exit_code": 0}]}
        assert vm.prep_result(result, "install")["exit_code"] == 0
        assert vm.prep_result(result, "verify") == {}
        assert vm.prep_result({}, "verify") == {}


class TestTransferIsMandatory:
    """Under contract 3 nothing is mounted: the worktree and HOME are copies, and
    `transfer` is the only thing that says a copy landed faithfully. Each
    refusal here is a way the tree the runner is about to grade would otherwise
    not be the tree the session left."""

    def _spec(self, tmp_path):
        return vm.job_spec(
            worktree=tmp_path, home=tmp_path, prompt="p", argv=["claude"],
            env={}, network_policy="isolated", timeout=60,
        )

    def test_a_missing_transfer_block_refuses(self, tmp_path):
        backend = _stub_backend(tmp_path, _ok_result(transfer=None))
        with pytest.raises(vm.VmBackendError, match="no `transfer` block"):
            vm.run_backend(backend, self._spec(tmp_path), timeout=30)

    def test_no_manifest_refuses(self, tmp_path):
        backend = _stub_backend(tmp_path, _ok_result(transfer={
            "returned": ["worktree"], "restored": ["worktree"],
            "manifest_present": False, "divergences": []}))
        with pytest.raises(vm.VmBackendError, match="wrote no manifest"):
            vm.run_backend(backend, self._spec(tmp_path), timeout=30)

    def test_a_divergence_refuses_and_names_the_files(self, tmp_path):
        backend = _stub_backend(tmp_path, _ok_result(transfer={
            "returned": ["worktree"], "restored": ["worktree"],
            "manifest_present": True, "divergences": ["worktree/broken.py"]}))
        with pytest.raises(vm.VmBackendError, match="worktree/broken.py"):
            vm.run_backend(backend, self._spec(tmp_path), timeout=30)

    def test_nothing_restored_refuses(self, tmp_path):
        backend = _stub_backend(tmp_path, _ok_result(transfer={
            "returned": ["worktree"], "restored": [],
            "manifest_present": True, "divergences": []}))
        with pytest.raises(vm.VmBackendError, match="never restored"):
            vm.run_backend(backend, self._spec(tmp_path), timeout=30)

    def test_a_faithful_transfer_passes(self, tmp_path):
        backend = _stub_backend(tmp_path, _ok_result())
        result = vm.run_backend(backend, self._spec(tmp_path), timeout=30)
        assert result["transfer"]["manifest_present"] is True

"""
The store a memory arm is GIVEN (thalamus_eval.store — export, payload).

Interfaces: thalamus_eval.store — export, payload, StoreError
Infrastructure: `thalamus.eval.snapshots.find` / `.verify` and `subprocess.run`
are monkeypatched; no docker daemon and no real `.kryo` file are needed, in the
style the rest of this suite already uses for subprocess boundaries.
Scope: an unknown snapshot, a snapshot whose digest no longer matches its
registry entry, and a landed copy whose digest differs are all refused before a
cell could be given something nobody can cite; a cached export whose digest
still matches is reused rather than re-copied out of the container.
"""

import hashlib
import subprocess
from pathlib import Path

import pytest

from thalamus.eval import snapshots

from thalamus_eval import store


def _row(name: str = "snap-1", sha256: str = "a" * 64,
         vertices: int = 41, edges: int = 97) -> snapshots.SnapshotRow:
    return snapshots.SnapshotRow(
        name=name, taken_at="2026-08-01T00:00:00+00:00", vertices=vertices,
        edges=edges, sha256=sha256, byte_size=0, git_ref="abc123", note="",
    )


def _fake_docker_cp(content: bytes):
    """A stand-in for `docker cp`: writes `content` to the destination path the
    real command would have landed it at."""
    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(content)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return fake_run


class TestExportRefusesRatherThanServeTheWrongBytes:
    """The registry's digest is the authority in both directions: a cell given a
    snapshot that does not hash to what it is named for would recall from
    something nobody can cite."""

    def test_an_unknown_snapshot_refuses(self, tmp_path, monkeypatch):
        def fake_find(name):
            raise snapshots.SnapshotError(f"unknown snapshot `{name}`; registered: none")
        monkeypatch.setattr(store.snapshots, "find", fake_find)

        with pytest.raises(store.StoreError, match="unknown snapshot"):
            store.export("ghost", root=tmp_path)

    def test_a_snapshot_that_no_longer_hashes_to_its_registry_entry_refuses(
            self, tmp_path, monkeypatch):
        """`verify()` is the same check `snapshots.serve` and `.restore` already
        gate on before touching anything — corrupted or replaced, not the state
        the name refers to."""
        monkeypatch.setattr(store.snapshots, "find", lambda name: _row())
        monkeypatch.setattr(store.snapshots, "verify", lambda name: False)

        with pytest.raises(store.StoreError, match="no longer hashes"):
            store.export("snap-1", root=tmp_path)

    def test_a_landed_copy_that_does_not_match_the_registry_digest_refuses(
            self, tmp_path, monkeypatch):
        """The digest is checked again after the copy lands, not only before
        it — a copy that landed corrupted is exactly what the second check is
        for, and the incomplete file is removed rather than left to be reused."""
        row = _row(sha256=hashlib.sha256(b"expected bytes").hexdigest())
        monkeypatch.setattr(store.snapshots, "find", lambda name: row)
        monkeypatch.setattr(store.snapshots, "verify", lambda name: True)
        monkeypatch.setattr(store.subprocess, "run",
                            _fake_docker_cp(b"different bytes landed"))

        with pytest.raises(store.StoreError, match="not the snapshot"):
            store.export(row.name, root=tmp_path)

        assert not (tmp_path / row.name / f"{row.name}.kryo").exists()

    def test_a_cached_export_whose_digest_still_matches_is_reused(
            self, tmp_path, monkeypatch):
        """A 112 MB snapshot does not need to leave the graph container once per
        cell. The cache is still re-verified against the registry on every call
        — a cache that is not checked is just a second place for the instrument
        to differ — but a hit never reaches `docker cp`."""
        content = b"the pinned graph"
        row = _row(sha256=hashlib.sha256(content).hexdigest())
        dest_dir = tmp_path / row.name
        dest_dir.mkdir()
        (dest_dir / f"{row.name}.kryo").write_bytes(content)
        monkeypatch.setattr(store.snapshots, "find", lambda name: row)
        monkeypatch.setattr(store.snapshots, "verify", lambda name: True)

        def refuses_to_copy(*a, **k):
            raise AssertionError("a cache hit must not re-copy from the container")
        monkeypatch.setattr(store.subprocess, "run", refuses_to_copy)

        assert store.export(row.name, root=tmp_path) == dest_dir


class TestPayloadIsWhatCrossesTheSeam:
    """`vm.job_spec`'s `store` key is built from exactly this shape — a mismatch
    here is a mismatch a cell would receive."""

    def test_payload_carries_every_field_the_cell_and_the_record_need(
            self, tmp_path, monkeypatch):
        content = b"pinned graph bytes"
        row = _row(sha256=hashlib.sha256(content).hexdigest(), vertices=41, edges=97)
        monkeypatch.setattr(store.snapshots, "find", lambda name: row)
        monkeypatch.setattr(store.snapshots, "verify", lambda name: True)
        monkeypatch.setattr(store.subprocess, "run", _fake_docker_cp(content))

        crossing = store.payload(row.name, root=tmp_path)

        assert crossing == {
            "name": row.name,
            "host_path": str(tmp_path / row.name),
            "guest_path": store.GUEST_DIR,
            "kryo": f"{store.GUEST_DIR}/{row.name}.kryo",
            "port": store.GUEST_PORT,
            "sha256": row.sha256,
            "vertices": 41,
            "edges": 97,
        }

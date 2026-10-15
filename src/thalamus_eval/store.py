"""The store a memory arm is GIVEN, exported once and handed to cells as payload.

A memory-on arm needs something to recall from. It used to reach one over a host
port — a pinned snapshot served on 8183, proxied into the cell — and that channel is
gone: the boundary opens no host port, because a knob that can open one is a knob
that can be set wrong. What replaces it is the rule the whole confinement design
rests on: everything conditioning the experiment is supplied rather than reached for.

So a snapshot crosses as a file with a recorded digest and the cell serves it to
itself. Three consequences worth stating, because they are the reason this is better
rather than merely different:

- Two arms now differ only in what they were GIVEN. Under the relay, a memory-on and
  a memory-off cell differed in their network boundary as well, and the isolation
  probe read differently for each — a confound sitting directly on the manipulated
  factor.
- The store becomes a recorded condition. `sha256` is in the run record, so two
  campaigns a month apart can be told apart by what they recalled from rather than
  by whoever remembers which snapshot was being served that week.
- A cell that was given no store must have nothing answering on its own loopback,
  which is a check `_graph_refusals` and the isolation probe can both make. Under the
  relay, "no store" and "the relay was down" looked the same.

The export is cached by digest. A 112 MB snapshot does not need to leave the graph
container once per cell, and the cached copy is re-verified against the registry
before it is handed over — a cache that is not checked is just a second place for the
instrument to differ.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from thalamus.eval import snapshots

#: Where the cell serves the store to itself. The product's own default
#: (`substrate/writer.py`'s `ws://localhost:8182/gremlin`) resolves to the cell's own
#: loopback inside a guest, so an arm that never reads THALAMUS_GRAPH_URL still finds
#: the store it was given and nothing else.
GUEST_PORT = 8182

#: Where the payload lands in the guest. It stays as it arrived: the cell copies the
#: snapshot out of here before serving it, because TinkerGraph flushes memory to
#: `graphLocation` on shutdown and would otherwise rewrite the file whose digest the
#: run record cites.
GUEST_DIR = "/var/lib/cell-store"


class StoreError(RuntimeError):
    """The store cannot be supplied, so a memory arm cannot honestly be run."""


def export(name: str, *, root: Path) -> Path:
    """Put snapshot `name` on the host filesystem, verified. Returns its directory.

    The registry's digest is the authority in both directions: the file is checked
    inside the container before it is copied and checked again after it lands, so a
    cell is never given a store that is not the state its record will name.
    """
    try:
        row = snapshots.find(name)
    except snapshots.SnapshotError as exc:
        raise StoreError(str(exc)) from exc
    if not snapshots.verify(name):
        raise StoreError(
            f"snapshot `{name}` no longer hashes to its registry entry, so it is not "
            "the state that name refers to. A cell given it would recall from "
            "something nobody can cite."
        )
    dest_dir = root / name
    dest = dest_dir / f"{name}.kryo"
    if dest.is_file() and _digest(dest) == row.sha256:
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("docker") is None:
        raise StoreError(
            "docker is needed once to copy the pinned snapshot out of the graph "
            "container; the cached copy is used on every run after that"
        )
    proc = subprocess.run(
        ["docker", "cp", f"{snapshots.CONTAINER}:{row.server_path}", str(dest)],
        capture_output=True, text=True, check=False, timeout=600,
    )
    if proc.returncode != 0:
        raise StoreError(
            f"could not copy `{name}` out of {snapshots.CONTAINER}: "
            f"{proc.stderr.strip()[-300:]}"
        )
    landed = _digest(dest)
    if landed != row.sha256:
        dest.unlink(missing_ok=True)
        raise StoreError(
            f"the copy of `{name}` hashes to {landed[:12]} and the registry says "
            f"{row.sha256[:12]} — the export is not the snapshot"
        )
    return dest_dir


def payload(name: str, *, root: Path) -> dict:
    """What crosses the seam as this cell's store.

    `guest_path` is the directory rather than the file: a workspace is a tree, and
    the cell copies it to somewhere it can write before serving it, because
    TinkerGraph flushes `graphLocation` on shutdown.
    """
    dest_dir = export(name, root=root)
    row = snapshots.find(name)
    return {
        "name": name,
        "host_path": str(dest_dir),
        "guest_path": GUEST_DIR,
        "kryo": f"{GUEST_DIR}/{name}.kryo",
        "port": GUEST_PORT,
        "sha256": row.sha256,
        "vertices": row.vertices,
        "edges": row.edges,
    }


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Stores that are not registry entries: the empty start, and a chain's own exports
# ---------------------------------------------------------------------------

#: A campaign's graph starts EMPTY and grows only from that run's own sessions, so its
#: first input is a snapshot of nothing, pinned by digest like any other — "no memory"
#: is a condition the record has to be able to cite, not an absence it leaves to inference.
EMPTY_NAME = "campaign-empty"
_EMPTY_PORT = 18199


def file_payload(path: Path, *, name: str, sha256: str, vertices: int, edges: int,
                 into: Path) -> dict:
    """Hand a snapshot FILE to a cell as its store, whether or not a registry names it.

    A chain's exports exist nowhere but on the host disk: they are produced by cells,
    not taken from the graph container, so `snapshots.find` has never heard of them.
    The digest is therefore the whole of their identity, and it is checked here against
    the bytes about to cross — a cell is never given a store that is not the file its
    record will cite.

    `into` is the directory that becomes the payload; the file lands in it as
    `<name>.kryo` (hard-linked when the filesystem allows, copied otherwise), because
    the cell copies a payload directory whole and serves a copy of the file inside.
    """
    if _digest(path) != sha256:
        raise StoreError(
            f"{path} does not hash to the {sha256[:12]} it was recorded with, so it is "
            "not the state the chain's last cell exported and a session given it would "
            "recall from something no record describes")
    into.mkdir(parents=True, exist_ok=True)
    dest = into / f"{name}.kryo"
    dest.unlink(missing_ok=True)
    try:
        os.link(path, dest)
    except OSError:
        shutil.copyfile(path, dest)
    return {
        "name": name,
        "host_path": str(into),
        "guest_path": GUEST_DIR,
        "kryo": f"{GUEST_DIR}/{name}.kryo",
        "port": GUEST_PORT,
        "sha256": sha256,
        "vertices": vertices,
        "edges": edges,
    }


def build_empty(dest: Path) -> dict:
    """Write an empty graph with the product's own snapshot call. Returns its record.

    A throwaway graph container on the same image the live graph runs, serving a graph
    that has never been written to, and `io().write()` through the server — the call
    `thalamus snapshot --name` makes, so the file is the format a cell's server loads.
    Counts are read from the graph before the write rather than assumed.
    """
    from thalamus.substrate.snapshot import snapshot
    from thalamus.substrate.writer import close_connection, connect

    if shutil.which("docker") is None:
        raise StoreError("docker is needed once to build the empty snapshot")
    container = "thalamus-empty-build"
    url = f"ws://127.0.0.1:{_EMPTY_PORT}/gremlin"
    conf = snapshots._write_snapshot_conf(EMPTY_NAME, "/data/empty.kryo")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="empty-graph-") as tmp:
        data = Path(tmp)
        data.chmod(0o777)
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)
        proc = subprocess.run(
            ["docker", "run", "--rm", "-d", "--name", container,
             "-p", f"127.0.0.1:{_EMPTY_PORT}:8182", "-v", f"{data}:/data",
             "-v", f"{conf}:/opt/gremlin-server/conf/thalamus:ro",
             "--entrypoint", "/opt/gremlin-server/bin/gremlin-server.sh",
             snapshots.IMAGE, "conf/thalamus/gremlin-server.yaml"],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise StoreError(f"could not start the empty-graph server: {proc.stderr.strip()[-300:]}")
        try:
            snapshots._wait_ready(url, container, 120)
            g = connect(url)
            try:
                vertices = int(g.V().count().next())
                edges = int(g.E().count().next())
                if vertices or edges:
                    raise StoreError(
                        f"the graph meant to be empty holds {vertices} vertices and "
                        f"{edges} edges")
                snapshot(g, "/data/empty.kryo")
            finally:
                close_connection(g)
            shutil.copyfile(data / "empty.kryo", dest)
        finally:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)
    return {"sha256": _digest(dest), "vertices": 0, "edges": 0,
            "bytes": dest.stat().st_size, "image": snapshots.IMAGE,
            "built_at": datetime.now(timezone.utc).isoformat(),
            # Measured: the product's snapshot call writes a graph with no elements as
            # a ZERO-BYTE file, so the digest is the empty-input sha256, and a
            # zero-byte graphLocation loads as an empty graph (count 0) on the live
            # server image. Recorded beside the digest so nobody reads it as a bug.
            "note": "io().write() of an empty graph yields a zero-byte file; a "
                    "zero-byte graphLocation loads as an empty graph"}


def empty_payload(*, root: Path, into: Path) -> dict:
    """The recorded, digest-pinned empty snapshot, as the store a chain's first cell is given.

    Built once under `root` and re-verified against its sidecar on every use, for the
    reason the registry export is: a cache nothing checks is a second place for the
    instrument to differ.
    """
    kryo = root / EMPTY_NAME / f"{EMPTY_NAME}.kryo"
    sidecar = kryo.with_suffix(".json")
    record = None
    if kryo.is_file() and sidecar.is_file():
        record = json.loads(sidecar.read_text())
        if _digest(kryo) != record["sha256"]:
            raise StoreError(
                f"{kryo} no longer hashes to its sidecar, so the empty start a record "
                "would cite is not the one that was built; delete both and rebuild")
    if record is None:
        record = build_empty(kryo)
        sidecar.write_text(json.dumps(record, indent=2))
    return file_payload(kryo, name=EMPTY_NAME, sha256=record["sha256"],
                        vertices=record["vertices"], edges=record["edges"], into=into)

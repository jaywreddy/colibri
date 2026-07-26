"""Subprocess fab-export jobs: registry, status merge, progress, one-at-a-time.

The thing under test is the JOB MECHANICS, not the archive: a real box job
spawns a worker that spends minutes on plate composition and klayout DRC, and
this host may not run that inside a test suite (CLAUDE.md). So every job here
runs a STUB worker — ``export_job.worker_argv`` is monkeypatched to a tiny
script that publishes the same progress records the real worker does and then
writes a two-entry zip. That exercises the parts that can actually break:

  * the progress file is published atomically and readable mid-flight;
  * ``get_status`` merges ``proc.poll()`` with the progress file, so a worker
    that dies without an archive reports ``failed`` — never ``running`` forever;
  * exactly ONE export runs at a time across the host, in BOTH directions: a
    second start is refused while a job lives, the legacy synchronous route is
    refused while a job lives, and a start is refused while the parent process
    is mid-build;
  * stale job dirs are swept when the next job starts (``data/`` is disposable);
  * the download route is 409 before ``done`` and serves the file after.

One test at the bottom is deliberately real but cheap: the legacy synchronous
``GET /export/box/{id}/fab.zip`` still builds and serves an archive, on the same
throwaway geometry ``test_cache_integrity`` uses (one face, coarsest legal weave,
smallest legal plate). That is the regression that matters after moving the build
body out of the endpoint — it is ONE plate compose plus ONE fine-mask build, and
it is the only heavy thing in this file.
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from .test_cache_integrity import _cheap_box_spec

# A stand-in worker. Speaks the same two contracts as app.export_job's real
# worker — progress published tmp+os.replace, archive published tmp+os.replace,
# exit code as the second half of the status signal — and nothing else.
#   argv: <mode> <box_id> <out.zip> <progress.json> [<sentinel path>]
# modes: quick (build now), wait (build once the sentinel file appears),
#        crash (exit non-zero with no archive and no error line),
#        error  (publish an error line, then exit non-zero)
_STUB_WORKER = '''
import json, os, sys, time, zipfile
from pathlib import Path

mode, box_id, out, prog = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sentinel = Path(sys.argv[5]) if len(sys.argv) > 5 else None
seq = 0


def step(phase, face=None, done=0, detail="", error=None):
    global seq
    seq += 1
    payload = {
        "phase": phase, "face": face, "faces_done": done, "faces_total": 1,
        "detail": detail, "updated_at": seq,
    }
    if error is not None:
        payload["error"] = error
    p = Path(prog)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, p)


step("resolve faces", detail="1 face(s)")
if mode == "crash":
    step("fine mask (compose)", face="front")
    sys.exit(3)
if mode == "error":
    step("failed", face="front", error="stub worker refused: no such face plate")
    sys.exit(1)
if sentinel is not None:
    deadline = time.time() + 30.0
    while not sentinel.exists() and time.time() < deadline:
        time.sleep(0.05)
for phase in ("rebuild plate", "svg bake", "fine mask (compose)", "fine mask (DRC heal)"):
    step(phase, face="front")
step("zip", face="front", done=1)
outp = Path(out)
tmp = outp.with_suffix(outp.suffix + ".tmp")
with zipfile.ZipFile(tmp, "w") as zf:
    zf.writestr("box.json", json.dumps({"id": box_id}))
    zf.writestr("CUTLIST.csv", "face,width_mm\\n")
os.replace(tmp, outp)
step("done", done=1, detail="stub archive")
'''


@pytest.fixture
def job_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test job root + box root, and a registry with no state carried in.

    ``jobs_root()`` resolves ``service.DATA_ROOT`` live, so repointing it here
    keeps ``data/export_jobs`` inside tmp. No plate/pattern cache is needed:
    the stub worker never touches one.
    """
    from app import boxes, export_job, service

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(service, "DATA_ROOT", data)
    monkeypatch.setattr(boxes, "BOXES_ROOT", data / "boxes")
    export_job.reset_registry()
    yield tmp_path
    export_job.reset_registry()  # kills any straggler subprocess


@pytest.fixture
def app_client(job_env: Path) -> TestClient:
    from app.main import create_app

    return TestClient(create_app())


def _stub_path(root: Path) -> Path:
    p = root / "stub_worker.py"
    p.write_text(_STUB_WORKER, encoding="utf-8")
    return p


def _use_stub(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    mode: str,
    sentinel: Path | None = None,
) -> None:
    """Point the registry at the stub worker instead of ``python -m app.export_job``."""
    from app import export_job

    stub = _stub_path(root)

    def argv(box_id: str, out_path: Path, progress_path: Path) -> list[str]:
        cmd = [
            sys.executable, str(stub), mode, str(box_id),
            str(out_path), str(progress_path),
        ]
        if sentinel is not None:
            cmd.append(str(sentinel))
        return cmd

    monkeypatch.setattr(export_job, "worker_argv", argv)


def _write_stub_box(box_id: str) -> dict:
    """A box manifest that PASSES the export pre-checks but can never be built.

    ``prepare_box_export`` only needs ``dimensions_um``, a derivable assembly
    block, and a face count, all of which this has. What it deliberately does
    NOT carry is any recoverable plate spec — no ``faces.front.spec`` and no
    box-level ``spec.faces`` entry — so if a build ever escaped the
    one-at-a-time gate it would 409 on the unrebuildable face INSTANTLY instead
    of quietly starting a real plate compose. That keeps a failed assertion in
    this file a failed assertion, never an unplanned heavy compute (CLAUDE.md).
    """
    from app import boxes
    from app.assembly import assembly_summary

    spec = _cheap_box_spec()
    spec.normalize_face_dims()
    spec_data = spec.to_dict()
    spec_data["faces"] = {}
    manifest = {
        "kind": "box",
        "id": box_id,
        "saved": True,
        "spec": spec_data,
        "name": f"Stub {box_id}",
        "faces": {"front": {"id": "0" * 12, "recipe_data": {}}},
        "dimensions_um": {
            "width": spec.width_um,
            "height": spec.height_um,
            "depth": spec.depth_um,
        },
        "assembly": assembly_summary(
            spec.width_um, spec.depth_um, spec.height_um,
            spec.glass.thickness_um, spec.foil, spec.hinge,
        ),
        "content_hash": "stub",
    }
    d = boxes.BOXES_ROOT / box_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "box.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _poll_until(
    client: TestClient, job_id: str, wanted: set[str], timeout: float = 30.0
) -> dict:
    deadline = time.monotonic() + timeout
    last: dict | None = None
    while time.monotonic() < deadline:
        r = client.get(f"/export/jobs/{job_id}")
        assert r.status_code == 200, r.text
        last = r.json()
        if last["status"] in wanted:
            return last
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} never reached {wanted}; last status: {last}")


# ----- progress records -------------------------------------------------------


def test_progress_publishes_atomically_and_counts_up(tmp_path: Path):
    from app.export_job import Progress, read_progress

    path = tmp_path / "progress.json"
    p = Progress(path=path, faces_total=6)
    p.step("svg bake", face="front", detail="preview SVG pair")
    first = read_progress(path)
    assert first is not None
    assert first["phase"] == "svg bake"
    assert first["face"] == "front"
    assert first["faces_done"] == 0
    assert first["faces_total"] == 6
    assert first["detail"] == "preview SVG pair"

    p.face_finished("front")
    p.step("fine mask (compose)", face="back")
    second = read_progress(path)
    assert second is not None
    assert second["faces_done"] == 1
    # updated_at is a counter, not a clock: strictly increasing per publish.
    assert second["updated_at"] > first["updated_at"]
    # No staging file survives a publish (same rename contract as the caches).
    assert sorted(f.name for f in tmp_path.iterdir()) == ["progress.json"]

    p.fail("boom")
    failed = read_progress(path)
    assert failed is not None and failed["phase"] == "failed"
    assert failed["error"] == "boom"


def test_progress_without_a_path_is_a_silent_accumulator(tmp_path: Path):
    """How the legacy in-process export runs the shared build body."""
    from app.export_job import Progress

    p = Progress()
    p.step("zip")
    p.face_finished("front")
    assert p.snapshot()["faces_done"] == 1
    assert list(tmp_path.iterdir()) == []


def test_unreadable_progress_file_reads_as_absent(tmp_path: Path):
    from app.export_job import read_progress

    path = tmp_path / "progress.json"
    path.write_text('{"phase": "svg ba', encoding="utf-8")  # killed mid-write
    assert read_progress(path) is None
    assert read_progress(tmp_path / "missing.json") is None


def test_the_mandated_per_face_phases_are_declared():
    """The UI can only name a phase the worker actually emits."""
    from app.export_job import PHASES

    required = {
        "rebuild plate",
        "svg bake",
        "fine mask (compose)",
        "fine mask (DRC heal)",
        "zip",
    }
    assert required <= set(PHASES)
    assert PHASES[-2:] == ("done", "failed")


# ----- the happy path, end to end through the routes -------------------------


def test_job_runs_to_done_and_serves_its_archive(
    app_client: TestClient, job_env: Path, monkeypatch: pytest.MonkeyPatch
):
    from app import export_job

    _write_stub_box("job-1")
    sentinel = job_env / "go"
    _use_stub(monkeypatch, job_env, "wait", sentinel=sentinel)

    r = app_client.post("/export/box/job-1/fab/start")
    assert r.status_code == 202, r.text
    started = r.json()
    job_id = started["job_id"]
    assert started["poll_url"] == f"/export/jobs/{job_id}"
    assert started["progress"]["faces_total"] == 1

    # Mid-flight: running, and the archive is NOT downloadable yet.
    running = app_client.get(f"/export/jobs/{job_id}").json()
    assert running["status"] == "running"
    early = app_client.get(f"/export/jobs/{job_id}/fab.zip")
    assert early.status_code == 409, early.text
    assert job_id in early.json()["detail"]

    sentinel.write_text("go", encoding="utf-8")
    done = _poll_until(app_client, job_id, {"done", "failed"})
    assert done["status"] == "done", done
    assert done["error"] is None
    assert done["progress"]["phase"] == "done"
    assert done["progress"]["faces_done"] == 1
    assert done["download_url"] == f"/export/jobs/{job_id}/fab.zip"

    rz = app_client.get(f"/export/jobs/{job_id}/fab.zip")
    assert rz.status_code == 200, rz.text
    assert rz.headers["content-type"] == "application/zip"
    names = set(zipfile.ZipFile(io.BytesIO(rz.content)).namelist())
    assert {"box.json", "CUTLIST.csv"} <= names

    # Artifacts live under data/export_jobs/<job_id>/ — a disposable cache slot.
    job_dir = export_job.jobs_root() / job_id
    assert (job_dir / "fab.zip").exists()
    assert (job_dir / "progress.json").exists()
    assert sorted(p.name for p in job_dir.glob("*.tmp")) == []


def test_unknown_box_404s_without_spawning_a_worker(
    app_client: TestClient, job_env: Path, monkeypatch: pytest.MonkeyPatch
):
    from app import export_job

    _use_stub(monkeypatch, job_env, "quick")
    r = app_client.post("/export/box/nope/fab/start")
    assert r.status_code == 404, r.text
    assert not export_job.jobs_root().exists() or list(export_job.jobs_root().iterdir()) == []


def test_unknown_job_id_404s(app_client: TestClient):
    assert app_client.get("/export/jobs/deadbeef").status_code == 404
    assert app_client.get("/export/jobs/deadbeef/fab.zip").status_code == 404


# ----- exactly one export at a time, in both directions ----------------------


def test_a_live_job_blocks_a_second_job_and_the_sync_route(
    app_client: TestClient, job_env: Path, monkeypatch: pytest.MonkeyPatch
):
    """The machine constraint is host-wide: the subprocess IS the heavy slot."""
    _write_stub_box("job-2")
    sentinel = job_env / "go"
    _use_stub(monkeypatch, job_env, "wait", sentinel=sentinel)

    r = app_client.post("/export/box/job-2/fab/start")
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    second = app_client.post("/export/box/job-2/fab/start")
    assert second.status_code == 429, second.text
    assert second.headers.get("Retry-After") == "10"

    # The legacy synchronous route must refuse too — and refuse BEFORE it
    # resolves any face, so nothing composes beside the live worker. This stub
    # box is deliberately unbuildable (see _write_stub_box), so a build that
    # got past the gate would answer 409 here, not 429.
    legacy = app_client.get("/export/box/job-2/fab.zip")
    assert legacy.status_code == 429, legacy.text

    sentinel.write_text("go", encoding="utf-8")
    assert _poll_until(app_client, job_id, {"done", "failed"})["status"] == "done"

    # Once it is finished the slot is free again.
    third = app_client.post("/export/box/job-2/fab/start")
    assert third.status_code == 202, third.text


def test_a_parent_side_build_blocks_a_new_job(
    app_client: TestClient, job_env: Path, monkeypatch: pytest.MonkeyPatch
):
    from app import export_job

    _write_stub_box("job-3")
    _use_stub(monkeypatch, job_env, "quick")
    with export_job.in_process_slot():
        r = app_client.post("/export/box/job-3/fab/start")
        assert r.status_code == 429, r.text
    assert app_client.post("/export/box/job-3/fab/start").status_code == 202


def test_the_slot_is_reentrant_for_one_builder_thread():
    """One export's nested claims (fine masks, plate rebuilds) must not deadlock."""
    from app import export_job

    with export_job.in_process_slot():
        with export_job.in_process_slot():
            assert export_job.slot_is_held()
        assert export_job.slot_is_held()
    assert not export_job.slot_is_held()


# ----- a dead worker is 'failed', never 'running' forever --------------------


def test_worker_that_dies_without_an_archive_reports_failed(
    app_client: TestClient, job_env: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_stub_box("job-4")
    _use_stub(monkeypatch, job_env, "crash")

    r = app_client.post("/export/box/job-4/fab/start")
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    st = _poll_until(app_client, job_id, {"done", "failed"})
    assert st["status"] == "failed", st
    # The worker never said why, so the diagnosis is the exit code + the fact
    # that no archive landed.
    assert st["returncode"] == 3
    assert "3" in st["error"] and "no archive" in st["error"]
    assert st["progress"]["phase"] == "failed"
    assert st["download_url"] is None

    rz = app_client.get(f"/export/jobs/{job_id}/fab.zip")
    assert rz.status_code == 409, rz.text


def test_worker_error_line_is_surfaced_verbatim(
    app_client: TestClient, job_env: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_stub_box("job-5")
    _use_stub(monkeypatch, job_env, "error")

    r = app_client.post("/export/box/job-5/fab/start")
    assert r.status_code == 202, r.text
    st = _poll_until(app_client, r.json()["job_id"], {"done", "failed"})
    assert st["status"] == "failed", st
    assert st["error"] == "stub worker refused: no such face plate"


# ----- disposable-cache housekeeping ----------------------------------------


def test_stale_job_dirs_are_swept_when_the_next_job_starts(
    job_env: Path, monkeypatch: pytest.MonkeyPatch
):
    from app import export_job

    stale = export_job.jobs_root() / "leftover-run"
    stale.mkdir(parents=True)
    (stale / "fab.zip").write_bytes(b"PK\x03\x04 stale")

    _use_stub(monkeypatch, job_env, "quick")
    job = export_job.start_job("job-6", faces_total=1)
    deadline = time.monotonic() + 30.0
    st: dict | None = None
    while time.monotonic() < deadline:
        st = export_job.get_status(job.job_id)
        assert st is not None
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st is not None and st["status"] == "done", st

    assert not stale.exists(), "a previous run's job dir survived the next start"
    assert job.zip_path.exists()

    # cleanup() forgets the job and takes its artifacts with it.
    assert export_job.cleanup(job.job_id) is True
    assert not job.dir.exists()
    assert export_job.get_status(job.job_id) is None
    assert export_job.cleanup(job.job_id) is False


# ----- the legacy synchronous route still works -----------------------------


def test_legacy_sync_endpoint_still_serves_a_zip(isolated_data, monkeypatch):
    """The refactor moved the build body out of the endpoint — it must still build.

    The ONE heavy test in this file: a single-face box on the cheapest legal
    geometry in the catalog (see test_cache_integrity), which is one plate
    compose plus one fine-mask build.
    """
    from app import export_job
    from app.boxes import materialize_box
    from app.main import create_app

    export_job.reset_registry()
    client = TestClient(create_app())
    materialize_box(_cheap_box_spec(), box_id="legacy-sync-1")

    r = client.get("/export/box/legacy-sync-1/fab.zip")
    assert r.status_code == 200, r.content[:400]
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"box.json", "CUTLIST.csv", "ASSEMBLY.md", "FINE_MASKS.json"} <= names
    assert "front/manifest.json" in names
    assert "front/front.svg" in names
    # The interleaved per-face pass must still produce ONE valid root summary
    # covering every face it wrote.
    fine = json.loads(zf.read("FINE_MASKS.json"))
    assert set(fine.get("faces", {})) <= {"front"}
    # Building it did not leave the export slot claimed.
    assert not export_job.slot_is_held()

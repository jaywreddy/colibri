"""Out-of-process fab export — the ``fab.zip`` build as a SUBPROCESS with
pollable progress.

Why a subprocess at all. A cold box export is the heaviest thing this app does:
six plate composes, six SVG bakes, and six merged-region klayout DRC heals,
minutes end to end. Run in a FastAPI worker thread it does two bad things at
once — the request hangs with no way to say "still working", and the klayout /
numpy stretches hold the GIL long enough to starve every other request (the
live preview stops answering). Moving the build into ``python -m app.export_job``
fixes both: the parent stays responsive and only polls a JSON file, and the one
heavy compute lives in a process the OS can schedule (and we can kill).

Three pieces live here:

* ``Progress`` — the progress record the worker publishes at every step, written
  tmp+``os.replace`` so a poller never reads a half-written file. Shape:
  ``{phase, face, faces_done, faces_total, detail, updated_at}`` where
  ``updated_at`` is a monotonically increasing counter (not a clock — the point
  is "is this newer than what I last saw", which a wall clock answers badly on a
  host whose time can step).
* the registry — ``start_job`` / ``get_status`` / ``cleanup``, one in-process
  dict of live jobs. ``get_status`` merges process liveness (``proc.poll()``)
  with the progress file, so a worker that dies without publishing a result
  reports ``failed`` instead of ``running`` forever.
* the process-wide heavy-compute slot (``claim_slot`` / ``in_process_slot``),
  shared with the legacy synchronous export in ``api/export.py``. CLAUDE.md's
  machine constraint is a HOST constraint, not a per-process one: exactly one
  export may be building at a time, and the parent must never do heavy work
  while a worker subprocess is alive. Both directions are enforced here —
  ``start_job`` refuses while the parent holds the slot, and ``claim_slot``
  refuses while a job subprocess is live. Polling a job is NOT heavy work and
  takes no slot.

Job artifacts land in ``data/export_jobs/<job_id>/`` (``fab.zip``,
``progress.json``, ``worker.log``) and follow the disposable-cache rules: every
file is published by rename, and stale job dirs are swept lazily when the next
job starts. Nothing here is required for a later run — delete ``data/`` freely.

One wart worth knowing: ``python -m app.export_job`` runs this file as
``__main__``, and the ``app.api.export`` it then imports imports it AGAIN as
``app.export_job`` — two module objects with independent state inside the child.
Harmless because nothing crosses that line by identity (the ``Progress`` handed
to the build is used duck-typed, and the child's slot bookkeeping all happens in
the ``app.export_job`` copy), but do not add ``isinstance`` checks or
module-global state that the worker and the build body must agree on.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("optics.export_job")

# ``python -m app.export_job`` must import ``app``, so the child runs with the
# backend root as its cwd (and on PYTHONPATH, belt and braces for launchers
# that clear sys.path[0]).
BACKEND_ROOT = Path(__file__).resolve().parent.parent

JOBS_DIRNAME = "export_jobs"
ZIP_NAME = "fab.zip"
PROGRESS_NAME = "progress.json"
WORKER_LOG_NAME = "worker.log"

# How long a caller waits for the in-process slot before being told to retry.
# Long enough to ride out a build that is already finishing its zip, short
# enough that the caller gets an answer instead of a socket held open.
SLOT_WAIT_S = 2.0

# Every phase the progress file can report, in build order. The five per-face
# steps are the ones a UI must be able to name so a long export never looks
# hung; ``starting``/``done``/``failed`` are the terminal bookends.
PHASES: tuple[str, ...] = (
    "starting",
    "resolve faces",
    "rebuild plate",          # plate cache miss — recomposing the face
    "svg bake",               # lazy front/back SVG pair for that face
    "fine mask (cached)",     # fine.gds served from data/plates/<hash>/fine_v<N>
    "fine mask (compose)",    # build_plate_fine vector composition
    "fine mask (DRC heal)",   # merged-region klayout heal (the slow half)
    "zip",                    # box.json + cut list + assembly steps + README
    "done",
    "failed",
)


def jobs_root() -> Path:
    """``data/export_jobs``, resolved live so tests can repoint ``DATA_ROOT``."""
    from . import service

    return service.DATA_ROOT / JOBS_DIRNAME


# --- progress publishing ------------------------------------------------------


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    """Publish a progress snapshot by rename, never in place.

    Same contract as ``service.write_json_atomic`` (which we do not reuse only
    because the worker must be able to write progress for a path outside the
    cache tree): a reader either sees the previous snapshot or the new one,
    never a truncated JSON object.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # Progress is telemetry: losing an update must never fail the build.
        _log.warning("progress write failed path=%s", path, exc_info=True)


def read_progress(path: Path) -> dict[str, Any] | None:
    """Read a progress snapshot, tolerating a read that lands mid-replace.

    ``os.replace`` is atomic, but a reader can still lose the race on Windows
    (the open can fail with a sharing violation while the rename lands), so one
    retry turns a transient miss into the snapshot it was about to read.
    Anything still unreadable after that is reported as absence — the caller
    treats it exactly like a job that has not published yet.
    """
    for attempt in (0, 1):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            if attempt == 0:
                time.sleep(0.05)
                continue
            _log.warning("progress file unreadable (treated as absent): %s", path)
            return None
        return data if isinstance(data, dict) else None
    return None


class Progress:
    """The build's step reporter: accumulates state, publishes on every step.

    ``path=None`` makes every method a no-op accumulator, which is how the
    legacy in-process export runs the same build body with no progress file.
    Not thread-safe and deliberately so — a build is strictly sequential by
    machine constraint, and one writer is the whole point.
    """

    def __init__(self, path: Path | None = None, faces_total: int = 0) -> None:
        self.path = Path(path) if path is not None else None
        self.faces_total = int(faces_total)
        self.faces_done = 0
        self.phase = "starting"
        self.face: str | None = None
        self.detail = ""
        self.error: str | None = None
        # Monotonic counter, not a timestamp: "newer than what I last saw" is
        # the only question a poller asks, and a counter answers it exactly.
        self.updated_at = 0
        self._t0 = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "phase": self.phase,
            "face": self.face,
            "faces_done": self.faces_done,
            "faces_total": self.faces_total,
            "detail": self.detail,
            "updated_at": self.updated_at,
            "elapsed_s": round(time.monotonic() - self._t0, 1),
        }
        if self.error:
            payload["error"] = self.error
        return payload

    def publish(self) -> None:
        self.updated_at += 1
        if self.path is not None:
            write_progress(self.path, self.snapshot())

    def step(self, phase: str, *, face: str | None = None, detail: str = "") -> None:
        """Enter a build phase. ``face=None`` means "not face-scoped"."""
        self.phase = phase
        self.face = face
        self.detail = detail
        self.publish()
        _log.info(
            "export progress phase=%s face=%s %d/%d %s",
            phase, face, self.faces_done, self.faces_total, detail,
        )

    def face_finished(self, face: str | None = None) -> None:
        """One face is fully in the archive (plate files + fab mask)."""
        self.faces_done += 1
        if self.faces_total < self.faces_done:
            self.faces_total = self.faces_done
        self.face = face
        self.publish()

    def done(self, detail: str = "archive ready") -> None:
        self.phase = "done"
        self.face = None
        self.detail = detail
        self.publish()

    def fail(self, error: str) -> None:
        self.phase = "failed"
        self.error = error
        self.detail = error
        self.publish()


def _default_progress(faces_total: int = 0) -> dict[str, Any]:
    """What ``get_status`` reports before the worker's first publish."""
    return {
        "phase": "starting",
        "face": None,
        "faces_done": 0,
        "faces_total": int(faces_total),
        "detail": "worker starting",
        "updated_at": 0,
    }


# --- the process-wide heavy-compute slot ---------------------------------------
# One slot for the whole host: either the parent is building an export in a
# worker thread, or a job subprocess is alive, never both and never two of
# either (CLAUDE.md — this 13.7 GB host has kernel-bugchecked under concurrent
# heavy compute). Reentrant by thread, because one export's nested claims
# (_FineMaskRun.write_face, _rebuild_face_plate) all run in the single worker
# thread handling that request.

_guard = threading.Lock()          # guards _jobs, _slot_owner, _slot_depth
_slot_owner: int | None = None     # thread ident holding the in-process slot
_slot_depth = 0


class ExportBusy(RuntimeError):
    """The single export slot is taken — the caller should retry, not queue."""


@dataclass
class Job:
    """One live export subprocess and the artifacts it owns."""

    job_id: str
    box_id: str
    dir: Path
    zip_path: Path
    progress_path: Path
    log_path: Path
    faces_total: int = 0
    proc: subprocess.Popen[bytes] | None = None
    returncode: int | None = None
    started_at: float = field(default_factory=time.time)
    log_fh: Any = None

    def close_log(self) -> None:
        if self.log_fh is not None:
            try:
                self.log_fh.close()
            except OSError:  # pragma: no cover — closing a dead pipe
                pass
            self.log_fh = None

    def log_tail(self, limit: int = 1200) -> str:
        """Last bytes of the worker's stdout+stderr, for the failure message."""
        try:
            raw = self.log_path.read_bytes()
        except OSError:
            return ""
        return raw[-limit:].decode("utf-8", "replace").strip()


_jobs: dict[str, Job] = {}


def _live_job_locked() -> Job | None:
    """The job whose subprocess is still alive, if any. Caller holds ``_guard``."""
    for job in _jobs.values():
        if job.proc is not None and job.proc.poll() is None:
            return job
    return None


def claim_slot(timeout: float = SLOT_WAIT_S) -> None:
    """Take the heavy-compute slot for this thread, or raise ``ExportBusy``.

    Reentrant for the owning thread. A blocked caller waits only ``timeout``
    and is then told to retry: exports run minutes, and a queued HTTP download
    has no way to say "waiting" — that is exactly what the job routes are for.
    """
    global _slot_owner, _slot_depth
    me = threading.get_ident()
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        with _guard:
            if _slot_owner == me:
                _slot_depth += 1
                return
            live = _live_job_locked()
            if live is not None:
                reason = (
                    f"export job {live.job_id} (box {live.box_id}) is still building — "
                    f"poll GET /export/jobs/{live.job_id} and retry when it finishes"
                )
            elif _slot_owner is not None:
                reason = "an export is already building in this process — retry shortly"
            else:
                _slot_owner = me
                _slot_depth = 1
                return
        if time.monotonic() >= deadline:
            raise ExportBusy(reason)
        time.sleep(0.05)


def release_slot() -> None:
    """Release one nested claim taken by this thread."""
    global _slot_owner, _slot_depth
    me = threading.get_ident()
    with _guard:
        if _slot_owner != me:
            raise RuntimeError("release_slot() from a thread that does not hold the slot")
        _slot_depth -= 1
        if _slot_depth <= 0:
            _slot_depth = 0
            _slot_owner = None


@contextmanager
def in_process_slot(timeout: float = SLOT_WAIT_S) -> Iterator[None]:
    """Hold the heavy-compute slot for an in-process build."""
    claim_slot(timeout)
    try:
        yield
    finally:
        release_slot()


def slot_is_held() -> bool:
    """True if some thread in THIS process is mid-build (diagnostics/tests)."""
    with _guard:
        return _slot_owner is not None


# --- the job registry ---------------------------------------------------------


def worker_argv(box_id: str, out_path: Path, progress_path: Path) -> list[str]:
    """Argv for one worker subprocess.

    A module-level function so tests can swap in a stub script: spawning the
    real worker means minutes of plate composition and klayout DRC, which is
    not something a test suite may do on this host (CLAUDE.md).
    ``sys.executable`` keeps the child in the same interpreter/venv as the
    parent, on Windows too.
    """
    return [
        sys.executable,
        "-m",
        "app.export_job",
        "--box-id", str(box_id),
        "--out", str(out_path),
        "--progress", str(progress_path),
    ]


def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{BACKEND_ROOT}{os.pathsep}{existing}" if existing else str(BACKEND_ROOT)
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _sweep_stale_dirs(keep: set[str]) -> None:
    """Drop job dirs nobody is waiting on. Caller holds ``_guard``.

    Runs when a new job starts (no other job can be live at that moment, so
    nothing in flight is at risk) and keeps exactly the incoming job's dir: an
    export archive is minutes of compute but megabytes on disk, and the client
    downloads it as soon as it is done. ``ignore_errors`` because a previous
    archive may still be open by a slow download on Windows — the next sweep
    gets it.
    """
    root = jobs_root()
    if not root.exists():
        return
    for child in root.iterdir():
        if child.name in keep or not child.is_dir():
            continue
        shutil.rmtree(child, ignore_errors=True)


def start_job(box_id: str, *, faces_total: int = 0) -> Job:
    """Spawn the one allowed export subprocess for ``box_id``.

    Raises ``ExportBusy`` if the parent is mid-export or another job is still
    alive. The parent does NOT take the heavy slot for the job's lifetime —
    liveness of the subprocess is what blocks a second heavy compute (see
    ``claim_slot``), so the parent stays free to serve previews and polls.
    """
    with _guard:
        if _slot_owner is not None:
            raise ExportBusy(
                "an export is already building in this process — retry shortly"
            )
        live = _live_job_locked()
        if live is not None:
            raise ExportBusy(
                f"export job {live.job_id} (box {live.box_id}) is still running — "
                "one export at a time; retry when it finishes"
            )
        # Every finished job is history now: reap its registry entry and sweep
        # its artifacts (lazy cache cleanup, nothing else references them).
        for dead_id in [j.job_id for j in _jobs.values()]:
            _jobs.pop(dead_id).close_log()

        job_id = uuid.uuid4().hex[:12]
        root = jobs_root()
        root.mkdir(parents=True, exist_ok=True)
        _sweep_stale_dirs(keep={job_id})
        job_dir = root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(
            job_id=job_id,
            box_id=box_id,
            dir=job_dir,
            zip_path=job_dir / ZIP_NAME,
            progress_path=job_dir / PROGRESS_NAME,
            log_path=job_dir / WORKER_LOG_NAME,
            faces_total=int(faces_total),
        )
        # Publish "starting" before the spawn so the very first poll — which can
        # arrive before the child has imported anything — sees a real record.
        write_progress(job.progress_path, _default_progress(faces_total))
        argv = worker_argv(box_id, job.zip_path, job.progress_path)
        try:
            job.log_fh = job.log_path.open("wb")
            job.proc = subprocess.Popen(  # noqa: S603 — argv is ours, no shell
                argv,
                cwd=str(BACKEND_ROOT),
                env=_worker_env(),
                stdin=subprocess.DEVNULL,
                stdout=job.log_fh,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            job.close_log()
            shutil.rmtree(job_dir, ignore_errors=True)
            raise RuntimeError(f"could not start the export worker: {exc}") from exc
        _jobs[job_id] = job
    _log.info(
        "export job %s started box=%s faces=%d pid=%s",
        job.job_id, box_id, job.faces_total, getattr(job.proc, "pid", None),
    )
    return job


def get_status(job_id: str) -> dict[str, Any] | None:
    """Merged status for one job, or ``None`` if the id is unknown.

    Liveness first, progress file second — the file is what the worker *said*,
    ``poll()`` is what actually happened. A worker that was killed, crashed on
    import, or died mid-DRC therefore reports ``failed`` with whatever
    diagnosis we have (its own error line, else the exit code + log tail),
    never ``running`` forever. The archive on disk is the authority for
    ``done``: it is published by rename, so its presence means complete.
    """
    with _guard:
        job = _jobs.get(job_id)
    if job is None:
        return None

    rc = job.proc.poll() if job.proc is not None else -1
    progress = read_progress(job.progress_path) or _default_progress(job.faces_total)
    progress = dict(progress)
    error: str | None = None

    if rc is None:
        status = "running"
    else:
        job.returncode = rc
        job.close_log()
        try:
            have_zip = job.zip_path.stat().st_size > 0
        except OSError:  # absent, or swept out from under us
            have_zip = False
        if rc == 0 and have_zip:
            status = "done"
            progress["phase"] = "done"
        else:
            status = "failed"
            progress["phase"] = "failed"
            error = str(progress.get("error") or "").strip() or None
            if error is None:
                what = (
                    "produced no archive" if not have_zip else "left an unusable archive"
                )
                tail = job.log_tail()
                error = f"export worker exited with code {rc} and {what}"
                if tail:
                    error = f"{error}\n{tail}"
    return {
        "job_id": job.job_id,
        "box_id": job.box_id,
        "status": status,
        "progress": progress,
        "error": error,
        "returncode": rc,
        "download_url": (
            f"/export/jobs/{job.job_id}/{ZIP_NAME}" if status == "done" else None
        ),
    }


def result_path(job_id: str) -> Path | None:
    """The finished archive for ``job_id``, or ``None`` if there isn't one."""
    with _guard:
        job = _jobs.get(job_id)
    if job is None:
        return None
    if job.proc is not None and job.proc.poll() is None:
        return None
    return job.zip_path if job.zip_path.exists() else None


def cleanup(job_id: str, *, kill: bool = False) -> bool:
    """Forget a job and delete its artifacts. Refuses a live job unless ``kill``."""
    with _guard:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.proc is not None and job.proc.poll() is None:
            if not kill:
                return False
            job.proc.kill()
            try:
                job.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                _log.warning("export worker %s ignored kill", job_id)
        job.close_log()
        del _jobs[job_id]
    shutil.rmtree(job.dir, ignore_errors=True)
    return True


def reset_registry() -> None:
    """Kill every job, forget them all, drop the slot. Tests and dev reloads."""
    global _slot_owner, _slot_depth
    with _guard:
        jobs = list(_jobs.values())
        _jobs.clear()
        _slot_owner = None
        _slot_depth = 0
    for job in jobs:
        if job.proc is not None and job.proc.poll() is None:
            job.proc.kill()
            try:
                job.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        job.close_log()
        shutil.rmtree(job.dir, ignore_errors=True)


# --- the worker entry point ---------------------------------------------------


def _error_detail(exc: BaseException) -> str:
    """One-line diagnosis for the progress file.

    ``HTTPException.detail`` carries the actionable text the routes already
    write for a missing/unexportable box (404/409), so prefer it over the
    class name — the polling client shows the same message it would have got
    from the synchronous endpoint.
    """
    detail = getattr(exc, "detail", None)
    if isinstance(detail, str) and detail:
        return detail
    return f"{type(exc).__name__}: {exc}"


def _worker_main(argv: list[str] | None = None) -> int:
    """``python -m app.export_job --box-id X --out Y --progress Z``.

    Builds the whole box fab archive with the SAME function the synchronous
    endpoint calls (``api.export.build_box_fab_archive``) — imported here, not
    at module scope, so the parent's ``api.export`` can import this module
    without a cycle. Publishes the archive by rename and the outcome to the
    progress file; the exit code is the second half of the status contract
    (``get_status`` trusts neither alone).
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.export_job",
        description="Build one box's fab.zip out of process, reporting progress.",
    )
    parser.add_argument("--box-id", required=True, help="saved box id to export")
    parser.add_argument("--out", required=True, help="archive path to publish")
    parser.add_argument("--progress", required=True, help="progress JSON path")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    progress = Progress(path=Path(args.progress))
    try:
        from .api.export import build_box_fab_archive

        out = build_box_fab_archive(args.box_id, Path(args.out), progress=progress)
    except Exception as exc:  # noqa: BLE001 — every failure must reach the poller
        detail = _error_detail(exc)
        _log.exception("fab export job failed box=%s", args.box_id)
        progress.fail(detail)
        return 1
    size = out.stat().st_size if out.exists() else 0
    progress.done(f"{size} bytes")
    _log.info("fab export job done box=%s bytes=%d -> %s", args.box_id, size, out)
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised as a subprocess
    raise SystemExit(_worker_main())

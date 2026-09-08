"""Background pipeline runs, started from the web UI.

Each stage runs as a **subprocess** of its own CLI rather than inside the
server. Three reasons: Whisper and the LLM are memory-hungry and get their
memory back cleanly when the process exits, a segfault in a native runtime
cannot take the web server with it, and the thing the browser triggers is
exactly the command you would have typed.

Jobs run one at a time. There is a single GPU, and two transcriptions racing
each other would be slower than doing them in turn.
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import common

# stage id -> (label, script, needs a specific lecture?)
STAGES = {
    "transcribe": ("Transcribe", "stage1_transcribe.py", True),
    "math":       ("Formulas", "stage2_math.py", True),
    "notes":      ("Notes", "stage2_notes.py", True),
    # Not in DEFAULT_STAGES: this is the one stage that calls a paid cloud
    # endpoint, so it is ticked deliberately rather than by default. It must
    # run after "notes", which is what it writes up -- the dict order is the
    # order the checkboxes appear in, and the order they are run in.
    "course_notes": ("Course notes", "stage2_course_notes.py", True),
    "structure":  ("Chapters", "stage3_structure.py", True),
    "index":      ("Search index", "stage6_index.py", False),
}

DEFAULT_STAGES = ["transcribe", "notes", "structure", "index"]


class Job:
    def __init__(self, course_id, lecture_id, stages, options):
        self.id = uuid.uuid4().hex[:12]
        self.course_id = course_id
        self.lecture_id = lecture_id
        self.stages = stages
        self.options = options or {}
        self.status = "queued"          # queued | running | done | failed | cancelled
        self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.started = None
        self.finished = None
        self.current = None
        self.done_stages = []
        self.failed_stages = []
        self.log = deque(maxlen=400)
        self.proc = None
        self.cancelled = False

    def snapshot(self):
        total = len(self.stages) or 1
        done = len(self.done_stages) + len(self.failed_stages)
        return {
            "id": self.id,
            "course_id": self.course_id,
            "lecture_id": self.lecture_id,
            "stages": self.stages,
            "status": self.status,
            "current": self.current,
            "done": self.done_stages,
            "failed": self.failed_stages,
            "progress": round(done / total, 3),
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            # Enough scrollback to actually read what happened; the
            # deque behind it holds 400.
            "log": list(self.log)[-200:],
        }


class Runner:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.jobs = {}
        self.order = deque(maxlen=100)
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.on_finish = None            # set by the server to drop its caches
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------- public --

    def submit(self, course_id, lecture_id, stages, options=None):
        stages = [s for s in (stages or DEFAULT_STAGES) if s in STAGES]
        if not stages:
            raise ValueError("no valid stages requested")
        job = Job(course_id, lecture_id, stages, options)
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.q.put(job.id)
        self.logger.info("job %s queued: %s %s %s",
                         job.id, course_id, lecture_id or "(all)", ",".join(stages))
        return job

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def list(self, limit=20):
        with self.lock:
            ids = list(self.order)[-limit:][::-1]
            return [self.jobs[i].snapshot() for i in ids if i in self.jobs]

    def cancel(self, job_id):
        job = self.get(job_id)
        if not job or job.status in ("done", "failed", "cancelled"):
            return False
        job.cancelled = True
        if job.proc and job.proc.poll() is None:
            try:
                job.proc.terminate()
            except Exception:
                pass
        if job.status == "queued":
            job.status = "cancelled"
            job.finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return True

    def busy(self):
        with self.lock:
            return any(j.status in ("queued", "running") for j in self.jobs.values())

    # -------------------------------------------------------------- worker --

    def _loop(self):
        while True:
            job_id = self.q.get()
            job = self.get(job_id)
            if job is None or job.cancelled:
                continue
            self._run(job)

    def _args_for(self, job, stage):
        _label, script, per_lecture = STAGES[stage]
        argv = [sys.executable, str(common.ROOT / "scripts" / script),
                "--course", job.course_id]
        if per_lecture:
            if job.lecture_id:
                argv += ["--lecture", job.lecture_id]
            else:
                argv += ["--all"]
            if self.options_bool(job, "force"):
                argv += ["--force"]
        return argv

    @staticmethod
    def options_bool(job, key):
        return bool(job.options.get(key))

    def _stage_env(self, job):
        """Per-run overrides land in a scratch config the subprocess reads.

        Nothing mutates config.yaml: two jobs with different models must not
        fight over the file, and a crash must not leave the project configured
        differently from how the user left it.
        """
        overrides = {}
        model = str(job.options.get("model") or "").strip()
        backend = str(job.options.get("backend") or "").strip()
        windowing = str(job.options.get("windowing") or "").strip()
        if model:
            overrides.setdefault("asr", {})["model"] = model
        if backend:
            overrides.setdefault("asr", {})["backend"] = backend
        # Whitelisted rather than passed through: this lands in a config file
        # a subprocess reads, and the only two values the notes pass knows how
        # to act on are these.
        if windowing in ("fixed", "topic"):
            overrides.setdefault("notes", {})["windowing"] = windowing
        if not overrides:
            return None

        cfg = json.loads(json.dumps({k: v for k, v in self.cfg.items() if k != "_root"}))
        for section, vals in overrides.items():
            cfg.setdefault(section, {}).update(vals)

        d = common.project_path(self.cfg, "logs") / "jobcfg"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{job.id}.yaml"
        import yaml
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return path

    def _run(self, job):
        job.status = "running"
        job.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cfg_path = None
        try:
            cfg_path = self._stage_env(job)
        except Exception as exc:
            job.log.append(f"could not write job config: {exc}")

        for stage in job.stages:
            if job.cancelled:
                job.status = "cancelled"
                break
            label = STAGES[stage][0]
            job.current = stage
            job.log.append(f"=== {label} ===")
            argv = self._args_for(job, stage)
            if cfg_path:
                argv += ["--config", str(cfg_path)]

            try:
                rc = self._spawn(job, argv)
            except Exception as exc:
                job.log.append(f"could not start {label}: {exc}")
                rc = -1

            if job.cancelled:
                job.status = "cancelled"
                break
            if rc == 0:
                job.done_stages.append(stage)
            else:
                job.failed_stages.append(stage)
                job.log.append(f"{label} exited with code {rc}")
                # Later stages read the earlier ones, so carrying on after a
                # failure just produces a second, more confusing failure.
                break

        job.current = None
        if job.status != "cancelled":
            job.status = "failed" if job.failed_stages else "done"
        job.finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.logger.info("job %s %s (%s)", job.id, job.status,
                         ",".join(job.done_stages) or "nothing")
        if self.on_finish:
            try:
                self.on_finish(job)
            except Exception:
                pass

    def _spawn(self, job, argv):
        job.proc = subprocess.Popen(
            argv, cwd=str(common.ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in job.proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            # Progress bars and HF chatter would drown the useful lines.
            if "it/s]" in line or line.startswith("Fetching"):
                continue
            job.log.append(line)
        job.proc.wait()
        return job.proc.returncode

"""Content-verified stage commits with separate writable review queues."""

from contextlib import contextmanager
import fcntl
import json
import time
from pathlib import Path

from haireder.data.formal import file_hash
from haireder.utils.io import write_json
from .review import fingerprint


class RunBusy(RuntimeError):
    """Another process owns the writable experiment directory."""


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunBusy(f"An experiment is already writing {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Stages:
    def __init__(self, directory, identity, from_stage=None, checkpoint=None):
        self.directory = Path(directory)
        self.path = (
            Path(checkpoint) if checkpoint else self.directory / "checkpoint.json"
        )
        self.identity = identity
        self.state = {"identity": identity, "stages": {}}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["identity"] != identity:
                raise ValueError(
                    "Configuration, implementation or dataset changed. Select a new run directory."
                )
        self.seek = from_stage

    def run(self, name, group, inputs, action):
        digest = fingerprint(inputs)
        old = self.state["stages"].get(name)
        if self.seek in (name, group):
            self.seek = None
        if old is not None:
            if (
                old["inputs"] != digest
                or fingerprint([self.identity, name, digest, old["artifacts"]])
                != old["commit"]
                or not all(
                    p.is_file() and file_hash(p) == h
                    for key, h in old["artifacts"].items()
                    for p in [self.directory / key]
                )
            ):
                raise ValueError(f"{name} checkpoint or artifact mismatch")
            return old["commit"]
        if self.seek:
            raise ValueError(
                f"--from-stage {self.seek} requires a verified {name} checkpoint"
            )
        started = time.monotonic()
        timing_path = self.directory / ".pending_compute_seconds.json"
        timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        timing_key = fingerprint([self.identity, name, digest])
        try:
            paths = list(action())
        except Exception:
            timings[timing_key] = (
                timings.get(timing_key, 0.0) + time.monotonic() - started
            )
            write_json(timing_path, timings)
            raise
        if not paths or any(not Path(p).is_file() for p in paths):
            raise ValueError(f"{name} did not produce its required files")
        artifacts = {
            str(Path(p).relative_to(self.directory)): file_hash(p) for p in paths
        }
        commit = fingerprint([self.identity, name, digest, artifacts])
        self.state["stages"][name] = dict(
            inputs=digest,
            artifacts=artifacts,
            commit=commit,
            elapsed_seconds=timings.get(timing_key, 0.0) + time.monotonic() - started,
        )
        write_json(self.path, self.state)
        return commit

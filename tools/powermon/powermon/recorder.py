"""Run configuration and the sampling loop that writes samples.csv.

The recorder is clock-injected so tests run hundreds of ticks in no wall time.
It never raises out of a tick: sensor failures are NaN cells, disk errors end
the run cleanly with state ``error`` and whatever was captured stays on disk.
"""

from __future__ import annotations

import csv
import errno
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .sources import ALL_SOURCES, Column, ProbeResult, Source, all_columns

SAMPLES_CSV = "samples.csv"
SENSORS_JSON = "sensors.json"
STATUS_JSON = "status.json"
RUN_JSON = "run.json"
WORKER_LOG = "worker.log"
PID_FILE = "powermon.pid"
EVENTS_CSV = "events.csv"
SUMMARY_JSON = "summary.json"

STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_STOPPED = "stopped"
STATE_ERROR = "error"

Finalizer = Callable[[Path], None]
Logger = Callable[[str], None]


@dataclass
class RunConfig:
    run_dir: str
    duration_s: float = 300.0  # 0 = until stopped
    interval_s: float = 1.0
    label: str = ""
    sources: List[str] = field(default_factory=lambda: list(ALL_SOURCES))
    ipmi_every: int = 1
    fsync_every: int = 60
    demo: bool = False
    plots: str = "auto"  # auto | png | svg
    per_core: bool = True  # record per-core coretemp inputs
    started_at: str = ""

    @property
    def path(self) -> Path:
        return Path(self.run_dir)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Optional[Path] = None) -> Path:
        target = path or (self.path / RUN_JSON)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class SystemClock:
    @staticmethod
    def monotonic() -> float:
        return time.monotonic()

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc).astimezone()

    @staticmethod
    def sleep(seconds: float) -> None:
        time.sleep(seconds)


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.3f}"


def read_status(run_dir: Path) -> Optional[dict]:
    path = Path(run_dir) / STATUS_JSON
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_sensors(run_dir: Path) -> Optional[dict]:
    path = Path(run_dir) / SENSORS_JSON
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Recorder:
    def __init__(
        self,
        config: RunConfig,
        sources: Sequence[Source],
        probed: Sequence[ProbeResult],
        clock=SystemClock(),
        log: Logger = print,
        finalizer: Optional[Finalizer] = None,
    ) -> None:
        self.config = config
        self.sources = list(sources)
        self.probed = list(probed)
        self.clock = clock
        self.log = log
        self.finalizer = finalizer
        self.columns: List[Column] = all_columns(self.sources)
        self.run_dir = config.path
        self.state = STATE_RUNNING
        self.samples = 0
        self.gaps = 0
        self.stop_reason = ""
        self._stop = False
        self._t0 = 0.0
        self._last_row: Dict[str, float] = {}
        self._elapsed = 0.0
        self._fh = None
        self._writer = None
        self.exit_code = 0

    # ----------------------------------------------------------------- control

    def request_stop(self, reason: str = "signal") -> None:
        self._stop = True
        self.stop_reason = reason

    def install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            self.log(f"received signal {signum}, stopping")
            self.request_stop(f"signal {signum}")

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # ------------------------------------------------------------------- files

    def _write_sensors(self) -> None:
        data = {
            "columns": [c.to_dict() for c in self.columns],
            "probe": [p.to_dict() for p in self.probed],
            "interval_s": self.config.interval_s,
            "label": self.config.label,
        }
        (self.run_dir / SENSORS_JSON).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def write_status(self, final: bool = False) -> None:
        errors = {s.id: s.errors for s in self.sources if s.errors}
        last = {k: v for k, v in self._last_row.items() if not (isinstance(v, float) and math.isnan(v))}
        data = {
            "state": self.state,
            "pid": os.getpid(),
            "started_at": self.config.started_at,
            "updated_at": self.clock.now().isoformat(timespec="seconds"),
            "duration_s": self.config.duration_s,
            "interval_s": self.config.interval_s,
            "elapsed_s": round(self._elapsed, 3),
            "samples": self.samples,
            "gaps": self.gaps,
            "errors": errors,
            "columns": len(self.columns),
            "last_row": {k: round(v, 3) for k, v in last.items()},
            "stop_reason": self.stop_reason,
            "final": final,
        }
        tmp = self.run_dir / (STATUS_JSON + ".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self.run_dir / STATUS_JSON)
        except OSError as exc:
            self.log(f"status.json write failed: {exc}")

    def _open_csv(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.run_dir / SAMPLES_CSV, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["timestamp", "elapsed_s"] + [c.name for c in self.columns])
        self._fh.flush()

    def _write_row(self, row: Dict[str, float]) -> None:
        assert self._writer is not None and self._fh is not None
        stamp = self.clock.now().isoformat(timespec="milliseconds")
        values = [stamp, _fmt(self._elapsed)] + [_fmt(row.get(c.name, float("nan"))) for c in self.columns]
        self._writer.writerow(values)
        self._fh.flush()
        if self.config.fsync_every > 0 and self.samples % self.config.fsync_every == 0:
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass

    # -------------------------------------------------------------------- loop

    def _sample(self, now: float) -> Dict[str, float]:
        row: Dict[str, float] = {}
        for src in self.sources:
            try:
                row.update(src.read(now))
            except Exception as exc:  # noqa: BLE001 - a source must never kill the run
                src.errors += 1
                self.log(f"[{src.id}] unexpected error: {exc!r}")
        return row

    def _sleep_until(self, target: float) -> None:
        # Sleep in short slices so a stop request is honoured within ~0.5 s
        # even with long intervals.
        while not self._stop:
            remaining = target - self.clock.monotonic()
            if remaining <= 0:
                return
            self.clock.sleep(min(0.5, remaining))

    def run(self) -> int:
        cfg = self.config
        if not cfg.started_at:
            cfg.started_at = self.clock.now().isoformat(timespec="seconds")
        try:
            self._open_csv()
            self._write_sensors()
        except OSError as exc:
            self.log(f"cannot create run files in {self.run_dir}: {exc}")
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
            self.state = STATE_ERROR
            self.exit_code = 2
            self.write_status(final=True)
            return self.exit_code
        self.log(
            f"recording {len(self.columns)} columns every {cfg.interval_s}s"
            + (f" for {cfg.duration_s:.0f}s" if cfg.duration_s > 0 else " until stopped")
            + f" -> {self.run_dir / SAMPLES_CSV}"
        )
        self.write_status()
        self._t0 = self.clock.monotonic()
        k = 0
        try:
            while not self._stop:
                now = self.clock.monotonic()
                self._elapsed = now - self._t0
                if cfg.duration_s > 0 and self._elapsed >= cfg.duration_s:
                    self.state = STATE_DONE
                    break
                row = self._sample(now)
                self._last_row = row
                self.samples += 1
                self._write_row(row)
                self.write_status()
                k += 1
                next_t = self._t0 + k * cfg.interval_s
                after = self.clock.monotonic()
                if after >= next_t + cfg.interval_s:
                    # Missed at least one whole tick: jump to the next aligned
                    # tick that is still in the future (or exactly now).
                    missed = int(math.ceil((after - next_t) / cfg.interval_s))
                    self.gaps += missed
                    k += missed
                    next_t = self._t0 + k * cfg.interval_s
                    self.log(f"sampling fell behind at sample {self.samples}, skipped {missed} tick(s)")
                self._sleep_until(next_t)
            else:
                self.state = STATE_STOPPED
        except OSError as exc:
            self.state = STATE_ERROR
            self.exit_code = 3
            if exc.errno == errno.ENOSPC:
                self.log("disk full while writing samples.csv; ending run")
            else:
                self.log(f"I/O error while recording: {exc}")
        except Exception as exc:  # noqa: BLE001 - keep the data we have
            self.state = STATE_ERROR
            self.exit_code = 4
            self.log(f"unexpected error while recording: {exc!r}")
        self._finalize()
        return self.exit_code

    def _finalize(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except OSError:
                pass
            try:
                self._fh.close()
            except OSError:
                pass
        self._elapsed = self.clock.monotonic() - self._t0 if self._t0 else self._elapsed
        self.log(f"finished: state={self.state} samples={self.samples} gaps={self.gaps}")
        # Publish the terminal state first (so the report can show it), build
        # the report, and only then flip `final` so `status`/`stop` never see a
        # finished run whose report is still being written.
        self.write_status(final=False)
        if self.finalizer is not None:
            try:
                self.finalizer(self.run_dir)
            except Exception as exc:  # noqa: BLE001 - report failure must not lose the CSV
                self.log(f"report generation failed: {exc!r}")
        self.write_status(final=True)

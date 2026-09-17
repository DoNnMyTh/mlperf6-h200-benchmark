from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Dict

from helpers import FakeClock

from powermon.recorder import (
    PID_FILE,
    RUN_JSON,
    SAMPLES_CSV,
    SENSORS_JSON,
    STATE_DONE,
    STATE_STOPPED,
    STATUS_JSON,
    Recorder,
    RunConfig,
    read_status,
)
from powermon.sources import Column, DemoSource, ProbeResult, Source


class StubSource(Source):
    """Two columns; behaviour scripted per tick via hooks."""

    id = "stub"
    title = "stub"

    def __init__(self, on_tick=None) -> None:
        super().__init__()
        self._columns = [
            Column("stub_power_w", "W", "power", "stub", "x"),
            Column("stub_temp_c", "C", "temp", "stub", "x"),
        ]
        self.tick = 0
        self.on_tick = on_tick

    def probe(self) -> ProbeResult:
        return self._result("ok")

    def read(self, now: float) -> Dict[str, float]:
        self.tick += 1
        if self.on_tick:
            override = self.on_tick(self.tick, now)
            if override is not None:
                return override
        return {"stub_power_w": 100.0 + self.tick, "stub_temp_c": 50.0}


class RecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "run"
        self.logs = []
        self.finalized = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make(self, src: Source, duration: float = 10.0, interval: float = 1.0, **kw) -> Recorder:
        cfg = RunConfig(run_dir=str(self.run_dir), duration_s=duration, interval_s=interval, fsync_every=kw.pop("fsync_every", 0))
        self.clock = FakeClock()
        res = src.probe()
        return Recorder(cfg, [src], [res], clock=self.clock, log=self.logs.append,
                        finalizer=self.finalized.append)

    def rows(self):
        with open(self.run_dir / SAMPLES_CSV, newline="") as fh:
            return list(csv.reader(fh))

    def test_duration_run_writes_expected_rows(self) -> None:
        rec = self.make(StubSource(), duration=10.0)
        rc = rec.run()
        self.assertEqual(rc, 0)
        self.assertEqual(rec.state, STATE_DONE)
        rows = self.rows()
        self.assertEqual(rows[0], ["timestamp", "elapsed_s", "stub_power_w", "stub_temp_c"])
        self.assertEqual(len(rows) - 1, 10)
        self.assertEqual([r[1] for r in rows[1:]], [f"{i:.3f}" for i in range(10)])
        self.assertEqual(rows[1][2], "101.000")
        self.assertTrue(rows[1][0].startswith("2026-01-01T12:00:00"))
        status = read_status(self.run_dir)
        self.assertEqual(status["state"], STATE_DONE)
        self.assertTrue(status["final"])
        self.assertEqual(status["samples"], 10)
        self.assertEqual(status["gaps"], 0)
        self.assertEqual(status["last_row"]["stub_power_w"], 110.0)
        sensors = json.loads((self.run_dir / SENSORS_JSON).read_text())
        self.assertEqual([c["name"] for c in sensors["columns"]], ["stub_power_w", "stub_temp_c"])
        self.assertEqual(sensors["probe"][0]["status"], "ok")
        self.assertEqual(self.finalized, [self.run_dir])
        # Never sleeps longer than the interval and honours the schedule.
        self.assertTrue(all(s <= 1.0 for s in self.clock.sleeps))

    def test_nan_becomes_blank_cell(self) -> None:
        def hook(tick, now):
            if tick == 3:
                return {"stub_power_w": float("nan"), "stub_temp_c": 51.0}
            return None

        rec = self.make(StubSource(hook), duration=4.0)
        rec.run()
        rows = self.rows()
        self.assertEqual(rows[3][2], "")
        self.assertEqual(rows[3][3], "51.000")

    def test_gap_detection_when_sampling_is_slow(self) -> None:
        def hook(tick, now):
            if tick == 2:
                self.clock.advance(2.6)  # sample took 2.6 s -> misses 2 ticks
            return None

        rec = self.make(StubSource(hook), duration=8.0)
        rec.run()
        self.assertEqual(rec.gaps, 2)
        rows = self.rows()
        elapsed = [float(r[1]) for r in rows[1:]]
        self.assertEqual(elapsed[:2], [0.0, 1.0])
        self.assertAlmostEqual(elapsed[2], 4.0)
        self.assertEqual(len(elapsed), 6)  # 0,1,4,5,6,7
        self.assertTrue(any("fell behind" in m for m in self.logs))

    def test_request_stop_finalizes_as_stopped(self) -> None:
        holder = {}

        def hook(tick, now):
            if tick == 4:
                holder["rec"].request_stop("test")
            return None

        rec = self.make(StubSource(hook), duration=0.0)  # until stopped
        holder["rec"] = rec
        rc = rec.run()
        self.assertEqual(rc, 0)
        self.assertEqual(rec.state, STATE_STOPPED)
        self.assertEqual(rec.samples, 4)
        self.assertEqual(read_status(self.run_dir)["stop_reason"], "test")
        self.assertEqual(self.finalized, [self.run_dir])

    def test_source_exception_is_counted_not_fatal(self) -> None:
        def hook(tick, now):
            if tick == 2:
                raise RuntimeError("sensor exploded")
            return None

        src = StubSource(hook)
        rec = self.make(src, duration=3.0)
        rec.run()
        self.assertEqual(rec.state, STATE_DONE)
        self.assertEqual(src.errors, 1)
        self.assertEqual(read_status(self.run_dir)["errors"], {"stub": 1})
        rows = self.rows()
        self.assertEqual(rows[2][2], "")

    def test_finalizer_failure_does_not_lose_data(self) -> None:
        def bad_finalizer(_d):
            raise RuntimeError("no report for you")

        cfg = RunConfig(run_dir=str(self.run_dir), duration_s=2.0, fsync_every=0)
        src = StubSource()
        rec = Recorder(cfg, [src], [src.probe()], clock=FakeClock(), log=self.logs.append, finalizer=bad_finalizer)
        self.assertEqual(rec.run(), 0)
        self.assertEqual(len(self.rows()), 3)
        self.assertTrue(any("report generation failed" in m for m in self.logs))

    def test_demo_source_end_to_end(self) -> None:
        rec = self.make(DemoSource(), duration=30.0)
        rec.run()
        rows = self.rows()
        self.assertEqual(len(rows) - 1, 30)
        self.assertEqual(len(rows[0]), 2 + 9)
        self.assertTrue(all(cell != "" for cell in rows[5]))

    def test_config_round_trip(self) -> None:
        cfg = RunConfig(run_dir=str(self.run_dir), duration_s=5, interval_s=0.5, label="x", sources=["rapl"], demo=True)
        path = cfg.save()
        self.assertEqual(path.name, RUN_JSON)
        loaded = RunConfig.load(path)
        self.assertEqual(loaded, cfg)
        # Unknown keys from a newer version are ignored.
        data = json.loads(path.read_text())
        data["future_field"] = 1
        self.assertEqual(RunConfig.from_dict(data), cfg)

    def test_unwritable_run_dir_is_error_state(self) -> None:
        cfg = RunConfig(run_dir=str(Path(self._tmp.name) / "file_not_dir" / "x"))
        Path(self._tmp.name, "file_not_dir").write_text("i am a file")
        src = StubSource()
        rec = Recorder(cfg, [src], [src.probe()], clock=FakeClock(), log=self.logs.append)
        self.assertEqual(rec.run(), 2)
        self.assertEqual(rec.state, "error")


if __name__ == "__main__":
    unittest.main()

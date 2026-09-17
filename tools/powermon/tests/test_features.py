"""Tests for the v0.2 features: events/marks, watch, compare, per-core split, sudo registry, summary.json."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from helpers import TOOL_ROOT, FakeClock, FakeSysfs, write

from powermon import cli, plots, report
from powermon.recorder import EVENTS_CSV, SUMMARY_JSON, Recorder, RunConfig, read_status
from powermon.sources import DemoSource, HwmonSource, is_per_core_column, is_per_core_label
from powermon.stats import load_events, summarize

SHIM = TOOL_ROOT / "powermon.py"


def record_demo(run_dir: Path, seconds: int = 60, label: str = "") -> None:
    cfg = RunConfig(run_dir=str(run_dir), duration_s=seconds, fsync_every=0, label=label)
    src = DemoSource()
    Recorder(cfg, [src], [src.probe()], clock=FakeClock(), log=lambda m: None,
             finalizer=lambda d: report.generate(d, "svg")).run()


class PerCoreTest(unittest.TestCase):
    def test_label_and_column_detection(self) -> None:
        self.assertTrue(is_per_core_label("Core 12"))
        self.assertTrue(is_per_core_label("core_3"))
        self.assertFalse(is_per_core_label("Package id 0"))
        self.assertFalse(is_per_core_label("Composite"))
        self.assertTrue(is_per_core_column("coretemp_core_12_c"))
        self.assertTrue(is_per_core_column("coretemp_2_core_3_c"))
        self.assertFalse(is_per_core_column("coretemp_package_id_0_c"))
        self.assertFalse(is_per_core_column("gpu0_temp_c"))

    def test_skip_per_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fs = FakeSysfs(Path(tmp))
            fs.hwmon(0, "coretemp", temps={1: (45000, "Package id 0"), 2: (41000, "Core 0"), 3: (42000, "Core 1")})
            src = HwmonSource(Path(tmp), skip_per_core=True)
            res = src.probe()
            self.assertEqual([c.name for c in src.columns], ["coretemp_package_id_0_c"])
            self.assertIn("2 per-core temps skipped", res.note)
            src = HwmonSource(Path(tmp))
            src.probe()
            self.assertEqual(len(src.columns), 3)


class EventsAndSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run = Path(self._tmp.name) / "run"
        record_demo(self.run, label="evt")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_mark_and_report_sections(self) -> None:
        at = cli.add_mark(self.run, "training start", at=10)
        self.assertEqual(at, 10)
        cli.add_mark(self.run, "eval, with comma", at=40.5)
        events = load_events(self.run)
        self.assertEqual([(e.elapsed_s, e.text) for e in events], [(10.0, "training start"), (40.5, "eval, with comma")])
        report.generate(self.run, "svg")
        md = (self.run / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Events", md)
        self.assertIn("training start", md)
        self.assertIn("| events (marks) | 2 |", md)
        svg = (self.run / "power.svg").read_text(encoding="utf-8")
        self.assertEqual(svg.count('stroke-dasharray="4 3"'), 2)
        self.assertIn("training start", svg)
        html = (self.run / "report.html").read_text(encoding="utf-8")
        self.assertIn("<h2>Events</h2>", html)
        text = report.summary_text(report.load_summary(self.run))
        self.assertIn("Events", text)

    def test_mark_without_start_time_needs_at(self) -> None:
        if (self.run / "run.json").exists():
            (self.run / "run.json").unlink()
        status = json.loads((self.run / "status.json").read_text())
        status.pop("started_at", None)
        write(self.run / "status.json", json.dumps(status))
        with self.assertRaises(ValueError):
            cli.add_mark(self.run, "x")

    def test_summary_json(self) -> None:
        data = json.loads((self.run / SUMMARY_JSON).read_text(encoding="utf-8"))
        self.assertEqual(data["label"], "evt")
        self.assertEqual(data["state"], "done")
        self.assertEqual(data["samples"], 60)
        self.assertEqual(data["headline_power"]["name"], "system_w")
        self.assertIsInstance(data["headline_power"]["energy_wh"], float)
        names = [c["name"] for c in data["columns"]]
        self.assertIn("gpu0_power_w", names)
        self.assertEqual([d["name"] for d in data["derived"]], ["total_gpu_w"])
        self.assertIn("+", data["start"]) if "+" in data["start"] else self.assertTrue(data["start"])

    def test_report_time_keeps_offset(self) -> None:
        md = (self.run / "report.md").read_text(encoding="utf-8")
        # FakeClock is UTC -> "+00:00" is kept
        self.assertIn("| start | 2026-01-01 12:00:00 +00:00 |", md)

    def test_compare(self) -> None:
        other = Path(self._tmp.name) / "other"
        record_demo(other, seconds=30, label="short")
        a = report.load_summary(self.run)
        b = report.load_summary(other)
        text = report.compare_text(a, b)
        self.assertIn("powermon compare", text)
        self.assertIn("system_w", text)
        self.assertIn("energy Wh A -> B", text)
        md = report.compare_markdown(a, b)
        self.assertIn("| duration | 59s | 29s |", md)
        self.assertIn("## Columns", md)


class ChartPlanTest(unittest.TestCase):
    def test_per_core_split_and_top_n(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            record_demo(run)
            summary = summarize(run)
            n = summary.data.n
            from powermon.stats import ColumnMeta

            for i in range(plots.MAX_SERIES + 10):
                name = f"coretemp_core_{i}_c"
                summary.data.columns.append(name)
                summary.data.values[name] = [20.0 + i] * n
                summary.meta[name] = ColumnMeta(name, "C", "temp", label=f"Core {i}")
            charts = {c.key: c for c in plots.build_charts(summary)}
            self.assertIn("core_temps", charts)
            self.assertEqual([s.name for s in charts["temps"].series], ["coretemp_package_id_0_c"])
            core = charts["core_temps"]
            self.assertEqual(len(core.series), plots.MAX_SERIES)
            # Highest series kept: the hottest core index survives, the coldest does not
            names = [s.name for s in core.series]
            self.assertIn(f"coretemp_core_{plots.MAX_SERIES + 9}_c", names)
            self.assertNotIn("coretemp_core_0_c", names)
            self.assertIn("highest", core.note)


class SudoRegistryTest(unittest.TestCase):
    def test_state_dir_precedence(self) -> None:
        old = dict(os.environ)
        try:
            os.environ["POWERMON_STATE_DIR"] = "/tmp/x"
            self.assertEqual(cli.state_dir(), Path("/tmp/x"))
            os.environ.pop("POWERMON_STATE_DIR")
            os.environ["XDG_STATE_HOME"] = "/tmp/xdg"
            os.environ.pop("SUDO_USER", None)
            self.assertEqual(cli.state_dir(), Path("/tmp/xdg/powermon"))
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_sudo_ids_none_when_not_root(self) -> None:
        if os.name == "posix" and os.geteuid() == 0:
            self.skipTest("running as root")
        os.environ["SUDO_USER"] = "somebody"
        try:
            self.assertIsNone(cli.sudo_ids())
            cli.chown_to_sudo_user(Path("/nonexistent"))  # must be a no-op, never raise
        finally:
            os.environ.pop("SUDO_USER", None)


class SelfCmdTest(unittest.TestCase):
    def test_launcher_env_used_in_hints(self) -> None:
        old = os.environ.get("POWERMON_LAUNCHER")
        os.environ["POWERMON_LAUNCHER"] = str(Path.cwd() / "tools" / "powermon.sh")
        try:
            self.assertEqual(cli._self_cmd().replace("sudo ", ""), "./tools/powermon.sh")
        finally:
            if old is None:
                os.environ.pop("POWERMON_LAUNCHER")
            else:
                os.environ["POWERMON_LAUNCHER"] = old

    def test_watch_line(self) -> None:
        line = cli.watch_line({"state": "running", "duration_s": 60, "elapsed_s": 5, "samples": 5, "gaps": 0,
                               "last_row": {"a_w": 10.0, "b_w": 300.0, "t_c": 55.0}})
        self.assertTrue(line.startswith("running 00:00:05/00:01:00  n=5 gaps=0  b_w=300W  a_w=10W  t_c=55C"))


@unittest.skipUnless(os.name == "posix", "subprocess workers need POSIX")
class EndToEndFeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.env = dict(os.environ, POWERMON_STATE_DIR=str(self.tmp / "state"), PYTHONDONTWRITEBYTECODE="1")
        self.env.pop("SUDO_USER", None)

    def tearDown(self) -> None:
        subprocess.run([sys.executable, str(SHIM), "stop", "--all"], env=self.env, capture_output=True, text=True)
        self._tmp.cleanup()

    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(SHIM), *args], capture_output=True, text=True,
                              env=self.env, timeout=60, cwd=str(self.tmp))

    def wait_for(self, predicate, timeout: float = 30) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.2)
        return False

    def test_mark_watch_compare_flow(self) -> None:
        out = self.tmp / "out"
        proc = self.run_cli("--plots", "svg", "start", "--demo", "--duration", "0", "--interval", "0.5", "--out", str(out), "--yes")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("mark \"training start\"", proc.stdout)
        run_dir = next(out.glob("run_*"))
        self.assertTrue(self.wait_for(lambda: (read_status(run_dir) or {}).get("samples", 0) >= 3))
        # mark by registry (single running run)
        proc = self.run_cli("mark", "phase", "one")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("marked 'phase one'", proc.stdout)
        self.assertTrue((run_dir / EVENTS_CSV).exists())
        # watch --once
        proc = self.run_cli("watch", "--once")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("running", proc.stdout)
        self.assertIn("W", proc.stdout)
        proc = self.run_cli("stop")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("Events", proc.stdout)
        # mark a finished run with --at regenerates the report
        proc = self.run_cli("mark", "--at", "1", "--run-dir", str(run_dir), "late", "note")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("report regenerated", proc.stdout)
        self.assertIn("late note", (run_dir / "report.md").read_text(encoding="utf-8"))
        # watch on a finished run prints the summary and exits
        proc = self.run_cli("watch", str(run_dir))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("powermon summary", proc.stdout)
        # second run for compare
        proc = self.run_cli("--plots", "svg", "start", "--demo", "--duration", "2", "--interval", "0.5", "--out", str(out), "--yes", "--foreground")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        runs = sorted(out.glob("run_*"))
        self.assertEqual(len(runs), 2)
        proc = self.run_cli("compare", str(runs[0]), str(runs[1]), "-o", str(self.tmp / "cmp.md"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("delta mean", proc.stdout)
        self.assertTrue((self.tmp / "cmp.md").exists())
        self.assertTrue((runs[0] / SUMMARY_JSON).exists())

    def test_no_per_core_flag_accepted(self) -> None:
        proc = self.run_cli("probe", "--demo", "--no-per-core")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        proc = self.run_cli("--plots", "svg", "start", "--demo", "--duration", "1", "--interval", "0.5", "--no-per-core", "--out", str(self.tmp / "o"), "--yes", "--foreground")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        cfg = json.loads(next((self.tmp / "o").glob("run_*/run.json")).read_text())
        self.assertFalse(cfg["per_core"])


if __name__ == "__main__":
    unittest.main()

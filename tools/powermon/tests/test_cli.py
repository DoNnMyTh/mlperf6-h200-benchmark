from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from helpers import TOOL_ROOT

from powermon import cli
from powermon.recorder import read_status

SHIM = TOOL_ROOT / "powermon.py"


class ParseTest(unittest.TestCase):
    def test_duration(self) -> None:
        self.assertEqual(cli.parse_duration("300"), 300)
        self.assertEqual(cli.parse_duration("90s"), 90)
        self.assertEqual(cli.parse_duration("5m"), 300)
        self.assertEqual(cli.parse_duration("1.5h"), 5400)
        self.assertEqual(cli.parse_duration("1d"), 86400)
        self.assertEqual(cli.parse_duration(" 0 "), 0)
        for bad in ("", "abc", "5x", "-1"):
            with self.assertRaises(ValueError):
                cli.parse_duration(bad)

    def test_interval(self) -> None:
        self.assertEqual(cli.parse_interval("1"), 1.0)
        self.assertEqual(cli.parse_interval("0.5"), 0.5)
        for bad in ("0.1", "4000", "x"):
            with self.assertRaises(ValueError):
                cli.parse_interval(bad)

    def test_sources(self) -> None:
        self.assertEqual(cli.parse_sources(None), list(cli.ALL_SOURCES))
        self.assertEqual(cli.parse_sources("ALL"), list(cli.ALL_SOURCES))
        self.assertEqual(cli.parse_sources("rapl, nvidia"), ["rapl", "nvidia"])
        with self.assertRaises(ValueError):
            cli.parse_sources("rapl,bogus")

    def test_fmt_hms(self) -> None:
        self.assertEqual(cli.fmt_hms(3661), "01:01:01")
        self.assertEqual(cli.fmt_hms(-5), "00:00:00")

    def test_parser_help_and_version(self) -> None:
        p = cli.build_parser()
        args = p.parse_args(["start", "-d", "5m", "-o", "/tmp/x", "--demo", "-y"])
        self.assertEqual(args.duration, "5m")
        self.assertTrue(args.demo)
        self.assertEqual(args.func, cli.cmd_start)


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._old = os.environ.get("POWERMON_STATE_DIR")
        os.environ["POWERMON_STATE_DIR"] = str(self.tmp / "state")

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("POWERMON_STATE_DIR", None)
        else:
            os.environ["POWERMON_STATE_DIR"] = self._old
        self._tmp.cleanup()

    def test_add_prune_and_cap(self) -> None:
        self.assertEqual(cli.registry_load(), [])
        live = self.tmp / "live"
        live.mkdir()
        cli.registry_add(live, 12345, "a")
        cli.registry_add(self.tmp / "gone", 1, "b")
        self.assertEqual(len(cli.registry_load()), 2)
        kept = cli.registry_prune()
        self.assertEqual([e["run_dir"] for e in kept], [str(live)])
        for i in range(cli.REGISTRY_MAX + 5):
            d = self.tmp / f"r{i}"
            d.mkdir()
            cli.registry_add(d, i, "")
        self.assertEqual(len(cli.registry_load()), cli.REGISTRY_MAX)

    def test_pid_alive(self) -> None:
        self.assertTrue(cli.pid_alive(os.getpid()) in (True, False))  # depends on /proc cmdline content
        self.assertFalse(cli.pid_alive(0))
        self.assertFalse(cli.pid_alive(2**22 - 1))

    def test_make_run_dir_label_sanitised(self) -> None:
        d = cli.make_run_dir(self.tmp, "my bench/run!", None)
        self.assertTrue(d.name.startswith("run_"))
        self.assertTrue(d.name.endswith("_my_bench_run"))
        explicit = cli.make_run_dir(self.tmp, "", str(self.tmp / "exact"))
        self.assertEqual(explicit, (self.tmp / "exact").resolve())

    def test_check_out_dir(self) -> None:
        self.assertIsNone(cli.check_out_dir(self.tmp / "new", create=True))
        self.assertTrue((self.tmp / "new").is_dir())
        self.assertIn("does not exist", cli.check_out_dir(self.tmp / "nope", create=False))
        f = self.tmp / "file"
        f.write_text("x")
        self.assertIn("not a directory", cli.check_out_dir(f, create=True))


@unittest.skipUnless(os.name == "posix", "background worker uses POSIX signals")
class EndToEndTest(unittest.TestCase):
    """Real subprocesses: start in background, status, stop, report."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.env = dict(os.environ, POWERMON_STATE_DIR=str(self.tmp / "state"), PYTHONDONTWRITEBYTECODE="1")

    def tearDown(self) -> None:
        # Make sure no worker outlives the test.
        for entry in self._registry():
            pid = int(entry.get("pid", 0))
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        self._tmp.cleanup()

    def _registry(self):
        try:
            return json.loads((self.tmp / "state" / "active.json").read_text())
        except (OSError, ValueError):
            return []

    def run_cli(self, *args: str, stdin: str = "", timeout: float = 60) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SHIM), *args], input=stdin, capture_output=True, text=True,
            env=self.env, timeout=timeout, cwd=str(self.tmp),
        )

    def wait_for(self, predicate, timeout: float = 30, step: float = 0.2) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(step)
        return False

    def single_run_dir(self, out: Path) -> Path:
        runs = sorted(p for p in out.iterdir() if p.name.startswith("run_"))
        self.assertEqual(len(runs), 1, runs)
        return runs[0]

    def test_background_run_completes_and_reports(self) -> None:
        out = self.tmp / "out"
        proc = self.run_cli("--plots", "svg", "start", "--demo", "--duration", "3", "--interval", "0.5",
                            "--out", str(out), "--label", "e2e", "--yes")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("Started. PID", proc.stdout)
        run_dir = self.single_run_dir(out)
        self.assertTrue(run_dir.name.endswith("_e2e"))
        self.assertTrue((run_dir / "powermon.pid").exists())
        self.assertTrue(self.wait_for(lambda: (read_status(run_dir) or {}).get("final")), read_status(run_dir))
        status = read_status(run_dir)
        self.assertEqual(status["state"], "done")
        self.assertGreaterEqual(status["samples"], 5)
        self.assertTrue(self.wait_for(lambda: (run_dir / "report.html").exists()))
        for name in ("samples.csv", "sensors.json", "status.json", "worker.log", "run.json", "report.md", "report.html", "power.svg"):
            self.assertTrue((run_dir / name).exists(), name)
        # status command on a finished run prints the summary
        proc = self.run_cli("status", str(run_dir))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("state    done", proc.stdout)
        self.assertIn("powermon summary", proc.stdout)
        # registry-based status (no args)
        proc = self.run_cli("status")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(str(run_dir), proc.stdout)
        # report regeneration works and returns 0
        proc = self.run_cli("report", str(run_dir / "samples.csv"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("gpu0_power_w", proc.stdout)

    def test_stop_ends_early_and_still_reports(self) -> None:
        out = self.tmp / "out"
        proc = self.run_cli("--plots", "svg", "start", "--demo", "--duration", "0", "--interval", "0.5",
                            "--out", str(out), "--yes")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        run_dir = self.single_run_dir(out)
        self.assertTrue(self.wait_for(lambda: (read_status(run_dir) or {}).get("samples", 0) >= 3))
        # A second start into the same directory is refused while running.
        proc = self.run_cli("start", "--demo", "--run-dir", str(run_dir), "--yes")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("already", proc.stdout)
        proc = self.run_cli("stop", str(run_dir))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        status = read_status(run_dir)
        self.assertEqual(status["state"], "stopped")
        self.assertTrue(status["final"])
        self.assertTrue((run_dir / "report.md").exists())
        self.assertIn("powermon summary", proc.stdout)
        # stopping again is a no-op with rc 0
        proc = self.run_cli("stop", str(run_dir))
        self.assertEqual(proc.returncode, 0)
        self.assertIn("not running", proc.stdout)

    def test_stop_without_args_uses_registry(self) -> None:
        out = self.tmp / "out"
        self.run_cli("--plots", "svg", "start", "--demo", "--duration", "0", "--out", str(out), "--yes")
        run_dir = self.single_run_dir(out)
        self.assertTrue(self.wait_for(lambda: (read_status(run_dir) or {}).get("samples", 0) >= 2))
        proc = self.run_cli("stop")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(read_status(run_dir)["state"], "stopped")

    def test_foreground_run(self) -> None:
        out = self.tmp / "fg"
        proc = self.run_cli("--plots", "svg", "start", "--demo", "--duration", "2", "--interval", "0.5",
                            "--out", str(out), "--foreground")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("powermon summary", proc.stdout)
        run_dir = self.single_run_dir(out)
        self.assertEqual(read_status(run_dir)["state"], "done")

    def test_probe_demo_and_non_tty_wizard(self) -> None:
        proc = self.run_cli("probe", "--demo")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("[ok]", proc.stdout)
        self.assertIn("9 column(s) would be recorded", proc.stdout)
        proc = self.run_cli()  # no args, stdin not a tty
        self.assertEqual(proc.returncode, 2)
        self.assertIn("stdin is not a terminal", proc.stdout)

    def test_bad_arguments(self) -> None:
        proc = self.run_cli("start", "--duration", "5x", "--yes")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid duration", proc.stdout)
        proc = self.run_cli("report", str(self.tmp / "missing"))
        self.assertEqual(proc.returncode, 2)
        empty = self.tmp / "empty"
        empty.mkdir()
        proc = self.run_cli("report", str(empty))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("no usable data", proc.stdout)


if __name__ == "__main__":
    unittest.main()

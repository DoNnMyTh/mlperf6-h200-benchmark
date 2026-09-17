from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helpers import FakeClock, write

from powermon import report
from powermon.recorder import Recorder, RunConfig
from powermon.sources import DemoSource


def record_demo(run_dir: Path, seconds: int = 60, finalize: bool = True) -> Recorder:
    cfg = RunConfig(run_dir=str(run_dir), duration_s=seconds, fsync_every=0, label="unit test")
    src = DemoSource()
    rec = Recorder(cfg, [src], [src.probe()], clock=FakeClock(), log=lambda m: None,
                   finalizer=(lambda d: report.generate(d, "svg")) if finalize else None)
    rec.run()
    return rec


class GenerateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run = Path(self._tmp.name) / "run"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_full_report_from_finalizer(self) -> None:
        record_demo(self.run)
        md = (self.run / report.REPORT_MD).read_text(encoding="utf-8")
        html = (self.run / report.REPORT_HTML).read_text(encoding="utf-8")
        for name in ("cpu_pkg0_w", "gpu0_power_w", "gpu1_temp_c", "system_w", "total_gpu_w"):
            self.assertIn(name, md)
            self.assertIn(name, html)
        self.assertIn("| state | done |", md)
        self.assertIn("unit test", md)
        self.assertIn("energy Wh", md)
        self.assertIn("## Graphs", md)
        self.assertIn("![Power](power.svg)", md)
        self.assertIn("demo (synthetic)", md)  # sensor probe table
        self.assertIn("<svg", html)
        self.assertIn("<title>powermon report - unit test</title>", html)
        for key in ("power", "gpu_power", "gpu_temps", "temps", "util"):
            self.assertTrue((self.run / f"{key}.svg").exists(), key)

    def test_partial_run_marked_incomplete(self) -> None:
        record_demo(self.run, finalize=False)
        status = json.loads((self.run / "status.json").read_text())
        status["state"] = "stopped"
        write(self.run / "status.json", json.dumps(status))
        paths = report.generate(self.run, "svg")
        self.assertFalse(paths.nodata)
        md = paths.markdown.read_text(encoding="utf-8")
        self.assertIn("incomplete", md)
        self.assertIn("stopped", md)

    def test_truncated_csv_still_reports(self) -> None:
        record_demo(self.run, finalize=False)
        csv_path = self.run / "samples.csv"
        text = csv_path.read_text(encoding="utf-8")
        csv_path.write_text(text[: len(text) - 17], encoding="utf-8")  # chop mid-row
        paths = report.generate(self.run, "svg")
        self.assertFalse(paths.nodata)
        self.assertIn("dropped malformed rows | 1", paths.markdown.read_text(encoding="utf-8"))

    def test_nodata(self) -> None:
        self.run.mkdir(parents=True)
        write(self.run / "samples.csv", "timestamp,elapsed_s,x_w\n")
        paths = report.generate(self.run, "svg")
        self.assertTrue(paths.nodata)
        self.assertIsNone(paths.html)
        self.assertIn("No usable data", paths.markdown.read_text(encoding="utf-8"))
        paths = report.generate(Path(self._tmp.name) / "nowhere", "svg")
        self.assertTrue(paths.nodata)

    def test_report_from_explicit_csv_path(self) -> None:
        record_demo(self.run, finalize=False)
        other = Path(self._tmp.name) / "elsewhere"
        other.mkdir()
        paths = report.generate(other, "svg", csv_path=self.run / "samples.csv")
        self.assertFalse(paths.nodata)
        self.assertTrue((other / report.REPORT_HTML).exists())

    def test_summary_text(self) -> None:
        record_demo(self.run)
        summary = report.load_summary(self.run)
        text = report.summary_text(summary)
        self.assertIn("powermon summary", text)
        self.assertIn("samples", text)
        self.assertIn("Power", text)
        self.assertIn("gpu0_power_w", text)
        self.assertIn("report.html", text)

    def test_png_embedded_when_available(self) -> None:
        from powermon import plots

        if not plots.matplotlib_available():
            self.skipTest("matplotlib not installed")
        record_demo(self.run, finalize=False)
        report.generate(self.run, "auto")
        html = (self.run / report.REPORT_HTML).read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("![Power](power.png)", (self.run / report.REPORT_MD).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

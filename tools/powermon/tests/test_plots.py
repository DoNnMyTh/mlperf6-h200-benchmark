from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from helpers import FakeClock

from powermon import plots
from powermon.plots import Chart, Series, build_charts, render_all, render_svg
from powermon.recorder import Recorder, RunConfig
from powermon.sources import DemoSource
from powermon.stats import ColumnMeta, summarize


def record_demo(run_dir: Path, seconds: int = 120) -> None:
    cfg = RunConfig(run_dir=str(run_dir), duration_s=seconds, fsync_every=0)
    src = DemoSource()
    Recorder(cfg, [src], [src.probe()], clock=FakeClock(), log=lambda m: None).run()


class ChartPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run = Path(self._tmp.name)
        record_demo(self.run)
        self.summary = summarize(self.run)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_build_charts_keys_and_series(self) -> None:
        charts = build_charts(self.summary)
        keys = [c.key for c in charts]
        self.assertEqual(keys, ["power", "gpu_power", "gpu_temps", "temps", "util"])
        power = charts[0]
        names = [s.name for s in power.series]
        self.assertIn("cpu_pkg0_w", names)
        self.assertIn("system_w", names)
        self.assertIn("total_gpu_w", names)
        self.assertNotIn("gpu0_power_w", names)
        self.assertEqual([s.name for s in charts[1].series], ["gpu0_power_w", "gpu1_power_w"])
        self.assertEqual(charts[4].y_label, "%")

    def test_all_nan_series_dropped(self) -> None:
        self.summary.data.values["gpu1_util_pct"] = [float("nan")] * self.summary.data.n
        charts = {c.key: c for c in build_charts(self.summary)}
        self.assertEqual([s.name for s in charts["util"].series], ["gpu0_util_pct"])

    def test_series_cap(self) -> None:
        n = self.summary.data.n
        for i in range(plots.MAX_SERIES + 5):
            name = f"extra{i}_c"
            self.summary.data.columns.append(name)
            self.summary.data.values[name] = [20.0 + i] * n
            self.summary.meta[name] = ColumnMeta(name, "C", "temp")
        charts = {c.key: c for c in build_charts(self.summary)}
        self.assertEqual(len(charts["temps"].series), plots.MAX_SERIES)
        self.assertIn("showing", charts["temps"].note)


class SvgTest(unittest.TestCase):
    def test_render_svg_basic(self) -> None:
        chart = Chart("power", "Power", "W", [0.0, 1.0, 2.0, 3.0], [
            Series("a_w", "a", [1.0, 2.0, float("nan"), 4.0]),
            Series("b_w", "b & c", [10.0, 11.0, 12.0, 13.0]),
        ])
        svg = render_svg(chart)
        self.assertTrue(svg.startswith("<svg xmlns"))
        self.assertTrue(svg.rstrip().endswith("</svg>"))
        self.assertIn("Power", svg)
        self.assertIn("b &amp; c", svg)
        # NaN breaks the path: series a has two M segments
        paths = [line for line in svg.splitlines() if line.startswith("<path")]
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0].count("M"), 2)
        self.assertEqual(paths[1].count("M"), 1)
        self.assertIn("time (s)", svg)

    def test_time_axis_switches_to_minutes_and_hours(self) -> None:
        elapsed = [float(i) for i in range(0, 600, 10)]
        chart = Chart("t", "T", "W", elapsed, [Series("x", "x", [1.0] * len(elapsed))])
        self.assertIn("time (min)", render_svg(chart))
        elapsed = [float(i) for i in range(0, 4 * 3600, 60)]
        chart = Chart("t", "T", "W", elapsed, [Series("x", "x", [1.0] * len(elapsed))])
        self.assertIn("time (h)", render_svg(chart))

    def test_downsample_large_series(self) -> None:
        n = 50000
        chart = Chart("t", "T", "W", [float(i) for i in range(n)], [Series("x", "x", [float(i % 7) for i in range(n)])])
        svg = render_svg(chart)
        path = next(line for line in svg.splitlines() if line.startswith("<path"))
        self.assertLessEqual(path.count("L"), plots.MAX_POINTS)

    def test_constant_series_does_not_divide_by_zero(self) -> None:
        chart = Chart("t", "T", "C", [0.0, 1.0], [Series("x", "x", [5.0, 5.0])])
        self.assertIn("<path", render_svg(chart))

    def test_nice_ticks(self) -> None:
        ticks = plots._nice_ticks(0, 100)
        self.assertEqual(ticks[0], 0)
        self.assertGreaterEqual(ticks[-1], 100)
        self.assertLessEqual(len(ticks), 12)


class RenderAllTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run = Path(self._tmp.name)
        record_demo(self.run, 30)
        self.summary = summarize(self.run)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_svg_mode_writes_svg_files(self) -> None:
        outputs = render_all(self.summary, self.run, mode="svg")
        self.assertEqual(len(outputs), 5)
        for out in outputs:
            self.assertIsNone(out.png)
            self.assertTrue((self.run / f"{out.key}.svg").exists())
            self.assertIn("<svg", out.svg)

    @unittest.skipUnless(plots.matplotlib_available(), "matplotlib not installed")
    def test_png_mode_writes_png_files(self) -> None:
        outputs = render_all(self.summary, self.run, mode="auto")
        for out in outputs:
            self.assertIsNotNone(out.png)
            self.assertTrue(out.png.exists())
            self.assertGreater(out.png.stat().st_size, 1000)
            self.assertEqual(out.png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    @unittest.skipIf(plots.matplotlib_available(), "matplotlib installed; fallback path covered elsewhere")
    def test_auto_mode_falls_back_to_svg(self) -> None:
        outputs = render_all(self.summary, self.run, mode="auto")
        self.assertTrue(all(o.png is None and o.svg for o in outputs))

    def test_png_mode_without_matplotlib_falls_back(self) -> None:
        original = plots.render_png
        plots.render_png = lambda chart, path: False
        try:
            outputs = render_all(self.summary, self.run, mode="png")
        finally:
            plots.render_png = original
        self.assertTrue(all(o.png is None and o.svg for o in outputs))


if __name__ == "__main__":
    unittest.main()

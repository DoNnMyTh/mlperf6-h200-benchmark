from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from helpers import write

from powermon.stats import (
    NoData,
    column_stats,
    energy_wh,
    infer_meta,
    load_csv,
    percentile,
    summarize,
)

HEADER = "timestamp,elapsed_s,cpu_pkg0_w,cpu_pkg0_dram_w,cpu_pkg1_w,gpu0_power_w,gpu1_power_w,gpu0_temp_c\n"


def csv_text(n: int = 5) -> str:
    lines = [HEADER]
    for i in range(n):
        lines.append(f"2026-01-01T12:00:{i:02d}.000+00:00,{i}.000,100.0,10.0,50.0,{200 + i},{300 + i},{60 + i}\n")
    return "".join(lines)


class LoadCsvTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_basic_and_truncated_last_line(self) -> None:
        path = write(self.dir / "samples.csv", csv_text(5) + "2026-01-01T12:00:05.000+00:00,5.000,100.0,10")
        data = load_csv(path)
        self.assertEqual(data.n, 5)
        self.assertEqual(data.dropped_rows, 1)
        self.assertEqual(data.columns[0], "cpu_pkg0_w")
        self.assertEqual(data.values["gpu1_power_w"], [300.0, 301.0, 302.0, 303.0, 304.0])
        self.assertEqual(data.elapsed, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_blank_cells_are_nan(self) -> None:
        text = HEADER + "t,0.000,,10,50,200,300,60\n" + "t,1.000,100,,50,200,300,60\n"
        data = load_csv(write(self.dir / "s.csv", text))
        self.assertTrue(math.isnan(data.values["cpu_pkg0_w"][0]))
        self.assertTrue(math.isnan(data.values["cpu_pkg0_dram_w"][1]))

    def test_nodata_cases(self) -> None:
        with self.assertRaises(NoData):
            load_csv(self.dir / "missing.csv")
        with self.assertRaises(NoData):
            load_csv(write(self.dir / "empty.csv", ""))
        with self.assertRaises(NoData):
            load_csv(write(self.dir / "one.csv", csv_text(1)))
        with self.assertRaises(NoData):
            load_csv(write(self.dir / "bad.csv", "a,b,c\n1,2,3\n4,5,6\n"))


class MathTest(unittest.TestCase):
    def test_percentile(self) -> None:
        self.assertTrue(math.isnan(percentile([], 95)))
        self.assertEqual(percentile([5.0], 95), 5.0)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 50), 3.0)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 95), 4.8)
        self.assertAlmostEqual(percentile([1, float("nan"), 3], 100), 3.0)

    def test_energy_wh(self) -> None:
        # 100 W for 3600 s = 100 Wh
        elapsed = [0.0, 1800.0, 3600.0]
        self.assertAlmostEqual(energy_wh(elapsed, [100.0, 100.0, 100.0]), 100.0)
        # ramp 0->100 over 3600 s = 50 Wh
        self.assertAlmostEqual(energy_wh([0.0, 3600.0], [0.0, 100.0]), 50.0)
        # NaN pair skipped
        self.assertAlmostEqual(energy_wh([0.0, 1800.0, 3600.0], [100.0, float("nan"), 100.0]), 0.0)
        self.assertAlmostEqual(energy_wh([0.0, 1800.0, 3600.0], [100.0, 100.0, float("nan")]), 50.0)

    def test_infer_meta(self) -> None:
        self.assertEqual(infer_meta("x_w").kind, "power")
        self.assertEqual(infer_meta("x_c").kind, "temp")
        self.assertEqual(infer_meta("x_pct").kind, "util")
        self.assertEqual(infer_meta("x_mib").kind, "mem")
        self.assertEqual(infer_meta("odd").kind, "other")

    def test_column_stats(self) -> None:
        meta = infer_meta("p_w")
        s = column_stats("p_w", meta, [0.0, 1.0, 2.0], [10.0, float("nan"), 30.0])
        self.assertEqual(s.n, 2)
        self.assertEqual(s.min, 10.0)
        self.assertEqual(s.max, 30.0)
        self.assertEqual(s.mean, 20.0)
        self.assertEqual(s.last, 30.0)
        self.assertAlmostEqual(s.energy_wh, 0.0)  # both pairs contain NaN
        empty = column_stats("p_w", meta, [0.0, 1.0], [float("nan"), float("nan")])
        self.assertEqual(empty.n, 0)
        self.assertTrue(math.isnan(empty.mean))


class SummarizeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run = Path(self._tmp.name)
        write(self.run / "samples.csv", csv_text(11))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_summary_without_sidecars(self) -> None:
        s = summarize(self.run)
        self.assertEqual(s.duration_s, 10.0)
        self.assertEqual(s.interval_s, 1.0)
        self.assertEqual(s.expected_samples, 11)
        self.assertEqual(s.gaps, 0)
        self.assertEqual(s.state, "unknown")
        self.assertFalse(s.complete)
        names = [st.name for st in s.stats]
        self.assertEqual(names, ["cpu_pkg0_w", "cpu_pkg0_dram_w", "cpu_pkg1_w", "gpu0_power_w", "gpu1_power_w", "gpu0_temp_c"])
        self.assertEqual(sorted(s.derived), ["total_cpu_w", "total_gpu_w"])
        # top-level packages only: 100 + 50, dram excluded
        self.assertEqual(s.derived["total_cpu_w"][0], 150.0)
        self.assertEqual(s.derived["total_gpu_w"][3], 203.0 + 303.0)
        self.assertEqual(len(s.by_kind("temp")), 1)
        gpu0 = next(st for st in s.stats if st.name == "gpu0_power_w")
        self.assertAlmostEqual(gpu0.energy_wh, (200 + 210) / 2 * 10 / 3600)

    def test_summary_uses_sidecars(self) -> None:
        write(self.run / "sensors.json", json.dumps({
            "interval_s": 1.0,
            "label": "bench",
            "columns": [{"name": "gpu0_power_w", "unit": "W", "kind": "power", "source": "nvidia", "label": "H200"}],
            "probe": [{"source": "nvidia", "title": "NVIDIA", "status": "ok", "columns": ["gpu0_power_w"], "note": ""}],
        }))
        write(self.run / "status.json", json.dumps({"state": "done", "gaps": 3, "final": True}))
        s = summarize(self.run)
        self.assertEqual(s.gaps, 3)
        self.assertTrue(s.complete)
        self.assertEqual(s.meta["gpu0_power_w"].label, "H200")
        self.assertEqual(s.meta["cpu_pkg0_w"].kind, "power")  # inferred fallback

    def test_missing_rows_counted_as_gaps_without_status(self) -> None:
        text = HEADER + "".join(f"t,{e}.000,1,1,1,1,1,1\n" for e in (0, 1, 2, 5, 6))
        write(self.run / "samples.csv", text)
        write(self.run / "sensors.json", json.dumps({"interval_s": 1.0, "columns": [], "probe": []}))
        s = summarize(self.run)
        self.assertEqual(s.data.n, 5)
        self.assertEqual(s.expected_samples, 7)
        self.assertEqual(s.gaps, 2)


if __name__ == "__main__":
    unittest.main()

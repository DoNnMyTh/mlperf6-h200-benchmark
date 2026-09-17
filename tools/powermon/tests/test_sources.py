from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path

from helpers import (
    IPMI_OUTPUT,
    NVSMI_TWO_GPUS,
    CannedRunner,
    FakeSysfs,
    timeout_error,
    which_all,
    which_none,
    write,
)

from powermon import sources
from powermon.sources import (
    BatterySource,
    DemoSource,
    HwmonSource,
    IpmiSource,
    NvidiaSmiSource,
    RaplSource,
    ThermalZoneSource,
    active_sources,
    all_columns,
    parse_ipmi_power,
    parse_nvidia_smi,
    probe_all,
    sanitize,
)


class TempRootCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.fs = FakeSysfs(self.root)
        self.logs = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def log(self, msg: str) -> None:
        self.logs.append(msg)


class SanitizeTest(unittest.TestCase):
    def test_sanitize(self) -> None:
        self.assertEqual(sanitize("Package id 0"), "package_id_0")
        self.assertEqual(sanitize("  Tctl--  "), "tctl")
        self.assertEqual(sanitize("NVIDIA H200 NVL"), "nvidia_h200_nvl")


class EmptyRootTest(TempRootCase):
    def test_probe_all_empty_root_never_crashes(self) -> None:
        probed = probe_all(root=self.root, which=which_none, log=self.log)
        ids = [src.id for src, _ in probed]
        self.assertEqual(ids, ["rapl", "hwmon", "thermal", "battery", "nvidia", "ipmi"])
        for _, res in probed:
            self.assertEqual(res.status, "absent", res)
        self.assertEqual(active_sources(probed), [])
        self.assertEqual(all_columns([]), [])

    def test_disabled_sources(self) -> None:
        probed = probe_all(root=self.root, enabled=["rapl"], which=which_none)
        statuses = {res.source: res.status for _, res in probed}
        self.assertEqual(statuses["rapl"], "absent")
        self.assertEqual(statuses["hwmon"], "disabled")
        self.assertEqual(statuses["nvidia"], "disabled")


class RaplTest(TempRootCase):
    def test_probe_columns_and_wraparound(self) -> None:
        pkg = self.fs.rapl(0, energy=1_000_000, max_range=5_000_000)
        self.fs.rapl(0, energy=500_000, sub=0, name="dram", max_range=5_000_000)
        self.fs.rapl(1, energy=10, max_range=5_000_000)
        src = RaplSource(self.root, self.log)
        res = src.probe()
        self.assertEqual(res.status, "ok")
        self.assertEqual([c.name for c in src.columns], ["cpu_pkg0_w", "cpu_pkg0_dram_w", "cpu_pkg1_w"])
        self.assertTrue(all(c.kind == "power" and c.unit == "W" for c in src.columns))
        # First tick establishes a baseline -> NaN.
        row = src.read(100.0)
        self.assertTrue(all(math.isnan(v) for v in row.values()))
        # +2 J over 2 s = 1 W
        write(pkg / "energy_uj", str(1_000_000 + 2_000_000))
        row = src.read(102.0)
        self.assertAlmostEqual(row["cpu_pkg0_w"], 1.0)
        self.assertAlmostEqual(row["cpu_pkg1_w"], 0.0)
        # Wraparound: counter goes from 3_000_000 down to 999_999 with range 5_000_000
        # -> delta = 999_999 - 3_000_000 + 5_000_001 = 3_000_000 uJ over 1 s = 3 W
        write(pkg / "energy_uj", "999999")
        row = src.read(103.0)
        self.assertAlmostEqual(row["cpu_pkg0_w"], 3.0)

    def test_mmio_duplicates_ignored_when_msr_present(self) -> None:
        self.fs.rapl(0, energy=1)
        self.fs.rapl(0, energy=1, mmio=True)
        src = RaplSource(self.root)
        src.probe()
        self.assertEqual([c.name for c in src.columns], ["cpu_pkg0_w"])
        self.assertIn("intel-rapl:0", src.columns[0].path)
        self.assertNotIn("mmio", src.columns[0].path)

    def test_mmio_used_when_only_option(self) -> None:
        self.fs.rapl(0, energy=1, mmio=True)
        src = RaplSource(self.root)
        self.assertEqual(src.probe().status, "ok")
        self.assertIn("mmio", src.columns[0].path)

    def test_psys_domain_named_by_name_file(self) -> None:
        self.fs.rapl(0, energy=1)
        self.fs.rapl(1, energy=1, name="psys")
        src = RaplSource(self.root)
        src.probe()
        self.assertEqual([c.name for c in src.columns], ["cpu_pkg0_w", "cpu_psys_w"])

    @unittest.skipIf(os.name != "posix" or os.geteuid() == 0, "needs non-root posix for chmod to deny")
    def test_denied_when_unreadable(self) -> None:
        pkg = self.fs.rapl(0, energy=1)
        os.chmod(pkg / "energy_uj", 0o000)
        try:
            src = RaplSource(self.root)
            res = src.probe()
            self.assertEqual(res.status, "denied")
            self.assertIn("sudo", res.note)
            self.assertEqual(src.columns, [])
        finally:
            os.chmod(pkg / "energy_uj", 0o644)

    def test_read_failure_mid_run_is_nan_and_counted(self) -> None:
        pkg = self.fs.rapl(0, energy=1)
        src = RaplSource(self.root, self.log)
        src.probe()
        src.read(1.0)
        write(pkg / "energy_uj", "garbage")
        row = src.read(2.0)
        self.assertTrue(math.isnan(row["cpu_pkg0_w"]))
        self.assertEqual(src.errors, 1)
        self.assertEqual(len(self.logs), 1)


class HwmonTest(TempRootCase):
    def test_naming_labels_dedup_and_scaling(self) -> None:
        self.fs.hwmon(0, "coretemp", temps={1: (45000, "Package id 0"), 2: (41000, "Core 0")})
        self.fs.hwmon(1, "nvme", temps={1: (38000, "Composite")})
        self.fs.hwmon(2, "nvme", temps={1: (39000, "Composite")})
        self.fs.hwmon(3, "amdgpu", temps={1: (50000, "edge"), 2: (55000, "junction")},
                      powers={1: (123_456_000, "average")})
        self.fs.hwmon(4, "acpitz", temps={1: (27800, None)})
        src = HwmonSource(self.root, self.log)
        res = src.probe()
        self.assertEqual(res.status, "ok")
        names = [c.name for c in src.columns]
        self.assertEqual(
            names,
            [
                "coretemp_package_id_0_c",
                "coretemp_core_0_c",
                "nvme_composite_c",
                "nvme_2_composite_c",
                "amdgpu_edge_c",
                "amdgpu_junction_c",
                "amdgpu_power1_w",
                "acpitz_temp1_c",
            ],
        )
        self.assertEqual(src.temp_count, 7)
        row = src.read(0.0)
        self.assertAlmostEqual(row["coretemp_package_id_0_c"], 45.0)
        self.assertAlmostEqual(row["amdgpu_power1_w"], 123.456)
        self.assertAlmostEqual(row["acpitz_temp1_c"], 27.8)
        kinds = {c.name: c.kind for c in src.columns}
        self.assertEqual(kinds["amdgpu_power1_w"], "power")
        self.assertEqual(kinds["amdgpu_edge_c"], "temp")

    def test_power_input_preferred_over_average(self) -> None:
        d = self.fs.hwmon(0, "amdgpu", powers={1: (10_000_000, "average")})
        write(d / "power1_input", "20000000")
        src = HwmonSource(self.root)
        src.probe()
        self.assertEqual([c.name for c in src.columns], ["amdgpu_power1_w"])
        self.assertAlmostEqual(src.read(0.0)["amdgpu_power1_w"], 20.0)

    def test_unreadable_inputs_skipped_at_probe(self) -> None:
        d = self.fs.hwmon(0, "iwlwifi", temps={1: (0, None)})
        write(d / "temp1_input", "")  # ENODATA-like: empty/unparseable
        self.fs.hwmon(1, "k10temp", temps={1: (60000, "Tctl")})
        src = HwmonSource(self.root)
        src.probe()
        self.assertEqual([c.name for c in src.columns], ["k10temp_tctl_c"])

    def test_column_cap(self) -> None:
        temps = {n: (30000 + n, None) for n in range(1, sources.MAX_HWMON_COLUMNS + 20)}
        self.fs.hwmon(0, "big", temps=temps)
        src = HwmonSource(self.root)
        res = src.probe()
        self.assertEqual(len(src.columns), sources.MAX_HWMON_COLUMNS)
        self.assertIn("capped", res.note)

    def test_read_error_mid_run(self) -> None:
        d = self.fs.hwmon(0, "k10temp", temps={1: (60000, "Tctl")})
        src = HwmonSource(self.root, self.log)
        src.probe()
        (d / "temp1_input").unlink()
        row = src.read(0.0)
        self.assertTrue(math.isnan(row["k10temp_tctl_c"]))
        self.assertEqual(src.errors, 1)


class ThermalTest(TempRootCase):
    def test_zones(self) -> None:
        self.fs.thermal(0, "x86_pkg_temp", 52000)
        self.fs.thermal(1, "acpitz", 27000)
        src = ThermalZoneSource(self.root)
        self.assertEqual(src.probe().status, "ok")
        self.assertEqual([c.name for c in src.columns], ["tz0_x86_pkg_temp_c", "tz1_acpitz_c"])
        self.assertAlmostEqual(src.read(0.0)["tz0_x86_pkg_temp_c"], 52.0)

    def test_thermal_disabled_when_hwmon_has_temps(self) -> None:
        self.fs.hwmon(0, "coretemp", temps={1: (45000, "Package id 0")})
        self.fs.thermal(0, "x86_pkg_temp", 45000)
        probed = {res.source: res for _, res in probe_all(root=self.root, which=which_none)}
        self.assertEqual(probed["hwmon"].status, "ok")
        self.assertEqual(probed["thermal"].status, "disabled")

    def test_thermal_used_when_hwmon_has_no_temps(self) -> None:
        self.fs.hwmon(0, "amdgpu", powers={1: (1_000_000, "average")})
        self.fs.thermal(0, "x86_pkg_temp", 45000)
        probed = {res.source: res for _, res in probe_all(root=self.root, which=which_none)}
        self.assertEqual(probed["hwmon"].status, "ok")
        self.assertEqual(probed["thermal"].status, "ok")


class BatteryTest(TempRootCase):
    def test_power_now(self) -> None:
        self.fs.battery("BAT0", power_now=15_500_000)
        src = BatterySource(self.root)
        self.assertEqual(src.probe().status, "ok")
        self.assertEqual([c.name for c in src.columns], ["bat0_w"])
        self.assertAlmostEqual(src.read(0.0)["bat0_w"], 15.5)

    def test_current_times_voltage(self) -> None:
        self.fs.battery("BAT1", current_now=2_000_000, voltage_now=12_000_000)
        src = BatterySource(self.root)
        self.assertEqual(src.probe().status, "ok")
        self.assertAlmostEqual(src.read(0.0)["bat1_w"], 24.0)

    def test_non_battery_supply_ignored(self) -> None:
        d = self.root / "sys" / "class" / "power_supply" / "AC"
        write(d / "type", "Mains")
        src = BatterySource(self.root)
        self.assertEqual(src.probe().status, "absent")


class NvidiaTest(unittest.TestCase):
    def test_parse(self) -> None:
        gpus = parse_nvidia_smi(NVSMI_TWO_GPUS)
        self.assertEqual(sorted(gpus), [0, 1])
        self.assertAlmostEqual(gpus[0]["power_w"], 412.3)
        self.assertAlmostEqual(gpus[0]["mem_temp_c"], 72.0)
        self.assertTrue(math.isnan(gpus[1]["power_w"]))
        self.assertTrue(math.isnan(gpus[1]["mem_temp_c"]))
        self.assertEqual(gpus[1]["name"], "NVIDIA H200 NVL")

    def test_absent_without_binary(self) -> None:
        src = NvidiaSmiSource(runner=CannedRunner(), which=which_none)
        self.assertEqual(src.probe().status, "absent")

    def test_probe_and_read(self) -> None:
        runner = CannedRunner({"nvidia-smi": [NVSMI_TWO_GPUS]})
        src = NvidiaSmiSource(interval=1.0, runner=runner, which=which_all)
        res = src.probe()
        self.assertEqual(res.status, "ok")
        self.assertIn("2 GPU(s)", res.note)
        self.assertEqual(len(src.columns), 10)
        self.assertEqual(src.columns[0].name, "gpu0_power_w")
        self.assertEqual(src.gpu_names, {0: "NVIDIA H200 NVL", 1: "NVIDIA H200 NVL"})
        row = src.read(0.0)
        self.assertAlmostEqual(row["gpu0_power_w"], 412.3)
        self.assertAlmostEqual(row["gpu1_util_pct"], 95.0)
        self.assertTrue(math.isnan(row["gpu1_power_w"]))
        self.assertIn("--query-gpu=index,name,power.draw", runner.calls[0][1])
        self.assertAlmostEqual(src.timeout, 0.8)

    def test_timeout_and_error_mid_run(self) -> None:
        logs = []
        runner = CannedRunner({"nvidia-smi": [NVSMI_TWO_GPUS, timeout_error(), RuntimeError("boom"), NVSMI_TWO_GPUS]})
        src = NvidiaSmiSource(runner=runner, which=which_all, log=logs.append)
        src.probe()
        row = src.read(0.0)
        self.assertTrue(all(math.isnan(v) for v in row.values()))
        row = src.read(1.0)
        self.assertTrue(all(math.isnan(v) for v in row.values()))
        self.assertEqual(src.errors, 2)
        self.assertEqual(len(logs), 2)
        row = src.read(2.0)
        self.assertAlmostEqual(row["gpu0_temp_c"], 61.0)

    def test_probe_error(self) -> None:
        runner = CannedRunner({"nvidia-smi": [RuntimeError("NVML: Driver/library version mismatch")]})
        src = NvidiaSmiSource(runner=runner, which=which_all)
        res = src.probe()
        self.assertEqual(res.status, "error")
        self.assertIn("mismatch", res.note)

    def test_failure_log_throttle(self) -> None:
        logs = []
        runner = CannedRunner({"nvidia-smi": [NVSMI_TWO_GPUS, RuntimeError("x")]})
        src = NvidiaSmiSource(runner=runner, which=which_all, log=logs.append)
        src.probe()
        for i in range(130):
            src.read(float(i))
        self.assertEqual(src.errors, 130)
        self.assertEqual(len(logs), 3 + 2)  # first 3, then #60 and #120


class IpmiTest(unittest.TestCase):
    def test_parse(self) -> None:
        self.assertAlmostEqual(parse_ipmi_power(IPMI_OUTPUT), 1234.0)
        self.assertTrue(math.isnan(parse_ipmi_power("nothing here")))

    def test_probe_and_every_n(self) -> None:
        runner = CannedRunner({"ipmitool": [IPMI_OUTPUT]})
        src = IpmiSource(interval=1.0, every=3, runner=runner, which=which_all)
        res = src.probe()
        self.assertEqual(res.status, "ok")
        self.assertEqual([c.name for c in src.columns], ["system_w"])
        vals = [src.read(float(i))["system_w"] for i in range(6)]
        self.assertAlmostEqual(vals[0], 1234.0)
        self.assertTrue(math.isnan(vals[1]))
        self.assertTrue(math.isnan(vals[2]))
        self.assertAlmostEqual(vals[3], 1234.0)
        self.assertEqual(len(runner.calls), 1 + 2)

    def test_absent_and_denied(self) -> None:
        self.assertEqual(IpmiSource(runner=CannedRunner(), which=which_none).probe().status, "absent")
        runner = CannedRunner({"ipmitool": [RuntimeError("Could not open device at /dev/ipmi0: Permission denied")]})
        res = IpmiSource(runner=runner, which=which_all).probe()
        self.assertEqual(res.status, "denied")
        self.assertIn("sudo", res.note)


class DemoTest(unittest.TestCase):
    def test_demo_rows_cover_all_columns(self) -> None:
        src = DemoSource()
        self.assertEqual(src.probe().status, "ok")
        row = src.read(10.0)
        self.assertEqual(set(row), {c.name for c in src.columns})
        self.assertTrue(all(not math.isnan(v) for v in row.values()))
        self.assertTrue(0 <= row["gpu0_util_pct"] <= 100)

    def test_probe_all_demo(self) -> None:
        probed = probe_all(demo=True)
        self.assertEqual([src.id for src, _ in probed], ["demo"])
        self.assertEqual(len(all_columns(active_sources(probed))), 9)


if __name__ == "__main__":
    unittest.main()

"""Sensor sources.

Every source is optional and auto-probed. A source never raises from ``read``:
failures become NaN cells plus an incremented error counter so one flaky sensor
cannot stop the recording.

Sysfs-backed sources take a ``root`` so tests can point them at a fake tree.
Subprocess-backed sources take a ``runner`` / ``which`` so tests can inject
canned output without the real binaries.
"""

from __future__ import annotations

import math
import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

NAN = float("nan")

Runner = Callable[[Sequence[str], float], str]
Which = Callable[[str], Optional[str]]
Logger = Callable[[str], None]

ALL_SOURCES = ("rapl", "hwmon", "thermal", "battery", "nvidia", "ipmi")
MAX_HWMON_COLUMNS = 128


def default_runner(args: Sequence[str], timeout: float) -> str:
    env = dict(os.environ, LC_ALL="C")
    proc = subprocess.run(
        list(args), capture_output=True, text=True, timeout=timeout, env=env
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")[:200]
        raise RuntimeError(f"{args[0]} exit {proc.returncode}: {detail}")
    return proc.stdout


def _noop_log(_msg: str) -> None:
    pass


def sanitize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


_PER_CORE_RE = re.compile(r"^core[\s_]*\d+$", re.I)


def is_per_core_label(label: str) -> bool:
    """True for hwmon labels like 'Core 12' (coretemp per-core inputs)."""
    return bool(_PER_CORE_RE.match(label.strip()))


def is_per_core_column(name: str) -> bool:
    """True for column names like 'coretemp_core_12_c' / 'coretemp_2_core_3_c'."""
    return bool(re.search(r"_core_\d+_c$", name))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _read_int(path: Path) -> int:
    return int(_read_text(path))


def _natural_key(path: Path) -> tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name))


@dataclass
class Column:
    name: str
    unit: str  # W, C, %, MiB
    kind: str  # power | temp | util | mem
    source: str
    path: str = ""
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "unit": self.unit,
            "kind": self.kind,
            "source": self.source,
            "path": self.path,
            "label": self.label,
        }


@dataclass
class ProbeResult:
    source: str
    title: str
    status: str  # ok | absent | denied | error | disabled
    columns: List[Column] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "status": self.status,
            "columns": [c.name for c in self.columns],
            "note": self.note,
        }


class Source:
    id = "base"
    title = "base"

    def __init__(self, log: Logger = _noop_log) -> None:
        self.log = log
        self.errors = 0
        self._columns: List[Column] = []
        self._fail_count = 0

    @property
    def columns(self) -> List[Column]:
        return self._columns

    def probe(self) -> ProbeResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def read(self, now: float) -> Dict[str, float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _nan_row(self) -> Dict[str, float]:
        return {c.name: NAN for c in self._columns}

    def _log_failure(self, what: str) -> None:
        # First three failures verbosely, then every 60th, so a dead sensor
        # does not flood worker.log at 1 Hz.
        self.errors += 1
        self._fail_count += 1
        if self._fail_count <= 3 or self._fail_count % 60 == 0:
            self.log(f"[{self.id}] read failed (#{self._fail_count}): {what}")

    def _result(self, status: str, note: str = "") -> ProbeResult:
        return ProbeResult(self.id, self.title, status, list(self._columns), note)


# --------------------------------------------------------------------------- RAPL


_RAPL_RE = re.compile(r"^intel-rapl(?P<mmio>-mmio)?:(?P<pkg>\d+)(?::(?P<sub>\d+))?$")


class RaplSource(Source):
    """CPU package energy counters via /sys/class/powercap (Intel and AMD Zen)."""

    id = "rapl"
    title = "CPU power (RAPL)"

    def __init__(self, root: Path = Path("/"), log: Logger = _noop_log) -> None:
        super().__init__(log)
        self.root = Path(root)
        self._paths: Dict[str, Path] = {}
        self._max_range: Dict[str, int] = {}
        self._prev: Dict[str, int] = {}
        self._prev_t: Optional[float] = None

    def probe(self) -> ProbeResult:
        base = self.root / "sys" / "class" / "powercap"
        if not base.is_dir():
            return self._result("absent", "no /sys/class/powercap")
        entries = []
        for entry in sorted(base.iterdir(), key=_natural_key):
            m = _RAPL_RE.match(entry.name)
            if m:
                entries.append((entry, m))
        if not entries:
            return self._result("absent", "no intel-rapl domains")
        # Prefer MSR-backed domains; MMIO domains duplicate the same package.
        non_mmio = [e for e in entries if not e[1].group("mmio")]
        if non_mmio:
            entries = non_mmio
        denied = False
        pkg_prefix: Dict[str, str] = {}
        for entry, m in entries:
            energy = entry / "energy_uj"
            if not energy.exists():
                continue
            try:
                name = _read_text(entry / "name")
            except OSError:
                name = entry.name
            try:
                cur = _read_int(energy)
            except PermissionError:
                denied = True
                continue
            except (OSError, ValueError):
                continue
            pkg = m.group("pkg")
            if m.group("sub") is None:
                pm = re.match(r"package-(\d+)", name)
                prefix = f"cpu_pkg{pm.group(1)}" if pm else f"cpu_{sanitize(name)}"
                pkg_prefix[pkg] = prefix
                col_name = f"{prefix}_w"
            else:
                prefix = pkg_prefix.get(pkg, f"cpu_pkg{pkg}")
                col_name = f"{prefix}_{sanitize(name)}_w"
            try:
                max_range = _read_int(entry / "max_energy_range_uj")
            except (OSError, ValueError):
                max_range = 2**32
            self._columns.append(Column(col_name, "W", "power", self.id, str(energy), name))
            self._paths[col_name] = energy
            self._max_range[col_name] = max_range
            self._prev[col_name] = cur
        if self._columns:
            note = "" if not denied else "some domains unreadable"
            return self._result("ok", note)
        if denied:
            return self._result(
                "denied", "energy_uj not readable: run with sudo for CPU power"
            )
        return self._result("absent", "no readable energy_uj")

    def read(self, now: float) -> Dict[str, float]:
        row = self._nan_row()
        if self._prev_t is None:
            # Counters were primed at probe time, but we do not know when
            # exactly; the first tick therefore yields NaN and a fresh baseline.
            for name, path in self._paths.items():
                try:
                    self._prev[name] = _read_int(path)
                except (OSError, ValueError) as exc:
                    self._log_failure(f"{path}: {exc}")
            self._prev_t = now
            return row
        dt = now - self._prev_t
        self._prev_t = now
        if dt <= 0:
            return row
        for name, path in self._paths.items():
            try:
                cur = _read_int(path)
            except (OSError, ValueError) as exc:
                self._log_failure(f"{path}: {exc}")
                continue
            prev = self._prev.get(name)
            self._prev[name] = cur
            if prev is None:
                continue
            delta = cur - prev
            if delta < 0:
                delta += self._max_range[name] + 1
            row[name] = delta / 1e6 / dt
        return row


# -------------------------------------------------------------------------- hwmon


class HwmonSource(Source):
    """Generic /sys/class/hwmon temperatures and power (coretemp, k10temp, nvme, amdgpu...)."""

    id = "hwmon"
    title = "hwmon temps/power"

    def __init__(self, root: Path = Path("/"), log: Logger = _noop_log, skip_per_core: bool = False) -> None:
        super().__init__(log)
        self.root = Path(root)
        self.skip_per_core = skip_per_core
        self._paths: Dict[str, Path] = {}
        self._scale: Dict[str, float] = {}
        self.skipped_per_core = 0

    def probe(self) -> ProbeResult:
        base = self.root / "sys" / "class" / "hwmon"
        if not base.is_dir():
            return self._result("absent", "no /sys/class/hwmon")
        chips = sorted((p for p in base.iterdir() if p.name.startswith("hwmon")), key=_natural_key)
        if not chips:
            return self._result("absent", "no hwmon chips")
        seen_chip: Dict[str, int] = {}
        used_cols: set = set()
        denied = 0
        capped = False
        for chip_dir in chips:
            try:
                chip = sanitize(_read_text(chip_dir / "name")) or chip_dir.name
            except OSError:
                chip = chip_dir.name
            seen_chip[chip] = seen_chip.get(chip, 0) + 1
            prefix = chip if seen_chip[chip] == 1 else f"{chip}_{seen_chip[chip]}"
            for kind, unit, scale, pattern, suffix in (
                ("temp", "C", 1000.0, r"^temp(\d+)_input$", "c"),
                ("power", "W", 1e6, r"^power(\d+)_(input|average)$", "w"),
            ):
                inputs: Dict[str, Path] = {}
                for f in sorted(chip_dir.iterdir(), key=_natural_key):
                    m = re.match(pattern, f.name)
                    if not m:
                        continue
                    idx = m.group(1)
                    # Prefer powerN_input over powerN_average when both exist.
                    if idx in inputs and inputs[idx].name.endswith("_input"):
                        continue
                    inputs[idx] = f
                for idx, f in inputs.items():
                    if len(self._columns) >= MAX_HWMON_COLUMNS:
                        capped = True
                        break
                    label_path = chip_dir / f"{kind}{idx}_label"
                    label = ""
                    if label_path.exists():
                        try:
                            label = _read_text(label_path)
                        except OSError:
                            label = ""
                    if kind == "temp" and self.skip_per_core and is_per_core_label(label):
                        self.skipped_per_core += 1
                        continue
                    part = sanitize(label) if label else f"{kind}{idx}"
                    name = f"{prefix}_{part}_{suffix}"
                    if name in used_cols:
                        name = f"{prefix}_{kind}{idx}_{suffix}"
                    if name in used_cols:
                        continue
                    try:
                        _read_int(f)
                    except PermissionError:
                        denied += 1
                        continue
                    except (OSError, ValueError):
                        # e.g. iwlwifi temp when radio is off: ENODATA. Skip.
                        continue
                    used_cols.add(name)
                    self._columns.append(Column(name, unit, kind, self.id, str(f), label))
                    self._paths[name] = f
                    self._scale[name] = scale
        if self._columns:
            notes = []
            if capped:
                notes.append(f"capped at {MAX_HWMON_COLUMNS} columns")
            if self.skipped_per_core:
                notes.append(f"{self.skipped_per_core} per-core temps skipped")
            return self._result("ok", "; ".join(notes))
        if denied:
            return self._result("denied", "hwmon inputs not readable")
        return self._result("absent", "no readable hwmon inputs")

    def read(self, now: float) -> Dict[str, float]:
        row = self._nan_row()
        for name, path in self._paths.items():
            try:
                row[name] = _read_int(path) / self._scale[name]
            except (OSError, ValueError) as exc:
                self._log_failure(f"{path}: {exc}")
        return row

    @property
    def temp_count(self) -> int:
        return sum(1 for c in self._columns if c.kind == "temp")


# ------------------------------------------------------------------ thermal zones


class ThermalZoneSource(Source):
    """/sys/class/thermal fallback, used only when hwmon offers no temperatures."""

    id = "thermal"
    title = "thermal zones"

    def __init__(self, root: Path = Path("/"), log: Logger = _noop_log) -> None:
        super().__init__(log)
        self.root = Path(root)
        self._paths: Dict[str, Path] = {}

    def probe(self) -> ProbeResult:
        base = self.root / "sys" / "class" / "thermal"
        if not base.is_dir():
            return self._result("absent", "no /sys/class/thermal")
        zones = sorted((p for p in base.iterdir() if p.name.startswith("thermal_zone")), key=_natural_key)
        used: set = set()
        for zone in zones:
            temp = zone / "temp"
            try:
                ztype = sanitize(_read_text(zone / "type")) or "zone"
            except OSError:
                ztype = "zone"
            idx = zone.name[len("thermal_zone"):]
            name = f"tz{idx}_{ztype}_c"
            if name in used:
                continue
            try:
                _read_int(temp)
            except (OSError, ValueError):
                continue
            used.add(name)
            self._columns.append(Column(name, "C", "temp", self.id, str(temp), ztype))
            self._paths[name] = temp
        if self._columns:
            return self._result("ok")
        return self._result("absent", "no readable thermal zones")

    def read(self, now: float) -> Dict[str, float]:
        row = self._nan_row()
        for name, path in self._paths.items():
            try:
                row[name] = _read_int(path) / 1000.0
            except (OSError, ValueError) as exc:
                self._log_failure(f"{path}: {exc}")
        return row


# ------------------------------------------------------------------------ battery


class BatterySource(Source):
    """Laptop battery discharge/charge power from /sys/class/power_supply."""

    id = "battery"
    title = "battery power"

    def __init__(self, root: Path = Path("/"), log: Logger = _noop_log) -> None:
        super().__init__(log)
        self.root = Path(root)
        self._readers: Dict[str, Callable[[], float]] = {}

    def probe(self) -> ProbeResult:
        base = self.root / "sys" / "class" / "power_supply"
        if not base.is_dir():
            return self._result("absent", "no /sys/class/power_supply")
        for supply in sorted(base.iterdir(), key=_natural_key):
            try:
                if _read_text(supply / "type").lower() != "battery":
                    continue
            except OSError:
                continue
            name = f"{sanitize(supply.name)}_w"
            power_now = supply / "power_now"
            current_now = supply / "current_now"
            voltage_now = supply / "voltage_now"
            reader: Optional[Callable[[], float]] = None
            if power_now.exists():
                reader = lambda p=power_now: abs(_read_int(p)) / 1e6
                path = power_now
            elif current_now.exists() and voltage_now.exists():
                reader = lambda c=current_now, v=voltage_now: abs(_read_int(c) * _read_int(v)) / 1e12
                path = current_now
            else:
                continue
            try:
                reader()
            except (OSError, ValueError):
                continue
            self._columns.append(Column(name, "W", "power", self.id, str(path), supply.name))
            self._readers[name] = reader
        if self._columns:
            return self._result("ok")
        return self._result("absent", "no battery")

    def read(self, now: float) -> Dict[str, float]:
        row = self._nan_row()
        for name, reader in self._readers.items():
            try:
                row[name] = reader()
            except (OSError, ValueError) as exc:
                self._log_failure(f"{name}: {exc}")
        return row


# ------------------------------------------------------------------------- nvidia


_NVSMI_FIELDS = (
    "index",
    "name",
    "power.draw",
    "temperature.gpu",
    "temperature.memory",
    "utilization.gpu",
    "memory.used",
)


def _nv_float(token: str) -> float:
    token = token.strip()
    if not token or token.upper() in ("N/A", "[N/A]", "[NOT SUPPORTED]", "[INSUFFICIENT PERMISSIONS]"):
        return NAN
    try:
        return float(token)
    except ValueError:
        return NAN


def parse_nvidia_smi(text: str) -> Dict[int, Dict[str, float]]:
    """Parse ``--format=csv,noheader,nounits`` output into {index: {field: value}}."""
    out: Dict[int, Dict[str, float]] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_NVSMI_FIELDS):
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        out[idx] = {
            "name": parts[1],  # type: ignore[dict-item]
            "power_w": _nv_float(parts[2]),
            "temp_c": _nv_float(parts[3]),
            "mem_temp_c": _nv_float(parts[4]),
            "util_pct": _nv_float(parts[5]),
            "mem_used_mib": _nv_float(parts[6]),
        }
    return out


class NvidiaSmiSource(Source):
    """Per-GPU power/temperature/utilisation via one ``nvidia-smi`` call per tick."""

    id = "nvidia"
    title = "NVIDIA GPUs (nvidia-smi)"

    def __init__(
        self,
        interval: float = 1.0,
        runner: Runner = default_runner,
        which: Which = shutil.which,
        log: Logger = _noop_log,
    ) -> None:
        super().__init__(log)
        self.interval = interval
        self.runner = runner
        self.which = which
        self.timeout = max(0.5, 0.8 * interval)
        self._binary: Optional[str] = None
        self._gpu_names: Dict[int, str] = {}

    def _args(self) -> List[str]:
        return [
            self._binary or "nvidia-smi",
            f"--query-gpu={','.join(_NVSMI_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]

    def probe(self) -> ProbeResult:
        self._binary = self.which("nvidia-smi")
        if not self._binary:
            return self._result("absent", "nvidia-smi not found")
        try:
            gpus = parse_nvidia_smi(self.runner(self._args(), 10.0))
        except Exception as exc:  # noqa: BLE001 - any failure means unusable
            return self._result("error", f"nvidia-smi failed: {str(exc)[:200]}")
        if not gpus:
            return self._result("absent", "nvidia-smi reports no GPUs")
        for idx in sorted(gpus):
            gname = str(gpus[idx]["name"])
            self._gpu_names[idx] = gname
            src = f"nvidia-smi index {idx}"
            self._columns.extend(
                [
                    Column(f"gpu{idx}_power_w", "W", "power", self.id, src, gname),
                    Column(f"gpu{idx}_temp_c", "C", "temp", self.id, src, f"{gname} core"),
                    Column(f"gpu{idx}_mem_temp_c", "C", "temp", self.id, src, f"{gname} memory"),
                    Column(f"gpu{idx}_util_pct", "%", "util", self.id, src, f"{gname} util"),
                    Column(f"gpu{idx}_mem_used_mib", "MiB", "mem", self.id, src, f"{gname} mem used"),
                ]
            )
        names = sorted(set(self._gpu_names.values()))
        return self._result("ok", f"{len(gpus)} GPU(s): {', '.join(names)}")

    @property
    def gpu_names(self) -> Dict[int, str]:
        return dict(self._gpu_names)

    def read(self, now: float) -> Dict[str, float]:
        row = self._nan_row()
        try:
            gpus = parse_nvidia_smi(self.runner(self._args(), self.timeout))
        except subprocess.TimeoutExpired:
            self._log_failure(f"nvidia-smi timed out after {self.timeout:.1f}s")
            return row
        except Exception as exc:  # noqa: BLE001
            self._log_failure(f"nvidia-smi: {str(exc)[:200]}")
            return row
        for idx in self._gpu_names:
            g = gpus.get(idx)
            if not g:
                continue
            for field_name in ("power_w", "temp_c", "mem_temp_c", "util_pct", "mem_used_mib"):
                row[f"gpu{idx}_{field_name}"] = g[field_name]
        return row


# --------------------------------------------------------------------------- IPMI


_IPMI_RE = re.compile(r"Instantaneous power reading:\s*([0-9]+(?:\.[0-9]+)?)\s*Watts", re.I)


def parse_ipmi_power(text: str) -> float:
    m = _IPMI_RE.search(text)
    return float(m.group(1)) if m else NAN


class IpmiSource(Source):
    """Whole-chassis power from the BMC via ``ipmitool dcmi power reading``."""

    id = "ipmi"
    title = "system power (IPMI DCMI)"
    column_name = "system_w"

    def __init__(
        self,
        interval: float = 1.0,
        every: int = 1,
        runner: Runner = default_runner,
        which: Which = shutil.which,
        log: Logger = _noop_log,
    ) -> None:
        super().__init__(log)
        self.interval = interval
        self.every = max(1, int(every))
        self.runner = runner
        self.which = which
        self.timeout = max(0.5, 0.8 * interval)
        self._binary: Optional[str] = None
        self._tick = 0

    def _args(self) -> List[str]:
        return [self._binary or "ipmitool", "dcmi", "power", "reading"]

    def probe(self) -> ProbeResult:
        self._binary = self.which("ipmitool")
        if not self._binary:
            return self._result("absent", "ipmitool not found")
        try:
            value = parse_ipmi_power(self.runner(self._args(), 10.0))
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)[:200]
            is_root = getattr(os, "geteuid", lambda: 0)() == 0
            if not is_root or "permission" in detail.lower():
                return self._result("denied", f"ipmitool failed (try sudo): {detail}")
            return self._result("absent", f"ipmitool failed: {detail}")
        if math.isnan(value):
            return self._result("absent", "no 'Instantaneous power reading' in output")
        self._columns.append(
            Column(self.column_name, "W", "power", self.id, " ".join(self._args()), "chassis")
        )
        note = f"every {self.every} tick(s)" if self.every > 1 else ""
        return self._result("ok", note)

    def read(self, now: float) -> Dict[str, float]:
        row = self._nan_row()
        self._tick += 1
        if (self._tick - 1) % self.every != 0:
            return row
        try:
            row[self.column_name] = parse_ipmi_power(self.runner(self._args(), self.timeout))
        except subprocess.TimeoutExpired:
            self._log_failure(f"ipmitool timed out after {self.timeout:.1f}s")
        except Exception as exc:  # noqa: BLE001
            self._log_failure(f"ipmitool: {str(exc)[:200]}")
        return row


# --------------------------------------------------------------------------- demo


class DemoSource(Source):
    """Synthetic CPU + 2 GPU + chassis signals. For demos, tests, sensor-less hosts."""

    id = "demo"
    title = "demo (synthetic)"

    def __init__(self, seed: int = 7, log: Logger = _noop_log) -> None:
        super().__init__(log)
        self._rng = random.Random(seed)
        self._t0: Optional[float] = None

    def probe(self) -> ProbeResult:
        self._columns = [
            Column("cpu_pkg0_w", "W", "power", self.id, "synthetic", "package-0"),
            Column("coretemp_package_id_0_c", "C", "temp", self.id, "synthetic", "Package id 0"),
            Column("gpu0_power_w", "W", "power", self.id, "synthetic", "Demo GPU"),
            Column("gpu0_temp_c", "C", "temp", self.id, "synthetic", "Demo GPU core"),
            Column("gpu0_util_pct", "%", "util", self.id, "synthetic", "Demo GPU util"),
            Column("gpu1_power_w", "W", "power", self.id, "synthetic", "Demo GPU"),
            Column("gpu1_temp_c", "C", "temp", self.id, "synthetic", "Demo GPU core"),
            Column("gpu1_util_pct", "%", "util", self.id, "synthetic", "Demo GPU util"),
            Column("system_w", "W", "power", self.id, "synthetic", "chassis"),
        ]
        return self._result("ok", "synthetic data, not real sensors")

    def read(self, now: float) -> Dict[str, float]:
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0
        n = self._rng.gauss
        load = 0.5 + 0.5 * math.sin(t / 20.0)
        cpu = 60 + 90 * load + n(0, 3)
        g0 = 120 + 550 * load + n(0, 8)
        g1 = 120 + 540 * (0.5 + 0.5 * math.sin(t / 20.0 + 0.6)) + n(0, 8)
        return {
            "cpu_pkg0_w": cpu,
            "coretemp_package_id_0_c": 40 + 30 * load + n(0, 0.5),
            "gpu0_power_w": g0,
            "gpu0_temp_c": 35 + 40 * load + n(0, 0.4),
            "gpu0_util_pct": max(0.0, min(100.0, 100 * load + n(0, 4))),
            "gpu1_power_w": g1,
            "gpu1_temp_c": 35 + 40 * (0.5 + 0.5 * math.sin(t / 20.0 + 0.6)) + n(0, 0.4),
            "gpu1_util_pct": max(0.0, min(100.0, 100 * (0.5 + 0.5 * math.sin(t / 20.0 + 0.6)) + n(0, 4))),
            "system_w": 250 + cpu + g0 + g1 + n(0, 10),
        }


# -------------------------------------------------------------------------- probe


def probe_all(
    interval: float = 1.0,
    enabled: Sequence[str] = ALL_SOURCES,
    root: Path = Path("/"),
    ipmi_every: int = 1,
    demo: bool = False,
    runner: Runner = default_runner,
    which: Which = shutil.which,
    log: Logger = _noop_log,
    per_core: bool = True,
) -> List[tuple]:
    """Build and probe every source. Returns ``[(source, ProbeResult), ...]``.

    Thermal zones are only kept when hwmon produced no temperature columns,
    because on most machines they duplicate coretemp/k10temp readings.
    """
    if demo:
        src = DemoSource(log=log)
        return [(src, src.probe())]
    enabled_set = set(enabled)
    results: List[tuple] = []

    def add(src: Source) -> ProbeResult:
        if src.id not in enabled_set:
            res = ProbeResult(src.id, src.title, "disabled")
        else:
            res = src.probe()
        results.append((src, res))
        return res

    add(RaplSource(root, log))
    hwmon = HwmonSource(root, log, skip_per_core=not per_core)
    add(hwmon)
    thermal = ThermalZoneSource(root, log)
    if "thermal" in enabled_set and hwmon.temp_count > 0:
        results.append((thermal, ProbeResult(thermal.id, thermal.title, "disabled", note="hwmon already provides temperatures")))
    else:
        add(thermal)
    add(BatterySource(root, log))
    add(NvidiaSmiSource(interval, runner, which, log))
    add(IpmiSource(interval, ipmi_every, runner, which, log))
    return results


def active_sources(probed: Sequence[tuple]) -> List[Source]:
    return [src for src, res in probed if res.status == "ok" and src.columns]


def all_columns(sources: Sequence[Source]) -> List[Column]:
    cols: List[Column] = []
    seen: set = set()
    for src in sources:
        for col in src.columns:
            if col.name in seen:
                continue
            seen.add(col.name)
            cols.append(col)
    return cols

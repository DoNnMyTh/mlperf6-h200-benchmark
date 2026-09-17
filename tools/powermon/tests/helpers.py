"""Shared fixtures: fake sysfs tree builder, fake clock, canned subprocess runners."""

from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TOOL_ROOT = Path(__file__).resolve().parent.parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class FakeSysfs:
    """Builds /sys/class/... under a temporary root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def rapl(self, pkg: int, energy: int, max_range: int = 262143328850, name: Optional[str] = None,
             sub: Optional[int] = None, mmio: bool = False) -> Path:
        prefix = "intel-rapl-mmio" if mmio else "intel-rapl"
        dirname = f"{prefix}:{pkg}" if sub is None else f"{prefix}:{pkg}:{sub}"
        d = self.root / "sys" / "class" / "powercap" / dirname
        write(d / "name", name or (f"package-{pkg}" if sub is None else "dram"))
        write(d / "energy_uj", str(energy))
        write(d / "max_energy_range_uj", str(max_range))
        return d

    def hwmon(self, idx: int, name: str, temps: Optional[Dict[int, tuple]] = None,
              powers: Optional[Dict[int, tuple]] = None) -> Path:
        """temps: {n: (millideg, label_or_None)}; powers: {n: (microwatt, suffix)}."""
        d = self.root / "sys" / "class" / "hwmon" / f"hwmon{idx}"
        write(d / "name", name)
        for n, (val, label) in (temps or {}).items():
            write(d / f"temp{n}_input", str(val))
            if label:
                write(d / f"temp{n}_label", label)
        for n, (val, suffix) in (powers or {}).items():
            write(d / f"power{n}_{suffix}", str(val))
        return d

    def thermal(self, idx: int, ztype: str, millideg: int) -> Path:
        d = self.root / "sys" / "class" / "thermal" / f"thermal_zone{idx}"
        write(d / "type", ztype)
        write(d / "temp", str(millideg))
        return d

    def battery(self, name: str = "BAT0", power_now: Optional[int] = None,
                current_now: Optional[int] = None, voltage_now: Optional[int] = None) -> Path:
        d = self.root / "sys" / "class" / "power_supply" / name
        write(d / "type", "Battery")
        if power_now is not None:
            write(d / "power_now", str(power_now))
        if current_now is not None:
            write(d / "current_now", str(current_now))
        if voltage_now is not None:
            write(d / "voltage_now", str(voltage_now))
        return d


class FakeClock:
    """Deterministic clock; sleep() advances monotonic time instead of blocking."""

    def __init__(self, start: float = 1000.0) -> None:
        self._mono = start
        self._wall = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self.sleeps: List[float] = []

    def monotonic(self) -> float:
        return self._mono

    def now(self) -> datetime:
        return self._wall

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(max(0.0, seconds))

    def advance(self, seconds: float) -> None:
        self._mono += seconds
        self._wall += timedelta(seconds=seconds)


NVSMI_TWO_GPUS = (
    "0, NVIDIA H200 NVL, 412.30, 61, 72, 97, 120345\n"
    "1, NVIDIA H200 NVL, [N/A], 58, N/A, 95, 118000\n"
)

IPMI_OUTPUT = """
    Instantaneous power reading:                   1234 Watts
    Minimum during sampling period:                 800 Watts
    Maximum during sampling period:                1500 Watts
    Average power reading over sample period:      1100 Watts
"""


class CannedRunner:
    """Returns queued outputs per binary name; raises queued exceptions."""

    def __init__(self, outputs: Optional[Dict[str, List]] = None) -> None:
        self.outputs = {k: list(v) for k, v in (outputs or {}).items()}
        self.calls: List[Sequence[str]] = []

    def __call__(self, args: Sequence[str], timeout: float) -> str:
        self.calls.append(list(args))
        binary = os.path.basename(args[0])
        queue = self.outputs.get(binary)
        if not queue:
            raise RuntimeError(f"no canned output for {binary}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item


def which_all(name: str) -> Optional[str]:
    return f"/usr/bin/{name}"


def which_none(name: str) -> Optional[str]:
    return None


def timeout_error(cmd: str = "nvidia-smi") -> subprocess.TimeoutExpired:
    return subprocess.TimeoutExpired(cmd, 0.8)

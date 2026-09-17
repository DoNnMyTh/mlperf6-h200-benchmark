"""Tolerant CSV loading and per-column statistics for a recorded run."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .recorder import SAMPLES_CSV, read_sensors, read_status

NAN = float("nan")


class NoData(Exception):
    """Raised when a CSV has fewer than two usable samples."""


@dataclass
class ColumnMeta:
    name: str
    unit: str
    kind: str
    source: str = ""
    label: str = ""


@dataclass
class RunData:
    columns: List[str]
    timestamps: List[str]
    elapsed: List[float]
    values: Dict[str, List[float]]
    dropped_rows: int = 0

    @property
    def n(self) -> int:
        return len(self.elapsed)


@dataclass
class ColumnStats:
    name: str
    unit: str
    kind: str
    n: int
    min: float
    mean: float
    max: float
    p95: float
    last: float
    energy_wh: Optional[float] = None
    label: str = ""
    source: str = ""


@dataclass
class Summary:
    run_dir: Path
    data: RunData
    meta: Dict[str, ColumnMeta]
    stats: List[ColumnStats]
    derived: Dict[str, List[float]] = field(default_factory=dict)
    derived_stats: List[ColumnStats] = field(default_factory=list)
    status: Optional[dict] = None
    sensors: Optional[dict] = None

    @property
    def duration_s(self) -> float:
        return self.data.elapsed[-1] - self.data.elapsed[0] if self.data.n else 0.0

    @property
    def interval_s(self) -> float:
        if self.sensors and self.sensors.get("interval_s"):
            return float(self.sensors["interval_s"])
        if self.data.n > 1:
            return self.duration_s / (self.data.n - 1)
        return 1.0

    @property
    def expected_samples(self) -> int:
        if self.interval_s <= 0:
            return self.data.n
        return int(round(self.duration_s / self.interval_s)) + 1

    @property
    def gaps(self) -> int:
        if self.status and "gaps" in self.status:
            return int(self.status["gaps"])
        return max(0, self.expected_samples - self.data.n)

    @property
    def state(self) -> str:
        if self.status and self.status.get("state"):
            return str(self.status["state"])
        return "unknown"

    @property
    def complete(self) -> bool:
        return self.state == "done"

    def by_kind(self, kind: str) -> List[ColumnStats]:
        return [s for s in self.stats if s.kind == kind]


# --------------------------------------------------------------------- loading


def _to_float(token: str) -> float:
    token = token.strip()
    if not token:
        return NAN
    try:
        return float(token)
    except ValueError:
        return NAN


def load_csv(path: Path) -> RunData:
    """Read samples.csv, skipping short/truncated rows. Raises NoData if < 2 rows."""
    path = Path(path)
    if not path.exists():
        raise NoData(f"{path} does not exist")
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise NoData(f"{path} is empty") from None
        if len(header) < 2 or header[0] != "timestamp" or header[1] != "elapsed_s":
            raise NoData(f"{path} has an unexpected header: {header[:3]}")
        columns = header[2:]
        timestamps: List[str] = []
        elapsed: List[float] = []
        values: Dict[str, List[float]] = {c: [] for c in columns}
        dropped = 0
        for row in reader:
            if len(row) != len(header):
                dropped += 1
                continue
            e = _to_float(row[1])
            if math.isnan(e):
                dropped += 1
                continue
            timestamps.append(row[0])
            elapsed.append(e)
            for c, tok in zip(columns, row[2:]):
                values[c].append(_to_float(tok))
    if len(elapsed) < 2:
        raise NoData(f"{path} has {len(elapsed)} usable sample(s); need at least 2")
    return RunData(columns, timestamps, elapsed, values, dropped)


def infer_meta(name: str) -> ColumnMeta:
    if name.endswith("_w"):
        return ColumnMeta(name, "W", "power")
    if name.endswith("_c"):
        return ColumnMeta(name, "C", "temp")
    if name.endswith("_pct"):
        return ColumnMeta(name, "%", "util")
    if name.endswith("_mib"):
        return ColumnMeta(name, "MiB", "mem")
    return ColumnMeta(name, "", "other")


def load_meta(run_dir: Path, columns: Sequence[str]) -> Dict[str, ColumnMeta]:
    sensors = read_sensors(run_dir) or {}
    meta: Dict[str, ColumnMeta] = {}
    for entry in sensors.get("columns", []):
        try:
            meta[entry["name"]] = ColumnMeta(
                entry["name"], entry.get("unit", ""), entry.get("kind", "other"),
                entry.get("source", ""), entry.get("label", ""),
            )
        except (KeyError, TypeError):
            continue
    for c in columns:
        meta.setdefault(c, infer_meta(c))
    return meta


# ------------------------------------------------------------------ statistics


def percentile(values: Sequence[float], pct: float) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return NAN
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(clean) - 1)
    frac = pos - lo
    return clean[lo] + (clean[hi] - clean[lo]) * frac


def energy_wh(elapsed: Sequence[float], watts: Sequence[float]) -> float:
    """Trapezoidal integral of watts over seconds -> watt-hours. NaN pairs are skipped."""
    total_ws = 0.0
    for i in range(1, len(elapsed)):
        a, b = watts[i - 1], watts[i]
        if math.isnan(a) or math.isnan(b):
            continue
        dt = elapsed[i] - elapsed[i - 1]
        if dt <= 0:
            continue
        total_ws += (a + b) / 2.0 * dt
    return total_ws / 3600.0


def column_stats(name: str, meta: ColumnMeta, elapsed: Sequence[float], values: Sequence[float]) -> ColumnStats:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return ColumnStats(name, meta.unit, meta.kind, 0, NAN, NAN, NAN, NAN, NAN, None, meta.label, meta.source)
    stats = ColumnStats(
        name=name,
        unit=meta.unit,
        kind=meta.kind,
        n=len(clean),
        min=min(clean),
        mean=sum(clean) / len(clean),
        max=max(clean),
        p95=percentile(clean, 95),
        last=clean[-1],
        label=meta.label,
        source=meta.source,
    )
    if meta.kind == "power":
        stats.energy_wh = energy_wh(elapsed, values)
    return stats


def _row_sum(series: Sequence[Sequence[float]], n: int) -> List[float]:
    out: List[float] = []
    for i in range(n):
        vals = [s[i] for s in series if not math.isnan(s[i])]
        out.append(sum(vals) if vals else NAN)
    return out


def derived_series(data: RunData, meta: Dict[str, ColumnMeta]) -> Dict[str, List[float]]:
    """Totals worth plotting: all GPUs, all CPU packages (top-level only)."""
    derived: Dict[str, List[float]] = {}
    gpu = [data.values[c] for c in data.columns if c.startswith("gpu") and c.endswith("_power_w")]
    cpu = [
        data.values[c]
        for c in data.columns
        if c.startswith("cpu_pkg") and c.endswith("_w") and c.count("_") == 2
    ]
    if len(gpu) > 1:
        derived["total_gpu_w"] = _row_sum(gpu, data.n)
    if len(cpu) > 1:
        derived["total_cpu_w"] = _row_sum(cpu, data.n)
    return derived


def summarize(run_dir: Path, csv_path: Optional[Path] = None) -> Summary:
    run_dir = Path(run_dir)
    data = load_csv(csv_path or run_dir / SAMPLES_CSV)
    meta = load_meta(run_dir, data.columns)
    stats = [column_stats(c, meta[c], data.elapsed, data.values[c]) for c in data.columns]
    derived = derived_series(data, meta)
    derived_stats = [
        column_stats(name, ColumnMeta(name, "W", "power", "derived", name.replace("_", " ")), data.elapsed, series)
        for name, series in derived.items()
    ]
    return Summary(
        run_dir=run_dir,
        data=data,
        meta=meta,
        stats=stats,
        derived=derived,
        derived_stats=derived_stats,
        status=read_status(run_dir),
        sensors=read_sensors(run_dir),
    )

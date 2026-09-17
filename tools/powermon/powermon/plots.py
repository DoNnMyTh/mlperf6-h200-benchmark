"""Charts for a run: matplotlib PNGs when available, stdlib SVG otherwise.

Both back ends draw the same ``Chart`` objects so the report looks the same
apart from rendering polish.
"""

from __future__ import annotations

import math
import os
import site
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .stats import Summary, is_per_core

PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf",
]
MAX_SERIES = 16
MAX_POINTS = 2000


@dataclass
class Series:
    name: str
    label: str
    values: List[float]


@dataclass
class Chart:
    key: str
    title: str
    y_label: str
    elapsed: List[float]
    series: List[Series] = field(default_factory=list)
    note: str = ""
    events: List[Tuple[float, str]] = field(default_factory=list)


@dataclass
class ChartOutput:
    key: str
    title: str
    png: Optional[Path] = None
    svg: Optional[str] = None


# ------------------------------------------------------------------ chart plan


def _has_data(values: Sequence[float]) -> bool:
    return any(not math.isnan(v) for v in values)


def _label(summary: Summary, name: str) -> str:
    meta = summary.meta.get(name)
    if meta and meta.label and meta.label != name:
        return f"{name} ({meta.label})" if len(meta.label) <= 24 else name
    return name


def _pick(summary: Summary, names: Sequence[str]) -> List[Series]:
    out = [Series(n, _label(summary, n), summary.data.values[n]) for n in names if _has_data(summary.data.values[n])]
    return out


def build_charts(summary: Summary) -> List[Chart]:
    cols = summary.data.columns
    meta = summary.meta
    elapsed = summary.data.elapsed
    is_gpu = lambda c: c.startswith("gpu")  # noqa: E731
    charts: List[Chart] = []

    power_other = [c for c in cols if meta[c].kind == "power" and not is_gpu(c)]
    gpu_power = [c for c in cols if meta[c].kind == "power" and is_gpu(c)]
    temps_other = [c for c in cols if meta[c].kind == "temp" and not is_gpu(c) and not is_per_core(c)]
    core_temps = [c for c in cols if meta[c].kind == "temp" and is_per_core(c)]
    gpu_temps = [c for c in cols if meta[c].kind == "temp" and is_gpu(c)]
    util = [c for c in cols if meta[c].kind == "util"]

    power_series = _pick(summary, power_other)
    for name, values in summary.derived.items():
        if _has_data(values):
            power_series.append(Series(name, name.replace("_", " "), values))
    if power_series:
        charts.append(Chart("power", "Power", "W", elapsed, power_series))
    if gpu_power:
        charts.append(Chart("gpu_power", "GPU power", "W", elapsed, _pick(summary, gpu_power)))
    if gpu_temps:
        charts.append(Chart("gpu_temps", "GPU temperatures", "°C", elapsed, _pick(summary, gpu_temps)))
    if temps_other:
        charts.append(Chart("temps", "Temperatures", "°C", elapsed, _pick(summary, temps_other)))
    if core_temps:
        charts.append(Chart("core_temps", "Per-core CPU temperatures", "°C", elapsed, _pick(summary, core_temps)))
    if util:
        charts.append(Chart("util", "GPU utilisation", "%", elapsed, _pick(summary, util)))

    out: List[Chart] = []
    for ch in charts:
        ch.series = [s for s in ch.series if _has_data(s.values)]
        if not ch.series:
            continue
        if len(ch.series) > MAX_SERIES:
            # Keep the hottest / highest series so the interesting ones survive.
            total = len(ch.series)
            ranked = sorted(ch.series, key=lambda s: max(v for v in s.values if not math.isnan(v)), reverse=True)
            keep = {s.name for s in ranked[:MAX_SERIES]}
            ch.series = [s for s in ch.series if s.name in keep]
            ch.note = f"showing {MAX_SERIES} highest of {total} series"
        ch.events = [(e.elapsed_s, e.text) for e in summary.events]
        out.append(ch)
    return out


def _downsample(chart: Chart) -> Tuple[List[float], List[List[float]]]:
    n = len(chart.elapsed)
    stride = max(1, int(math.ceil(n / MAX_POINTS)))
    xs = chart.elapsed[::stride]
    ys = [s.values[::stride] for s in chart.series]
    return xs, ys


def _time_axis(elapsed: Sequence[float]) -> Tuple[float, str]:
    span = (elapsed[-1] - elapsed[0]) if elapsed else 0.0
    if span >= 3 * 3600:
        return 3600.0, "time (h)"
    if span >= 180:
        return 60.0, "time (min)"
    return 1.0, "time (s)"


# --------------------------------------------------------------------- SVG


def _nice_ticks(lo: float, hi: float, count: int = 5) -> List[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0]
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, count)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if raw <= step:
            break
    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + step * 0.5:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _fmt_tick(v: float) -> str:
    if abs(v) >= 100 or v == int(v):
        return f"{v:.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def render_svg(chart: Chart, width: int = 960, height: int = 380) -> str:
    xs, ys = _downsample(chart)
    div, x_label = _time_axis(xs)
    xs = [x / div for x in xs]
    finite = [v for col in ys for v in col if not math.isnan(v)]
    y_lo, y_hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
    if y_hi - y_lo < 1e-9:
        y_hi = y_lo + 1.0
    pad = (y_hi - y_lo) * 0.08
    y_lo -= pad
    y_hi += pad
    if chart.y_label in ("%",):
        y_lo, y_hi = 0.0, 100.0
    elif y_lo < 0 and min(finite or [0]) >= 0:
        y_lo = 0.0
    yticks = _nice_ticks(y_lo, y_hi)
    y_lo, y_hi = min(y_lo, yticks[0]), max(y_hi, yticks[-1])
    x_lo, x_hi = (xs[0], xs[-1]) if xs else (0.0, 1.0)
    if x_hi - x_lo < 1e-9:
        x_hi = x_lo + 1.0
    xticks = _nice_ticks(x_lo, x_hi, 8)

    legend_rows = len(chart.series)
    legend_w = 230
    ml, mr, mt, mb = 64, 20 + legend_w, 40, 48
    pw, ph = width - ml - mr, height - mt - mb

    def sx(x: float) -> float:
        return ml + (x - x_lo) / (x_hi - x_lo) * pw

    def sy(y: float) -> float:
        return mt + ph - (y - y_lo) / (y_hi - y_lo) * ph

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" font-size="12">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(f'<text x="{ml}" y="22" font-size="16" font-weight="600" fill="#222">{_esc(chart.title)}</text>')
    if chart.note:
        parts.append(f'<text x="{width - mr}" y="22" text-anchor="end" fill="#777">{_esc(chart.note)}</text>')
    for t in yticks:
        if t < y_lo or t > y_hi:
            continue
        y = sy(t)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#555">{_fmt_tick(t)}</text>')
    for t in xticks:
        if t < x_lo or t > x_hi:
            continue
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" stroke="#f0f0f0"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt + ph + 18}" text-anchor="middle" fill="#555">{_fmt_tick(t)}</text>')
    parts.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>')
    parts.append(f'<text x="{ml + pw / 2:.1f}" y="{height - 12}" text-anchor="middle" fill="#333">{_esc(x_label)}</text>')
    parts.append(
        f'<text transform="translate(16 {mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle" fill="#333">{_esc(chart.y_label)}</text>'
    )
    for i, (series, col) in enumerate(zip(chart.series, ys)):
        color = PALETTE[i % len(PALETTE)]
        d: List[str] = []
        pen_down = False
        for x, y in zip(xs, col):
            if math.isnan(y):
                pen_down = False
                continue
            d.append(f"{'L' if pen_down else 'M'}{sx(x):.1f} {sy(y):.1f}")
            pen_down = True
        if d:
            parts.append(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>')
    for k, (ev_x, ev_text) in enumerate(chart.events):
        exx = ev_x / div
        if exx < x_lo or exx > x_hi:
            continue
        x = sx(exx)
        parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" stroke="#444" stroke-dasharray="4 3"/>')
        ty = mt + 12 + (k % 4) * 14
        parts.append(f'<text x="{x + 4:.1f}" y="{ty}" fill="#444" font-size="11">{_esc(ev_text[:28])}</text>')
    lx = ml + pw + 14
    for i, series in enumerate(chart.series[:legend_rows]):
        color = PALETTE[i % len(PALETTE)]
        ly = mt + 8 + i * 18
        if ly > mt + ph:
            break
        parts.append(f'<rect x="{lx}" y="{ly - 8}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx + 18}" y="{ly + 2}" fill="#333">{_esc(series.label[:34])}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# -------------------------------------------------------------------- PNG


def _ensure_user_site() -> None:
    """Add ~/.local site-packages if it appeared after this interpreter started.

    Python only registers the user site directory at startup; a worker that
    began before `pip install --user matplotlib` would otherwise never see it.
    """
    try:
        user_site = site.getusersitepackages()
    except Exception:  # noqa: BLE001
        return
    if user_site and os.path.isdir(user_site) and user_site not in sys.path:
        site.addsitedir(user_site)


def matplotlib_available() -> bool:
    _ensure_user_site()
    try:
        import matplotlib  # noqa: F401
    except Exception:  # noqa: BLE001 - any import problem means "no"
        return False
    return True


def render_png(chart: Chart, path: Path) -> bool:
    _ensure_user_site()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    xs, ys = _downsample(chart)
    div, x_label = _time_axis(xs)
    xs = [x / div for x in xs]
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=110)
    for i, (series, col) in enumerate(zip(chart.series, ys)):
        ax.plot(xs, col, label=series.label[:34], color=PALETTE[i % len(PALETTE)], linewidth=1.2)
    for k, (ev_x, ev_text) in enumerate(chart.events):
        exx = ev_x / div
        ax.axvline(exx, color="#444", linestyle="--", linewidth=0.9)
        ax.text(exx, 0.98 - (k % 4) * 0.07, " " + ev_text[:28], transform=ax.get_xaxis_transform(),
                fontsize=7.5, color="#444", va="top")
    ax.set_title(chart.title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(chart.y_label)
    if chart.y_label == "%":
        ax.set_ylim(0, 100)
    ax.grid(True, color="#e5e5e5")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    if chart.note:
        ax.text(1.0, 1.02, chart.note, transform=ax.transAxes, ha="right", fontsize=8, color="#777")
    fig.tight_layout()
    try:
        fig.savefig(path)
    finally:
        plt.close(fig)
    return True


def render_all(summary: Summary, out_dir: Path, mode: str = "auto") -> List[ChartOutput]:
    """Render every chart. mode: auto (png if matplotlib present) | png | svg."""
    out_dir = Path(out_dir)
    charts = build_charts(summary)
    use_png = mode == "png" or (mode == "auto" and matplotlib_available())
    outputs: List[ChartOutput] = []
    for chart in charts:
        result = ChartOutput(chart.key, chart.title)
        if use_png:
            png_path = out_dir / f"{chart.key}.png"
            if render_png(chart, png_path):
                result.png = png_path
        if result.png is None:
            result.svg = render_svg(chart)
            (out_dir / f"{chart.key}.svg").write_text(result.svg, encoding="utf-8")
        outputs.append(result)
    return outputs

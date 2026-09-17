"""report.md, self-contained report.html, and the terminal summary for a run."""

from __future__ import annotations

import base64
import html
import math
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import __version__
from .plots import ChartOutput, render_all
from .stats import ColumnStats, NoData, Summary, summarize

REPORT_MD = "report.md"
REPORT_HTML = "report.html"


@dataclass
class ReportPaths:
    run_dir: Path
    markdown: Path
    html: Optional[Path]
    images: List[Path]
    nodata: bool = False
    message: str = ""


# ------------------------------------------------------------------ helpers


def _f(value: Optional[float], digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _host_info() -> dict:
    info = {"host": platform.node(), "kernel": platform.release(), "os": ""}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                info["os"] = line.split("=", 1)[1].strip().strip('"')
                break
    except OSError:
        pass
    return info


def _wall(summary: Summary, index: int) -> str:
    try:
        return summary.data.timestamps[index].replace("T", " ")[:19]
    except IndexError:
        return "-"


def _stat_rows(stats: List[ColumnStats], energy: bool) -> List[List[str]]:
    rows = []
    for s in stats:
        row = [s.name, s.label or "", _f(s.min), _f(s.mean), _f(s.max), _f(s.p95), _f(s.last)]
        if energy:
            row.append(_f(s.energy_wh, 2))
        rows.append(row)
    return rows


def _md_table(header: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _html_table(header: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "<p class='muted'>none</p>"
    th = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _sections(summary: Summary) -> List[tuple]:
    """(title, header, rows) per stats table."""
    power_hdr = ["column", "label", "min W", "mean W", "max W", "p95 W", "last W", "energy Wh"]
    temp_hdr = ["column", "label", "min °C", "mean °C", "max °C", "p95 °C", "last °C"]
    other_hdr = ["column", "label", "min", "mean", "max", "p95", "last"]
    sections = []
    power = summary.by_kind("power") + summary.derived_stats
    if power:
        sections.append(("Power", power_hdr, _stat_rows(power, True)))
    temps = summary.by_kind("temp")
    if temps:
        sections.append(("Temperatures", temp_hdr, _stat_rows(temps, False)))
    other = [s for s in summary.stats if s.kind not in ("power", "temp")]
    if other:
        sections.append(("Utilisation and memory", other_hdr, _stat_rows(other, False)))
    return sections


def _overview(summary: Summary) -> List[List[str]]:
    info = _host_info()
    label = (summary.sensors or {}).get("label") or ""
    rows = [
        ["run directory", str(summary.run_dir)],
        ["label", label or "-"],
        ["host", f"{info['host']} ({info['os'] or 'unknown OS'}, kernel {info['kernel']})"],
        ["state", summary.state + ("" if summary.complete else "  (incomplete: partial data)")],
        ["start", _wall(summary, 0)],
        ["end", _wall(summary, -1)],
        ["duration", _fmt_duration(summary.duration_s)],
        ["interval", f"{summary.interval_s:g} s"],
        ["samples", f"{summary.data.n} (expected about {summary.expected_samples})"],
        ["gaps (skipped ticks)", str(summary.gaps)],
    ]
    if summary.data.dropped_rows:
        rows.append(["dropped malformed rows", str(summary.data.dropped_rows)])
    errors = (summary.status or {}).get("errors") or {}
    if errors:
        rows.append(["sensor read errors", ", ".join(f"{k}: {v}" for k, v in errors.items())])
    return rows


def _probe_rows(summary: Summary) -> List[List[str]]:
    rows = []
    for p in (summary.sensors or {}).get("probe", []):
        rows.append([p.get("title", p.get("source", "")), p.get("status", ""), str(len(p.get("columns", []))), p.get("note", "")])
    return rows


# ---------------------------------------------------------------- markdown


def render_markdown(summary: Summary, charts: List[ChartOutput]) -> str:
    out = [f"# powermon report\n"]
    out.append(_md_table(["field", "value"], _overview(summary)))
    probe = _probe_rows(summary)
    if probe:
        out.append("\n## Sensors\n")
        out.append(_md_table(["source", "status", "columns", "note"], probe))
    for title, header, rows in _sections(summary):
        out.append(f"\n## {title}\n")
        out.append(_md_table(header, rows))
    if charts:
        out.append("\n## Graphs\n")
        for ch in charts:
            fname = ch.png.name if ch.png else f"{ch.key}.svg"
            out.append(f"### {ch.title}\n\n![{ch.title}]({fname})\n")
    out.append(f"\n---\ngenerated by powermon {__version__} on {datetime.now().astimezone().isoformat(timespec='seconds')}\n")
    return "\n".join(out)


# -------------------------------------------------------------------- html


_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;color:#222;background:#fafafa;max-width:1200px}
h1{font-size:22px;margin:0 0 16px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:14px;margin:18px 0 6px;color:#444}
table{border-collapse:collapse;font-size:13px;background:#fff;margin:6px 0}
th,td{border:1px solid #e2e2e2;padding:4px 8px;text-align:left;white-space:nowrap}
th{background:#f0f0f0}td:nth-child(n+3){text-align:right;font-variant-numeric:tabular-nums}
.kv td:nth-child(2){text-align:left}.muted{color:#777}
.chart{background:#fff;border:1px solid #e2e2e2;padding:8px;margin:8px 0;overflow-x:auto}
.chart img,.chart svg{max-width:100%;height:auto;display:block}
footer{margin-top:32px;color:#777;font-size:12px}
"""


def render_html(summary: Summary, charts: List[ChartOutput]) -> str:
    title = "powermon report" + ((" - " + (summary.sensors or {}).get("label", "")) if (summary.sensors or {}).get("label") else "")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<table class='kv'>" + "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in _overview(summary)) + "</table>",
    ]
    probe = _probe_rows(summary)
    if probe:
        parts.append("<h2>Sensors</h2>" + _html_table(["source", "status", "columns", "note"], probe))
    if charts:
        parts.append("<h2>Graphs</h2>")
        for ch in charts:
            parts.append(f"<h3>{html.escape(ch.title)}</h3><div class='chart'>")
            if ch.png is not None:
                try:
                    b64 = base64.b64encode(ch.png.read_bytes()).decode("ascii")
                    parts.append(f"<img alt='{html.escape(ch.title)}' src='data:image/png;base64,{b64}'>")
                except OSError:
                    parts.append(f"<img alt='{html.escape(ch.title)}' src='{html.escape(ch.png.name)}'>")
            elif ch.svg:
                parts.append(ch.svg)
            parts.append("</div>")
    for title_, header, rows in _sections(summary):
        parts.append(f"<h2>{html.escape(title_)}</h2>" + _html_table(header, rows))
    parts.append(
        f"<footer>generated by powermon {__version__} on "
        f"{html.escape(datetime.now().astimezone().isoformat(timespec='seconds'))}</footer></body></html>"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------- terminal


def summary_text(summary: Summary, charts: Optional[List[ChartOutput]] = None) -> str:
    lines = ["powermon summary"]
    width = max(len(k) for k, _ in _overview(summary))
    for k, v in _overview(summary):
        lines.append(f"  {k:<{width}}  {v}")
    for title, header, rows in _sections(summary):
        lines.append("")
        lines.append(title)
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        lines.append("  " + "  ".join(h.ljust(widths[i]) if i < 2 else h.rjust(widths[i]) for i, h in enumerate(header)))
        for r in rows:
            lines.append("  " + "  ".join(c.ljust(widths[i]) if i < 2 else c.rjust(widths[i]) for i, c in enumerate(r)))
    lines.append("")
    lines.append(f"  csv     {summary.run_dir / 'samples.csv'}")
    lines.append(f"  report  {summary.run_dir / REPORT_MD}")
    lines.append(f"  html    {summary.run_dir / REPORT_HTML}")
    if charts:
        for ch in charts:
            lines.append(f"  graph   {ch.png if ch.png else summary.run_dir / (ch.key + '.svg')}")
    return "\n".join(lines)


# ---------------------------------------------------------------- generate


def generate(run_dir: Path, plots: str = "auto", csv_path: Optional[Path] = None) -> ReportPaths:
    """Build stats, graphs, report.md and report.html for a run directory.

    Never raises for missing/short data: writes a short report.md explaining it
    and returns ``nodata=True`` so callers can decide the exit code.
    """
    run_dir = Path(run_dir)
    md_path = run_dir / REPORT_MD
    try:
        summary = summarize(run_dir, csv_path)
    except NoData as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        md_path.write_text(f"# powermon report\n\nNo usable data: {exc}\n", encoding="utf-8")
        return ReportPaths(run_dir, md_path, None, [], nodata=True, message=str(exc))
    charts = render_all(summary, run_dir, plots)
    md_path.write_text(render_markdown(summary, charts), encoding="utf-8")
    html_path = run_dir / REPORT_HTML
    html_path.write_text(render_html(summary, charts), encoding="utf-8")
    images = [c.png if c.png else run_dir / f"{c.key}.svg" for c in charts]
    return ReportPaths(run_dir, md_path, html_path, images)


def load_summary(run_dir: Path, csv_path: Optional[Path] = None) -> Summary:
    return summarize(run_dir, csv_path)

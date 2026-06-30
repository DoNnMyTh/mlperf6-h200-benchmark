#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


RESULT_RE = re.compile(r"^RESULT,([^,]+),,([0-9]+(?:\.[0-9]+)?),(.+)$")

# Throughput / step-time patterns emitted across the benchmarks' logs. Used to
# surface hardware-performance numbers (especially for --quick-run perf windows,
# which intentionally do not converge and so produce no RESULT/run_stop line).
_NUM = r"([0-9]+(?:\.[0-9]+)?)"
THROUGHPUT_RES = [
    re.compile(r'"throughput"\s*:\s*' + _NUM),                       # MLLOG tracked_stats (flux/gpt_oss/llama31)
    re.compile(r"train_samples_per_second['\"]?\s*[:=]\s*" + _NUM),  # HF Trainer (llama2)
    re.compile(_NUM + r"\s*samples\s*/\s*s"),                        # generic "N samples/s"
]
STEPTIME_RES = [
    re.compile(r'"train_step_time"\s*:\s*' + _NUM),                  # flux MLLOG
    re.compile(r"step[_ ]time['\"]?\s*[:=]\s*" + _NUM),              # generic step_time
]


@dataclass
class BenchmarkReport:
    name: str
    results_dir: Path
    latest_log: Path | None
    runtime_seconds: str
    status: str
    notes: str
    perf: str


def _collect(patterns: list[re.Pattern[str]], lines: list[str]) -> list[float]:
    values: list[float] = []
    for line in lines:
        for rx in patterns:
            match = rx.search(line)
            if match:
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    pass
                break
    return values


def extract_perf(lines: list[str] | None) -> str:
    """Best-effort hardware-perf summary: mean throughput/step-time over the last
    measured steps (steady state). Returns '-' when no perf numbers are present."""
    if not lines:
        return "-"
    thr = _collect(THROUGHPUT_RES, lines)
    step = _collect(STEPTIME_RES, lines)
    if not thr and not step:
        return "-"
    parts: list[str] = []
    if thr:
        tail = thr[-50:]
        parts.append(f"~{sum(tail) / len(tail):.1f} samples/s (n={len(thr)})")
    if step:
        tail = step[-50:]
        parts.append(f"step ~{sum(tail) / len(tail):.3f}s")
    return ", ".join(parts)


@dataclass
class PipelineStatus:
    benchmark: str
    stage: str
    status: str
    note: str


def parse_env_file(path: Path) -> dict[str, str]:
    bash_path = resolve_bash()
    command = (
        "set -a && "
        f"source '{path}' >/dev/null 2>&1 && "
        "env -0"
    )
    proc = subprocess.run(
        [bash_path, "-lc", command],
        check=True,
        capture_output=True,
    )
    env: dict[str, str] = {}
    for entry in proc.stdout.decode("utf-8", errors="replace").split("\0"):
        if not entry or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        env[key] = value
    return env


def resolve_bash() -> str:
    candidates = [
        os.environ.get("BASH"),
        shutil.which("bash"),
        shutil.which("sh"),
        r"C:\Program Files\Git\bin\bash.exe",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            probe = subprocess.run(
                [candidate, "-lc", "printf ok"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        if probe.stdout == "ok":
            return candidate

    raise RuntimeError("Unable to find a working bash executable to source the env file.")


def latest_log_file(results_dir: Path) -> Path | None:
    if not results_dir.exists():
        return None
    candidates = [
        path for path in results_dir.rglob("*")
        if path.is_file() and path.suffix in {".log", ".out", ".txt"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def inspect_log(log_path: Path | None, lines: list[str] | None) -> tuple[str, str, str]:
    if log_path is None:
        return ("not-run", "-", "No log files found")
    if lines is None:
        return ("unreadable", "-", f"Failed to read {log_path}")

    runtime = "-"
    status = "log-found"
    notes = f"Latest log: {log_path}"

    # A quick-run perf window is stopped by timeout, so it never emits a
    # convergence RESULT/run_stop; flag it so the status reads as perf-only.
    quick = any("quick-run: sustained perf window" in line for line in lines)

    for line in lines:
        match = RESULT_RE.match(line.strip())
        if match:
            runtime = match.group(2)
            status = "result-found"
            notes = f"RESULT line found in {log_path.name}"

    for line in reversed(lines):
        if "run_stop" in line.lower() and "success" in line.lower():
            status = "success"
            notes = f"MLPerf run_stop success found in {log_path.name}"
            break
        if "Training failed with exit code" in line:
            status = "failed"
            notes = line.strip()
            break

    if quick and status not in {"failed"}:
        status = "perf-only (quick-run)"
        notes = "Quick-run perf window (time-boxed, not convergence); see Hardware Perf column"

    return (status, runtime, notes)


def build_reports(env: dict[str, str]) -> list[BenchmarkReport]:
    rows = [
        ("Small LLM Pretraining", Path(env["MLPERF_LLAMA31_RESULTS_PATH"])),
        ("Llama 2 70B LoRA", Path(env["MLPERF_LLAMA2_RESULTS_PATH"])),
        ("GPT-OSS 20B", Path(env["MLPERF_GPT_OSS_RESULTS_PATH"])),
        ("FLUX.1", Path(env["MLPERF_FLUX_RESULTS_PATH"])),
    ]
    reports: list[BenchmarkReport] = []
    for name, results_dir in rows:
        log_path = latest_log_file(results_dir)
        lines: list[str] | None = None
        if log_path is not None:
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = None
        status, runtime, notes = inspect_log(log_path, lines)
        reports.append(
            BenchmarkReport(
                name=name,
                results_dir=results_dir,
                latest_log=log_path,
                runtime_seconds=runtime,
                status=status,
                notes=notes,
                perf=extract_perf(lines),
            )
        )
    return reports


def parse_pipeline_status(path: Path | None) -> list[PipelineStatus]:
    if path is None or not path.exists():
        return []

    rows: list[PipelineStatus] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[1:]:
        parts = line.split("\t", 3)
        if len(parts) != 4:
            continue
        rows.append(
            PipelineStatus(
                benchmark=parts[0],
                stage=parts[1],
                status=parts[2],
                note=parts[3],
            )
        )
    return rows


def render_markdown(
    env: dict[str, str],
    reports: list[BenchmarkReport],
    pipeline_rows: list[PipelineStatus],
) -> str:
    lines = [
        "# MLPerf 6.0 H200 Report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"- System: {env['MLPERF_SYSTEM_NAME']}",
        f"- GPUs: {env['MLPERF_GPU_COUNT']}x {env['MLPERF_GPU_MODEL']}",
        f"- GPU memory: {env['MLPERF_GPU_MEMORY_MIB']} MiB each",
        f"- Driver: {env['MLPERF_GPU_DRIVER_VERSION']}",
        f"- CPU affinity near GPUs: {env['MLPERF_GPU_CPU_AFFINITY']}",
        f"- GPU NUMA affinity: {env['MLPERF_GPU_NUMA_AFFINITY']}",
        f"- InfiniBand device: {env['MLPERF_IB_DEVICE']} ({env['MLPERF_IB_LINK_LAYER']})",
        f"- Llama2 LoRA mode: {env.get('MLPERF_LLAMA2_MODE', 'official')}",
        "",
        "## Benchmark Summary",
        "",
        "| Benchmark | Status | Runtime (s) | Hardware Perf | Results Dir | Notes |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]

    for report in reports:
        lines.append(
            f"| {report.name} | {report.status} | {report.runtime_seconds} | "
            f"{report.perf} | `{report.results_dir}` | {report.notes} |"
        )

    if pipeline_rows:
        lines.extend(
            [
                "",
                "## Pipeline Status",
                "",
                "| Benchmark | Stage | Status | Note |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in pipeline_rows:
            lines.append(
                f"| {row.benchmark} | {row.stage} | {row.status} | {row.note} |"
            )

    lines.extend(
        [
            "",
            "## Next Checks",
            "",
            "- Review the latest log under each results directory for target loss and compliance messages.",
            "- Attach the generated markdown plus the raw benchmark logs to your final benchmark packet.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a markdown summary for MLPerf 6.0 runs.")
    parser.add_argument("--env-file", required=True, help="Path to configs/mlperf6-h200-4gpu.env")
    parser.add_argument("--output", help="Optional output markdown path")
    args = parser.parse_args()

    env = parse_env_file(Path(args.env_file))
    reports = build_reports(env)
    pipeline_rows = parse_pipeline_status(
        Path(env["MLPERF_PIPELINE_STATUS_PATH"])
        if "MLPERF_PIPELINE_STATUS_PATH" in env
        else None
    )
    markdown = render_markdown(env, reports, pipeline_rows)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(output_path)
    else:
        print(markdown, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

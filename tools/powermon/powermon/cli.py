"""Command-line interface: interactive wizard, start/status/stop/report/probe."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__, report
from .recorder import (
    PID_FILE,
    RUN_JSON,
    SAMPLES_CSV,
    STATE_RUNNING,
    WORKER_LOG,
    Recorder,
    RunConfig,
    read_status,
)
from .sources import ALL_SOURCES, ProbeResult, active_sources, all_columns, probe_all

TOOL_ROOT = Path(__file__).resolve().parent.parent
SHIM = TOOL_ROOT / "powermon.py"
DEFAULT_OUT = "./powermon_runs"
DEFAULT_DURATION = "300"
MIN_FREE_BYTES = 50 * 1024 * 1024
REGISTRY_MAX = 20


# ---------------------------------------------------------------- utilities


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_line(msg: str) -> None:
    print(f"[{_stamp()}] {msg}", flush=True)


def parse_duration(text: str) -> float:
    """'300', '90s', '5m', '2h', '1.5h', '1d' -> seconds. '0' means until stopped."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", str(text).lower())
    if not m:
        raise ValueError(f"invalid duration {text!r}; use e.g. 300, 90s, 5m, 2h, or 0 for until-stopped")
    value = float(m.group(1))
    mult = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    seconds = value * mult
    if seconds < 0:
        raise ValueError("duration must be >= 0")
    return seconds


def parse_interval(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"invalid interval {text!r}") from None
    if not 0.2 <= value <= 3600:
        raise ValueError("interval must be between 0.2 and 3600 seconds")
    return value


def parse_sources(text: Optional[str]) -> List[str]:
    if not text or text.strip().lower() == "all":
        return list(ALL_SOURCES)
    chosen = [s.strip().lower() for s in text.split(",") if s.strip()]
    bad = [s for s in chosen if s not in ALL_SOURCES]
    if bad:
        raise ValueError(f"unknown source(s) {', '.join(bad)}; choose from {', '.join(ALL_SOURCES)}")
    return chosen


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "posix":
        # On Windows os.kill(pid, 0) would TERMINATE the process. Never probe there.
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():
        try:
            return b"powermon" in cmdline.read_bytes()
        except OSError:
            return True
    return True


def state_dir() -> Path:
    env = os.environ.get("POWERMON_STATE_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / "powermon"


def registry_path() -> Path:
    return state_dir() / "active.json"


def registry_load() -> List[dict]:
    try:
        data = json.loads(registry_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def registry_save(entries: List[dict]) -> None:
    try:
        registry_path().parent.mkdir(parents=True, exist_ok=True)
        registry_path().write_text(json.dumps(entries[-REGISTRY_MAX:], indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not write registry {registry_path()}: {exc}", file=sys.stderr)


def registry_add(run_dir: Path, pid: int, label: str) -> None:
    entries = [e for e in registry_load() if e.get("run_dir") != str(run_dir)]
    entries.append({"run_dir": str(run_dir), "pid": pid, "label": label, "started_at": _stamp()})
    registry_save(entries)


def registry_prune() -> List[dict]:
    kept = [e for e in registry_load() if Path(str(e.get("run_dir", ""))).is_dir()]
    registry_save(kept)
    return kept


def run_pid(run_dir: Path) -> int:
    try:
        return int((run_dir / PID_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        status = read_status(run_dir)
        return int(status.get("pid", 0)) if status else 0


def is_running(run_dir: Path) -> bool:
    status = read_status(run_dir)
    if status and status.get("final"):
        return False
    return pid_alive(run_pid(run_dir))


def probe_table(probed: Sequence[tuple]) -> str:
    lines = []
    for src, res in probed:
        tag = f"[{res.status}]"
        detail = f"{len(res.columns)} column(s)" if res.columns else ""
        note = res.note
        if res.status == "disabled" and not note:
            note = "disabled"
        text = f"  {tag:<10} {res.title:<28} {detail:<14} {note}".rstrip()
        lines.append(text)
    return "\n".join(lines)


def tail(path: Path, n: int = 15) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


# ------------------------------------------------------------------ worker


def _run_recorder(cfg: RunConfig, logger, install_signals: bool) -> int:
    logger(f"powermon {__version__} worker pid {os.getpid()}")
    probed = probe_all(
        interval=cfg.interval_s, enabled=cfg.sources, ipmi_every=cfg.ipmi_every, demo=cfg.demo, log=logger
    )
    logger("sensors:\n" + probe_table(probed))
    sources = active_sources(probed)
    results: List[ProbeResult] = [res for _, res in probed]
    rec = Recorder(
        cfg, sources, results, log=logger,
        finalizer=lambda d: _finalize_report(d, cfg.plots, logger),
    )
    if not sources:
        logger("no readable sensors found; nothing to record (try sudo, or --demo)")
        rec.state = "error"
        rec.exit_code = 2
        rec.write_status(final=True)
        return 2
    if install_signals:
        rec.install_signal_handlers()
    cfg.path.mkdir(parents=True, exist_ok=True)
    (cfg.path / PID_FILE).write_text(str(os.getpid()), encoding="utf-8")
    cfg.save()
    return rec.run()


def _finalize_report(run_dir: Path, plots: str, logger) -> None:
    paths = report.generate(run_dir, plots)
    if paths.nodata:
        logger(f"report: {paths.message}")
    else:
        logger(f"report written: {paths.markdown} and {paths.html}")
        for img in paths.images:
            logger(f"graph: {img}")


def cmd_worker(args: argparse.Namespace) -> int:
    cfg = RunConfig.load(Path(args.config))
    return _run_recorder(cfg, log_line, install_signals=True)


# ------------------------------------------------------------------- start


def make_run_dir(out: Path, label: str, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{re.sub(r'[^A-Za-z0-9_-]+', '_', label).strip('_')}" if label else ""
    return (out / f"run_{stamp}{suffix}").resolve()


def check_out_dir(out: Path, create: bool) -> Optional[str]:
    if not out.exists():
        if not create:
            return f"{out} does not exist"
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"cannot create {out}: {exc}"
    if not out.is_dir():
        return f"{out} is not a directory"
    probe = out / f".powermon_write_test_{os.getpid()}"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return f"{out} is not writable: {exc}"
    return None


def free_space_warning(out: Path) -> Optional[str]:
    try:
        free = shutil.disk_usage(out).free
    except OSError:
        return None
    if free < MIN_FREE_BYTES:
        return f"warning: only {free / 1024 / 1024:.0f} MB free in {out}"
    return None


def launch_background(cfg: RunConfig) -> int:
    run_dir = cfg.path
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg.save()
    log_path = run_dir / WORKER_LOG
    if SHIM.exists():
        cmd = [sys.executable, str(SHIM), "_worker", str(cfg_path)]
    else:  # installed as a package without the shim
        cmd = [sys.executable, "-m", "powermon.cli", "_worker", str(cfg_path)]
    popen_kwargs = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    with open(log_path, "ab") as log_fh:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=log_fh, stderr=subprocess.STDOUT,
            cwd=str(TOOL_ROOT), **popen_kwargs,
        )
    (run_dir / PID_FILE).write_text(str(proc.pid), encoding="utf-8")
    registry_add(run_dir, proc.pid, cfg.label)
    # Give the worker a moment to probe and fail fast if it cannot record.
    deadline = time.monotonic() + 3.0
    status = None
    while time.monotonic() < deadline:
        status = read_status(run_dir)
        if status is not None or proc.poll() is not None:
            break
        time.sleep(0.1)
    if proc.poll() is not None and (status is None or status.get("state") != "done"):
        print(f"worker exited early (code {proc.returncode}). Last log lines:")
        print(tail(log_path))
        return 2
    print(f"Started. PID {proc.pid}")
    print(f"  run dir : {run_dir}")
    print(f"  csv     : {run_dir / SAMPLES_CSV}")
    print(f"  log     : {log_path}")
    print(f"  report  : {run_dir / report.REPORT_HTML}  (written when the run ends)")
    print("Commands:")
    print(f"  {_self_cmd()} status {run_dir}")
    print(f"  {_self_cmd()} stop {run_dir}     (ends early; report still generated)")
    return 0


def _self_cmd() -> str:
    return "powermon" if shutil.which("powermon") else f"python3 {SHIM.name}"


def build_config(args: argparse.Namespace, out: Path, label: str, duration: float, interval: float,
                 sources: List[str]) -> RunConfig:
    run_dir = make_run_dir(out, label, getattr(args, "run_dir", None))
    return RunConfig(
        run_dir=str(run_dir),
        duration_s=duration,
        interval_s=interval,
        label=label,
        sources=sources,
        ipmi_every=max(1, int(getattr(args, "ipmi_every", 1) or 1)),
        fsync_every=max(0, int(getattr(args, "fsync_every", 60))),
        demo=bool(getattr(args, "demo", False)),
        plots=getattr(args, "plots", "auto") or "auto",
    )


def require_posix(action: str) -> bool:
    if os.name == "posix":
        return True
    print(f"error: '{action}' needs a Linux host (sysfs, nvidia-smi, POSIX signals). "
          "'report' and 'probe --demo' work anywhere.")
    return False


def start_run(cfg: RunConfig, foreground: bool, overwrite: bool) -> int:
    if not require_posix("start"):
        return 2
    run_dir = cfg.path
    if (run_dir / SAMPLES_CSV).exists() and not overwrite:
        print(f"error: {run_dir} already contains {SAMPLES_CSV}; choose another folder or pass --overwrite")
        return 2
    if is_running(run_dir):
        print(f"error: a powermon worker is already recording into {run_dir}")
        return 2
    if foreground:
        rc = _run_recorder(cfg, log_line, install_signals=True)
        if rc == 0:
            try:
                summary = report.load_summary(run_dir)
                print(report.summary_text(summary))
            except Exception as exc:  # noqa: BLE001
                print(f"(summary unavailable: {exc})")
        return rc
    return launch_background(cfg)


def cmd_start(args: argparse.Namespace) -> int:
    try:
        duration = parse_duration(args.duration)
        interval = parse_interval(str(args.interval))
        sources = parse_sources(args.sources)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    out = Path(args.out).expanduser().resolve()
    problem = check_out_dir(out, create=True)
    if problem:
        print(f"error: {problem}")
        return 2
    warn = free_space_warning(out)
    if warn:
        print(warn)
    cfg = build_config(args, out, args.label or "", duration, interval, sources)
    if not args.yes and sys.stdin.isatty() and not args.foreground:
        print(f"Will record every {interval:g}s "
              + (f"for {fmt_hms(duration)}" if duration else "until stopped")
              + f" into {cfg.run_dir}")
        if not ask_yes_no("Start in background?", True):
            print("aborted")
            return 130
    return start_run(cfg, args.foreground, args.overwrite)


# ------------------------------------------------------------------ wizard


def ask(prompt: str, default: str, parse=None) -> str:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        value = raw or default
        if parse is None:
            return value
        try:
            parse(value)
            return value
        except ValueError as exc:
            print(f"  {exc}")


def ask_yes_no(prompt: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def wizard(args: argparse.Namespace) -> int:
    print(f"powermon {__version__} - Linux power and temperature recorder")
    if not require_posix("recording"):
        return 2
    print("Probing sensors...")
    probed = probe_all(interval=1.0, enabled=parse_sources(args.sources), ipmi_every=1)
    print(probe_table(probed))
    sources = active_sources(probed)
    denied = [res for _, res in probed if res.status == "denied"]
    if denied:
        print("  hint: [denied] sources need root. Re-run with sudo to include them.")
    demo = False
    if not sources:
        print("No readable sensors found on this host.")
        if not ask_yes_no("Record synthetic demo data instead (to try the tool)?", False):
            print("aborted")
            return 2
        demo = True
        probed = probe_all(demo=True)
        sources = active_sources(probed)
    n_cols = len(all_columns(sources))
    print()
    duration_txt = ask("Duration (e.g. 300, 5m, 2h; 0 = until 'stop')", DEFAULT_DURATION, parse_duration)
    duration = parse_duration(duration_txt)
    while True:
        out_txt = ask("Output folder", DEFAULT_OUT)
        out = Path(out_txt).expanduser().resolve()
        if not out.exists():
            if not ask_yes_no(f"{out} does not exist. Create it?", True):
                continue
        problem = check_out_dir(out, create=True)
        if problem:
            print(f"  {problem}")
            continue
        break
    warn = free_space_warning(out)
    if warn:
        print(f"  {warn}")
    interval = parse_interval(ask("Sample interval seconds", "1", parse_interval))
    label = ask("Run label (optional, used in folder name)", "")
    ns = argparse.Namespace(
        run_dir=None, ipmi_every=1, fsync_every=60, demo=demo, plots=args.plots,
    )
    cfg = build_config(ns, out, label, duration, interval, parse_sources(args.sources))
    print()
    print(f"Will record {n_cols} columns every {interval:g}s "
          + (f"for {fmt_hms(duration)}" if duration else "until stopped")
          + f"\n  into {cfg.run_dir}")
    if not ask_yes_no("Start in background?", True):
        print("aborted")
        return 130
    return start_run(cfg, foreground=False, overwrite=False)


# ------------------------------------------------------------------ status


def _status_line(run_dir: Path) -> str:
    status = read_status(run_dir)
    if status is None:
        alive = pid_alive(run_pid(run_dir))
        return f"{run_dir}\n  state    starting" if alive else f"{run_dir}\n  state    unknown (no status.json)"
    running = is_running(run_dir)
    state = status.get("state", "?")
    if state == STATE_RUNNING and not running:
        state = "dead (worker gone without finalizing)"
    duration = float(status.get("duration_s") or 0)
    elapsed = float(status.get("elapsed_s") or 0)
    lines = [f"{run_dir}", f"  state    {state}   pid {status.get('pid')}"]
    prog = f"  elapsed  {fmt_hms(elapsed)}"
    if duration:
        prog += f" / {fmt_hms(duration)}   remaining {fmt_hms(duration - elapsed)}"
    lines.append(prog)
    errors = status.get("errors") or {}
    err_txt = ", ".join(f"{k}={v}" for k, v in errors.items()) or "none"
    lines.append(f"  samples  {status.get('samples', 0)}   gaps {status.get('gaps', 0)}   read errors {err_txt}")
    last = status.get("last_row") or {}
    if last:
        power = [(k, v) for k, v in last.items() if k.endswith("_w")][:8]
        temps = [(k, v) for k, v in last.items() if k.endswith("_c")][:6]
        if power:
            lines.append("  power    " + "  ".join(f"{k}={v:.1f}" for k, v in power))
        if temps:
            lines.append("  temps    " + "  ".join(f"{k}={v:.1f}" for k, v in temps))
    return "\n".join(lines)


def _print_finished(run_dir: Path, plots: str) -> None:
    md = run_dir / report.REPORT_MD
    if not md.exists():
        paths = report.generate(run_dir, plots)
        if paths.nodata:
            print(f"  {paths.message}")
            return
    try:
        summary = report.load_summary(run_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"  summary unavailable: {exc}")
        return
    print(report.summary_text(summary))


def cmd_status(args: argparse.Namespace) -> int:
    if args.run_dir:
        dirs = [Path(args.run_dir).expanduser().resolve()]
    else:
        dirs = [Path(e["run_dir"]) for e in registry_prune()]
        if not dirs:
            print("no runs known to this user. Pass a run directory: powermon status <run_dir>")
            return 1
    rc = 0
    for run_dir in dirs:
        if not run_dir.is_dir():
            print(f"{run_dir}: not a directory")
            rc = 1
            continue
        print(_status_line(run_dir))
        status = read_status(run_dir)
        if status and status.get("final"):
            _print_finished(run_dir, args.plots)
        print()
    return rc


# -------------------------------------------------------------------- stop


def stop_run(run_dir: Path, timeout: float) -> int:
    pid = run_pid(run_dir)
    if not is_running(run_dir):
        status = read_status(run_dir)
        state = status.get("state") if status else "unknown"
        print(f"{run_dir}: not running (state {state})")
        return 0 if status else 1
    print(f"stopping pid {pid} ...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        print(f"error: could not signal pid {pid}: {exc}")
        return 1
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = read_status(run_dir)
        if status and status.get("final") and (run_dir / report.REPORT_MD).exists():
            break
        if not pid_alive(pid):
            break
        time.sleep(0.2)
    else:
        print("worker did not finish within timeout; it may still be writing the report")
        return 1
    print(_status_line(run_dir))
    _print_finished(run_dir, "auto")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    if not require_posix("stop"):
        return 2
    if args.run_dir:
        dirs = [Path(args.run_dir).expanduser().resolve()]
    else:
        dirs = [Path(e["run_dir"]) for e in registry_prune() if is_running(Path(e["run_dir"]))]
        if not dirs:
            print("no running powermon workers registered for this user. Pass a run directory: powermon stop <run_dir>")
            return 1
        if len(dirs) > 1 and not args.all:
            print("several workers are running; pass a run directory or --all:")
            for d in dirs:
                print(f"  {d}")
            return 1
    rc = 0
    for run_dir in dirs:
        rc = max(rc, stop_run(run_dir, args.timeout))
    return rc


# ------------------------------------------------------------------ report


def cmd_report(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve()
    if target.is_file():
        run_dir, csv_path = target.parent, target
    else:
        run_dir, csv_path = target, None
    if not run_dir.is_dir():
        print(f"error: {run_dir} is not a directory")
        return 2
    paths = report.generate(run_dir, args.plots, csv_path)
    if paths.nodata:
        print(f"no usable data: {paths.message}")
        return 2
    summary = report.load_summary(run_dir, csv_path)
    print(report.summary_text(summary))
    return 0


# ------------------------------------------------------------------- probe


def cmd_probe(args: argparse.Namespace) -> int:
    try:
        sources = parse_sources(args.sources)
        interval = parse_interval(str(args.interval))
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    probed = probe_all(interval=interval, enabled=sources, ipmi_every=1, demo=args.demo)
    print(probe_table(probed))
    active = active_sources(probed)
    cols = all_columns(active)
    print(f"\n{len(cols)} column(s) would be recorded:")
    for c in cols:
        print(f"  {c.name:<36} {c.unit:<4} {c.kind:<6} {c.path}")
    if any(res.status == "denied" for _, res in probed):
        print("\n[denied] sources need root: re-run with sudo.")
    return 0 if active else 1


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="powermon",
        description="Record Linux power and temperature sensors every second to CSV, then graph and report. "
                    "Run without arguments for the interactive wizard.",
    )
    p.add_argument("--version", action="version", version=f"powermon {__version__}")
    p.add_argument("--sources", default="all", help=f"comma list of sources to probe (default all): {', '.join(ALL_SOURCES)}")
    p.add_argument("--plots", choices=("auto", "png", "svg"), default="auto", help="graph back end (auto: PNG if matplotlib is installed, else SVG)")
    sub = p.add_subparsers(dest="command", metavar="{start,status,stop,report,probe}")

    s = sub.add_parser("start", help="start a recording (background unless --foreground)")
    s.add_argument("-d", "--duration", default=DEFAULT_DURATION, help="e.g. 300, 90s, 5m, 2h; 0 = until 'stop' (default 300)")
    s.add_argument("-o", "--out", default=DEFAULT_OUT, help=f"folder that receives run_<timestamp>/ (default {DEFAULT_OUT})")
    s.add_argument("-i", "--interval", default="1", help="seconds between samples (default 1)")
    s.add_argument("-l", "--label", default="", help="short label added to the run folder name")
    s.add_argument("--run-dir", default=None, help="exact run directory instead of <out>/run_<timestamp>")
    s.add_argument("--ipmi-every", type=int, default=1, help="query IPMI only every N samples (default 1)")
    s.add_argument("--fsync-every", type=int, default=60, help="fsync the CSV every N rows (0 disables; default 60)")
    s.add_argument("--demo", action="store_true", help="record synthetic data instead of real sensors")
    s.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    s.add_argument("--foreground", action="store_true", help="record in this terminal instead of the background")
    s.add_argument("--overwrite", action="store_true", help="allow reusing a run directory that already has samples.csv")
    s.set_defaults(func=cmd_start)

    st = sub.add_parser("status", help="show progress of running/finished recordings")
    st.add_argument("run_dir", nargs="?", help="run directory (default: all runs started by this user)")
    st.set_defaults(func=cmd_status)

    sp = sub.add_parser("stop", help="stop a background recording early (report is still generated)")
    sp.add_argument("run_dir", nargs="?", help="run directory (default: the single running one)")
    sp.add_argument("--all", action="store_true", help="stop every running recording")
    sp.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for the worker to finish (default 15)")
    sp.set_defaults(func=cmd_stop)

    r = sub.add_parser("report", help="(re)generate graphs and report from a run directory or samples.csv")
    r.add_argument("target", help="run directory or path to samples.csv")
    r.set_defaults(func=cmd_report)

    pr = sub.add_parser("probe", help="list detected sensors and the columns that would be recorded")
    pr.add_argument("-i", "--interval", default="1")
    pr.add_argument("--demo", action="store_true")
    pr.set_defaults(func=cmd_probe)

    w = sub.add_parser("_worker")  # internal: no help entry, hidden by the metavar above
    w.add_argument("config")
    w.set_defaults(func=cmd_worker)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            if not sys.stdin.isatty():
                parser.print_help()
                print("\nstdin is not a terminal; use: powermon start --duration 5m --out <folder> --yes")
                return 2
            return wizard(args)
        return int(args.func(args))
    except (KeyboardInterrupt, EOFError):
        print("\naborted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

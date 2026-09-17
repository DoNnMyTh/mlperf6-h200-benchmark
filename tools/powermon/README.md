# powermon — Linux power and temperature recorder

Records power (W) and temperatures (°C) once per second on any Linux host,
saves them as CSV, and when the run ends writes graphs plus a Markdown/HTML
report into the same folder. Runs in the background so you can start it, run
your workload, and come back for the results.

Python 3.8+ standard library only. `matplotlib` is optional (PNG graphs
instead of the built-in SVG charts).

## Quick start

```bash
cd tools/powermon
python3 powermon.py            # interactive wizard
```

The wizard probes the sensors, asks for duration, output folder, interval and
an optional label, then starts a background worker:

```text
powermon 0.1.0 - Linux power and temperature recorder
Probing sensors...
  [ok]       CPU power (RAPL)             3 column(s)
  [ok]       hwmon temps/power            8 column(s)
  [disabled] thermal zones                              hwmon already provides temperatures
  [absent]   battery power                              no battery
  [ok]       NVIDIA GPUs (nvidia-smi)     40 column(s)  8 GPU(s): NVIDIA H200 NVL
  [denied]   system power (IPMI DCMI)                   ipmitool failed (try sudo): ...
  hint: [denied] sources need root. Re-run with sudo to include them.

Duration (e.g. 300, 5m, 2h; 0 = until 'stop') [300]: 30m
Output folder [./powermon_runs]: /data/power
Sample interval seconds [1]:
Run label (optional, used in folder name) []: llama31-quickrun

Will record 51 columns every 1s for 00:30:00
  into /data/power/run_20260917_142233_llama31-quickrun
Start in background? [Y/n]:
Started. PID 41233
  run dir : /data/power/run_20260917_142233_llama31-quickrun
  csv     : .../samples.csv
  log     : .../worker.log
  report  : .../report.html  (written when the run ends)
Commands:
  python3 powermon.py status /data/power/run_20260917_142233_llama31-quickrun
  python3 powermon.py stop   /data/power/run_20260917_142233_llama31-quickrun
```

Non-interactive form (scripts, ssh one-liners):

```bash
python3 powermon.py start --duration 30m --out /data/power --label llama31 --yes
python3 powermon.py status            # progress of every run you started
python3 powermon.py stop              # end early; the report is still generated
python3 powermon.py report /data/power/run_20260917_142233_llama31   # regenerate graphs/report
python3 powermon.py probe             # what would be recorded, without recording
```

Try it without real sensors: `python3 powermon.py start --demo --duration 60 --yes`.

## What gets recorded

| Source | Where it reads | Columns | Needs root? |
| --- | --- | --- | --- |
| CPU package power (RAPL) | `/sys/class/powercap/intel-rapl:*/energy_uj` (Intel and AMD Zen) | `cpu_pkg0_w`, `cpu_pkg0_dram_w`, … | Yes on kernels ≥ 5.10 (`energy_uj` is root-only) |
| hwmon temperatures and power | `/sys/class/hwmon/hwmon*/temp*_input`, `power*_input` — coretemp, k10temp, nvme, acpitz, amdgpu (edge/junction/mem + power) | `coretemp_package_id_0_c`, `nvme_composite_c`, `amdgpu_power1_w`, … | No |
| Thermal zones | `/sys/class/thermal/thermal_zone*/temp` — only used when hwmon has no temperatures | `tz0_x86_pkg_temp_c`, … | No |
| Battery | `/sys/class/power_supply/BAT*/power_now` (or current × voltage) | `bat0_w` | No |
| NVIDIA GPUs | one `nvidia-smi --query-gpu=…` call per sample | `gpuN_power_w`, `gpuN_temp_c`, `gpuN_mem_temp_c`, `gpuN_util_pct`, `gpuN_mem_used_mib` | No |
| Chassis power (IPMI) | `ipmitool dcmi power reading` | `system_w` | Yes (BMC device access) |

Every source is optional and probed at start. Missing hardware shows as
`[absent]`, unreadable sensors as `[denied]` with a sudo hint. A sensor that
fails mid-run produces an empty cell, never a crash; failures are counted in
`status.json` and logged (first three, then every 60th).

`--sources rapl,nvidia` limits probing; `--ipmi-every 5` throttles slow BMCs.

## Output folder

Each run gets `<out>/run_<YYYYmmdd_HHMMSS>[_label]/` containing:

| File | Content |
| --- | --- |
| `samples.csv` | one row per sample: `timestamp` (local ISO 8601 with offset), `elapsed_s` (monotonic seconds since start), then one column per sensor. Empty cell = no reading. |
| `sensors.json` | column → unit, kind (`power`/`temp`/`util`/`mem`), source, sysfs path or command; probe results |
| `status.json` | live progress: state (`running`/`done`/`stopped`/`error`), samples, gaps, per-source error counts, last row |
| `report.md`, `report.html` | overview, sensor table, min/mean/max/p95/last per column, energy in Wh per power column, graphs. HTML is self-contained. |
| `power.png`, `gpu_power.png`, `gpu_temps.png`, `temps.png`, `util.png` | graphs (`.svg` instead when matplotlib is not installed) |
| `worker.log`, `run.json`, `powermon.pid` | worker log, run configuration, worker PID |

Energy (Wh) is the trapezoidal integral of each power column over
`elapsed_s`. `total_gpu_w` and `total_cpu_w` are derived sums (CPU total uses
package domains only, not DRAM/core sub-domains).

## How the background run works

`start` launches a detached worker (`setsid`, stdin closed, output to
`worker.log`), so closing the terminal or an ssh session does not stop it.
The worker samples on an absolute schedule (`t0 + k × interval`); if a sample
takes longer than the interval the missed ticks are counted as `gaps`. The CSV
is flushed every row and fsynced every 60 rows (`--fsync-every`), so a crash
or power loss keeps everything up to the last few seconds.

`stop` sends SIGTERM; the worker closes the CSV, marks the run `stopped` and
still writes the report. `report <dir|csv>` regenerates graphs and report at
any time, including from a partial CSV of a run that is still going.

`status` and `stop` with no argument use a per-user registry in
`$XDG_STATE_HOME/powermon/active.json` (default `~/.local/state/powermon/`,
override with `POWERMON_STATE_DIR`). A run started with `sudo` is registered
under root's home, so as a normal user pass the run directory explicitly:
`powermon status /data/power/run_…`.

## Graphs

`--plots auto` (default) uses matplotlib PNGs when it is importable and falls
back to standard-library SVG charts otherwise. `--plots svg` forces the
fallback; `--plots png` tries matplotlib and falls back if it is missing.

```bash
python3 -m pip install --user matplotlib     # optional, for PNG graphs
```

## Tests (Docker only)

Tests use fake sysfs trees, canned `nvidia-smi`/`ipmitool` output and a fake
clock, so they need no real sensors. The end-to-end tests spawn real
background workers with `--demo`, which needs POSIX signals: run them in the
provided container, not on Windows.

```bash
# from the repository root
docker build -f tools/powermon/Dockerfile.test -t powermon-test tools/powermon && docker run --rm powermon-test
# stdlib-only matrix (no matplotlib) and a newer interpreter
docker build -f tools/powermon/Dockerfile.test --build-arg WITH_MPL=0 -t powermon-test-nompl tools/powermon && docker run --rm powermon-test-nompl
docker build -f tools/powermon/Dockerfile.test --build-arg PYTHON_VERSION=3.12 -t powermon-test-312 tools/powermon && docker run --rm powermon-test-312
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `[denied] CPU power (RAPL)` | `energy_uj` is root-only on modern kernels. Run `sudo python3 powermon.py …`, or `sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj` for the current boot. |
| `[absent] NVIDIA GPUs` | `nvidia-smi` not on `PATH` or no driver loaded. |
| `[error] NVIDIA GPUs … version mismatch` | driver/library mismatch; reload the driver or reboot. |
| `[denied] system power (IPMI DCMI)` | `ipmitool` needs `/dev/ipmi0` (root) and a BMC that supports DCMI. |
| "No readable sensors found" | VM/WSL/container without sysfs sensors. Use `--demo` to try the tool, or run on the bare host. |
| `worker exited early` | see the printed tail of `worker.log`; usually an unwritable output folder. |
| status shows `dead (worker gone without finalizing)` | worker was killed with SIGKILL or the host rebooted. Data up to that point is in `samples.csv`; run `report <dir>` to build the report. |
| Graphs are `.svg` not `.png` | matplotlib not installed for the interpreter that ran the worker; install it and re-run `report <dir>`. |

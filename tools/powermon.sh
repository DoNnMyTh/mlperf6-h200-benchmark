#!/usr/bin/env bash
# One-shot launcher for powermon: sets up dependencies, then starts the tool.
#
#   ./tools/powermon.sh                 interactive wizard
#   ./tools/powermon.sh start -d 30m -o /data/power --yes
#   ./tools/powermon.sh status | stop | report <dir> | probe
#
# What it does on first run:
#   1. finds a python3 >= 3.8
#   2. creates tools/powermon/.venv (inherits distro packages, so an apt/dnf
#      python3-matplotlib is picked up without downloading anything)
#   3. installs tools/powermon/requirements.txt (matplotlib) if missing
#   4. execs powermon.py with every argument passed through
# Every step degrades gracefully: without venv/pip/network powermon still runs
# with the standard library and draws SVG graphs instead of PNG.
#
# Knobs: POWERMON_NO_INSTALL=1 skips dependency setup; POWERMON_PYTHON=<path>
# picks the interpreter.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOL_DIR="${SCRIPT_DIR}/powermon"
ENTRY="${TOOL_DIR}/powermon.py"
VENV="${TOOL_DIR}/.venv"
REQS="${TOOL_DIR}/requirements.txt"

note() { printf 'powermon.sh: %s\n' "$*" >&2; }

if [[ ! -f "${ENTRY}" ]]; then
  note "cannot find ${ENTRY}; run this script from a checkout of the repository"
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  note "powermon records Linux sensors; on $(uname -s) only 'report' and 'probe --demo' work"
fi

# --- 1. interpreter ---------------------------------------------------------
find_python() {
  local candidates=()
  [[ -n "${POWERMON_PYTHON:-}" ]] && candidates+=("${POWERMON_PYTHON}")
  candidates+=(python3 python3.13 python3.12 python3.11 python3.10 python3.9 python3.8 python)
  local c
  for c in "${candidates[@]}"; do
    if command -v "$c" >/dev/null 2>&1 \
       && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
      command -v "$c"
      return 0
    fi
  done
  return 1
}

if ! BASE_PY=$(find_python); then
  note "no python3 >= 3.8 found. Install it with your package manager, e.g.:"
  note "  Debian/Ubuntu: sudo apt install python3 python3-venv"
  note "  RHEL/Fedora:   sudo dnf install python3"
  note "  Arch:          sudo pacman -S python"
  exit 2
fi

PY="${BASE_PY}"

has_matplotlib() { "$1" -c 'import matplotlib' >/dev/null 2>&1; }

# --- 2./3. dependencies -----------------------------------------------------
if [[ "${POWERMON_NO_INSTALL:-0}" != "1" ]]; then
  if [[ -x "${VENV}/bin/python" ]]; then
    PY="${VENV}/bin/python"
  elif has_matplotlib "${BASE_PY}"; then
    : # distro already provides matplotlib for the base interpreter; no venv needed
  else
    note "creating virtual environment in ${VENV}"
    if "${BASE_PY}" -m venv --system-site-packages "${VENV}" >/dev/null 2>&1; then
      PY="${VENV}/bin/python"
    else
      rm -rf "${VENV}"
      note "could not create a venv (python3-venv missing?). Trying a user-level pip install instead."
      note "  Debian/Ubuntu fix: sudo apt install python3-venv"
    fi
  fi

  if ! has_matplotlib "${PY}"; then
    note "installing optional dependencies from ${REQS} (matplotlib, for PNG graphs)"
    if [[ "${PY}" == "${VENV}/bin/python" ]]; then
      pip_cmd=("${PY}" -m pip install --quiet --disable-pip-version-check -r "${REQS}")
    else
      pip_cmd=("${PY}" -m pip install --quiet --disable-pip-version-check --no-warn-script-location --user -r "${REQS}")
      # PEP 668 distros (Debian 12+, Ubuntu 23.04+) refuse pip outside a venv even
      # with --user. The flag only lifts that marker; --user still installs into
      # ~/.local, nothing under /usr is touched.
      if "${PY}" -m pip install --help 2>/dev/null | grep -q -- '--break-system-packages'; then
        pip_cmd+=(--break-system-packages)
      fi
    fi
    if ! "${pip_cmd[@]}"; then
      note "dependency install failed (offline, or pip missing). Continuing without matplotlib:"
      note "  recording works unchanged; graphs will be SVG instead of PNG."
      note "  later: ${PY} -m pip install -r ${REQS} && ./tools/powermon.sh report <run_dir>"
    fi
  fi
fi

# --- 4. run -----------------------------------------------------------------
if [[ $# -eq 0 && ! -t 0 ]]; then
  note "stdin is not a terminal; the wizard needs one. Example:"
  note "  ./tools/powermon.sh start --duration 5m --out ./powermon_runs --yes"
  exit 2
fi

exec "${PY}" "${ENTRY}" "$@"

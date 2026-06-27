#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${1:-${REPO_ROOT}/configs/mlperf6-h200-4gpu.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: env file not found: ${ENV_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

mkdir -p \
  "${MLPERF_WORK_ROOT}" \
  "${MLPERF_DATA_ROOT}" \
  "${MLPERF_RESULTS_ROOT}" \
  "${MLPERF_HF_CACHE}" \
  "${MLPERF_LLAMA31_RESULTS_PATH}" \
  "${MLPERF_LLAMA31_CHECKPOINT_PATH}" \
  "${MLPERF_LLAMA31_INDEX_PATH}" \
  "${MLPERF_LLAMA2_RESULTS_PATH}" \
  "${MLPERF_GPT_OSS_RESULTS_PATH}" \
  "${MLPERF_FLUX_RESULTS_PATH}" \
  "${REPO_ROOT}/generated"

if [[ -d "${MLPERF_UPSTREAM_DIR}/.git" ]]; then
  git -C "${MLPERF_UPSTREAM_DIR}" fetch --all --tags --prune
  git -C "${MLPERF_UPSTREAM_DIR}" pull --ff-only --recurse-submodules
else
  git clone --depth 1 --recurse-submodules https://github.com/mlcommons/training.git "${MLPERF_UPSTREAM_DIR}"
fi

git -C "${MLPERF_UPSTREAM_DIR}" submodule update --init --recursive text_to_image/torchtitan

cat <<EOF
Bootstrap complete.

Upstream MLCommons training repo:
  ${MLPERF_UPSTREAM_DIR}

Key directories:
  work root:    ${MLPERF_WORK_ROOT}
  data root:    ${MLPERF_DATA_ROOT}
  results root: ${MLPERF_RESULTS_ROOT}
  HF cache:     ${MLPERF_HF_CACHE}

Next:
  1. Review and edit ${ENV_FILE} if your data/model paths differ.
  2. Run ./scripts/run_mlperf6_h200.sh show all
  3. Run ./scripts/run_mlperf6_h200.sh --execute run <benchmark>
EOF


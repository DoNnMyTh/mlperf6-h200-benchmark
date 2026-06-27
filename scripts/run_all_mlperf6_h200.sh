#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/configs/mlperf6-h200-4gpu.env"
REPORT_OUTPUT=""
CONTINUE_ON_FAILURE=1
SKIP_BOOTSTRAP=0
SKIP_DOWNLOADS=0
SKIP_RUNS=0
BENCHMARKS_CSV="llama31,llama2_lora,gpt_oss20b,flux"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_all_mlperf6_h200.sh [options]

Options:
  --env-file PATH          Alternate env file
  --report-output PATH     Final markdown report path
  --benchmarks CSV         Comma-separated list from: llama31,llama2_lora,gpt_oss20b,flux
  --skip-bootstrap         Reuse the existing upstream checkout
  --skip-downloads         Skip data/model downloads
  --skip-runs              Skip benchmark execution
  --fail-fast              Stop at the first failing stage
  -h, --help               Show this help

Behavior:
  - bootstraps the official mlcommons/training repo
  - downloads public benchmark assets where official public URIs exist
  - uses the gated upstream Llama2 downloader when MLPERF_LLAMA2_RCLONE_CONFIG is set
  - runs the selected benchmarks sequentially
  - always attempts to write a final report, even if some stages fail
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --report-output)
      REPORT_OUTPUT="$2"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS_CSV="$2"
      shift 2
      ;;
    --skip-bootstrap)
      SKIP_BOOTSTRAP=1
      shift
      ;;
    --skip-downloads)
      SKIP_DOWNLOADS=1
      shift
      ;;
    --skip-runs)
      SKIP_RUNS=1
      shift
      ;;
    --fail-fast)
      CONTINUE_ON_FAILURE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: run_all_mlperf6_h200.sh must be run on the Linux benchmark server." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: env file not found: ${ENV_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

find_llama2_rclone_config() {
  local candidates=()

  if [[ -n "${MLPERF_LLAMA2_RCLONE_CONFIG:-}" ]]; then
    candidates+=("${MLPERF_LLAMA2_RCLONE_CONFIG}")
  fi

  candidates+=(
    "${HOME}/.config/mlperf/llama2-rclone.conf"
    "${HOME}/.config/rclone/rclone.conf"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]] && grep -q "\[mlc-llama2\]" "${candidate}" 2>/dev/null; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

resolve_llama2_mode() {
  local requested="${MLPERF_LLAMA2_MODE:-auto}"
  local official_dataset_dir="${MLPERF_LLAMA2_DATASET_PATH}/${MLPERF_LLAMA2_LOCAL_DATASET_SUBDIR}"
  local official_model_dir="${MLPERF_LLAMA2_MODEL_ROOT}/${MLPERF_LLAMA2_LOCAL_MODEL_SUBDIR}"
  local detected_rclone_config=""

  detected_rclone_config="$(find_llama2_rclone_config || true)"

  case "${requested}" in
    official)
      if [[ -d "${official_dataset_dir}" && -d "${official_model_dir}" ]]; then
        printf 'official\n'
      elif [[ -n "${detected_rclone_config}" ]]; then
        printf 'official\n'
      else
        echo "ERROR: official llama2_lora mode requested, but no official assets or usable rclone config were found." >&2
        return 1
      fi
      ;;
    local-only)
      if [[ -d "${official_dataset_dir}" ]]; then
        printf 'local-only\n'
      else
        echo "ERROR: local-only llama2_lora mode requested, but ${official_dataset_dir} is missing." >&2
        return 1
      fi
      ;;
    smoke-test|skip)
      printf '%s\n' "${requested}"
      ;;
    auto)
      if [[ -d "${official_dataset_dir}" && -d "${official_model_dir}" ]]; then
        printf 'official\n'
      elif [[ -n "${detected_rclone_config}" ]]; then
        printf 'official\n'
      elif [[ -d "${official_dataset_dir}" ]]; then
        printf 'local-only\n'
      else
        printf 'smoke-test\n'
      fi
      ;;
    *)
      echo "ERROR: unsupported MLPERF_LLAMA2_MODE=${requested}" >&2
      return 1
      ;;
  esac
}

MLPERF_LLAMA2_RCLONE_CONFIG="$(find_llama2_rclone_config || true)"
EFFECTIVE_MLPERF_LLAMA2_MODE="$(resolve_llama2_mode)"

if [[ -z "${REPORT_OUTPUT}" ]]; then
  REPORT_OUTPUT="${MLPERF_FINAL_REPORT_PATH}"
fi

mkdir -p "${MLPERF_ORCH_ROOT}"
STATUS_FILE="${MLPERF_PIPELINE_STATUS_PATH}"
RUN_LOG="${MLPERF_ORCH_ROOT}/orchestrator.log"
: > "${RUN_LOG}"
printf 'benchmark\tstage\tstatus\tnote\n' > "${STATUS_FILE}"

IFS=',' read -r -a BENCHMARKS <<< "${BENCHMARKS_CSV}"

record_status() {
  local benchmark="$1"
  local stage="$2"
  local status="$3"
  local note="$4"
  printf '%s\t%s\t%s\t%s\n' "${benchmark}" "${stage}" "${status}" "${note}" >> "${STATUS_FILE}"
}

log() {
  printf '%s %s\n' "[$(date -u +%Y-%m-%dT%H:%M:%SZ)]" "$*" | tee -a "${RUN_LOG}"
}

run_logged() {
  local benchmark="$1"
  local stage="$2"
  local command_text="$3"
  local stage_log="${MLPERF_ORCH_ROOT}/${benchmark}-${stage}.log"

  log "Starting ${benchmark}/${stage}"
  if bash -lc "${command_text}" 2>&1 | tee "${stage_log}"; then
    record_status "${benchmark}" "${stage}" "success" "${stage_log}"
    log "Finished ${benchmark}/${stage}"
    return 0
  fi

  record_status "${benchmark}" "${stage}" "failed" "${stage_log}"
  log "Failed ${benchmark}/${stage}"
  return 1
}

handle_failure() {
  local benchmark="$1"
  local stage="$2"
  if [[ "${CONTINUE_ON_FAILURE}" -eq 0 ]]; then
    log "Stopping after ${benchmark}/${stage} because --fail-fast was requested."
    exit 1
  fi
}

require_cmd() {
  local benchmark="$1"
  local stage="$2"
  local command_name="$3"
  if command -v "${command_name}" > /dev/null 2>&1; then
    record_status "${benchmark}" "${stage}" "success" "Found command: ${command_name}"
    return 0
  fi
  record_status "${benchmark}" "${stage}" "failed" "Missing command: ${command_name}"
  log "Missing required command for ${benchmark}/${stage}: ${command_name}"
  return 1
}

ensure_dir() {
  mkdir -p "$1"
}

download_r2_named() {
  local benchmark="$1"
  local name="$2"
  local destination="$3"
  local uri="$4"
  local completion_marker="${destination}/.mlperf-download-complete"

  if [[ -f "${completion_marker}" ]]; then
    record_status "${benchmark}" "download-${name}" "skipped" "Already present at ${destination}"
    return 0
  fi

  ensure_dir "$(dirname "${destination}")"
  local command_text
  command_text=$(cat <<EOF
set -euo pipefail
mkdir -p "$(dirname "${destination}")"
cd "$(dirname "${destination}")"
bash <(curl -fsSL "${MLPERF_R2_DOWNLOADER_URL}") -d "$(basename "${destination}")" "${uri}"
touch "${completion_marker}"
EOF
)
  run_logged "${benchmark}" "download-${name}" "${command_text}"
}

sync_directory() {
  local source_dir="$1"
  local destination_dir="$2"
  mkdir -p "${destination_dir}"
  if command -v rsync > /dev/null 2>&1; then
    rsync -a --delete "${source_dir}/" "${destination_dir}/"
  else
    rm -rf "${destination_dir}"
    mkdir -p "${destination_dir}"
    cp -a "${source_dir}/." "${destination_dir}/"
  fi
}

download_llama31() {
  download_r2_named "llama31" "dataset" "${MLPERF_LLAMA31_PREPROCESSED_PATH}" "${MLPERF_LLAMA31_DATASET_URI}" || return 1
  download_r2_named "llama31" "tokenizer" "${MLPERF_LLAMA31_TOKENIZER_PATH}" "${MLPERF_LLAMA31_TOKENIZER_URI}" || return 1
}

download_llama2_lora() {
  case "${EFFECTIVE_MLPERF_LLAMA2_MODE}" in
    official)
      if [[ -d "${MLPERF_LLAMA2_DATASET_PATH}/${MLPERF_LLAMA2_LOCAL_DATASET_SUBDIR}" && -d "${MLPERF_LLAMA2_MODEL_ROOT}/${MLPERF_LLAMA2_LOCAL_MODEL_SUBDIR}" ]]; then
        record_status "llama2_lora" "download" "skipped" "Official dataset and model already present"
        return 0
      fi

      if [[ -z "${MLPERF_LLAMA2_RCLONE_CONFIG}" ]]; then
        record_status "llama2_lora" "download" "failed" "Set MLPERF_LLAMA2_RCLONE_CONFIG to the MLCommons rclone.conf path"
        log "Llama2 LoRA official mode could not find a usable rclone config."
        return 1
      fi

      require_cmd "llama2_lora" "download" "rclone" || return 1

      local command_text
      command_text=$(cat <<EOF
set -euo pipefail
cd "${MLPERF_UPSTREAM_DIR}/llama2_70b_lora"
bash ./scripts/download_data.sh --data_dir="${MLPERF_LLAMA2_DATASET_PATH}" --model_dir="${MLPERF_LLAMA2_MODEL_ROOT}" --rclone_config="${MLPERF_LLAMA2_RCLONE_CONFIG}"
EOF
)
      run_logged "llama2_lora" "download" "${command_text}"
      ;;
    local-only)
      if [[ ! -d "${MLPERF_LLAMA2_DATASET_PATH}/${MLPERF_LLAMA2_LOCAL_DATASET_SUBDIR}" ]]; then
        record_status "llama2_lora" "download" "failed" "Local dataset not found at ${MLPERF_LLAMA2_DATASET_PATH}/${MLPERF_LLAMA2_LOCAL_DATASET_SUBDIR}"
        return 1
      fi
      record_status "llama2_lora" "download" "success" "Local-only mode will reuse local dataset and fetch the public HF model during run"
      ;;
    smoke-test)
      record_status "llama2_lora" "download" "success" "Smoke-test mode will prepare a public GovReport subset and public model during run"
      ;;
    skip)
      record_status "llama2_lora" "download" "skipped" "Llama2 benchmark skipped by configuration"
      ;;
    *)
      record_status "llama2_lora" "download" "failed" "Unknown resolved mode ${EFFECTIVE_MLPERF_LLAMA2_MODE}"
      return 1
      ;;
  esac
}

download_gpt_oss20b() {
  download_r2_named "gpt_oss20b" "dataset" "${MLPERF_GPT_OSS_DATA_PATH}" "${MLPERF_GPT_OSS_DATASET_URI}" || return 1

  if [[ ! -d "${MLPERF_GPT_OSS_TOKENIZER_SOURCE}" ]]; then
    record_status "gpt_oss20b" "tokenizer" "failed" "Tokenizer source missing: ${MLPERF_GPT_OSS_TOKENIZER_SOURCE}"
    return 1
  fi

  sync_directory "${MLPERF_GPT_OSS_TOKENIZER_SOURCE}" "${MLPERF_GPT_OSS_MODEL_PATH}"
  record_status "gpt_oss20b" "tokenizer" "success" "Synced tokenizer into ${MLPERF_GPT_OSS_MODEL_PATH}"
}

download_flux() {
  download_r2_named "flux" "cc12m-preprocessed" "${MLPERF_FLUX_DATASET_PATH}/cc12m_preprocessed" "${MLPERF_FLUX_CC12M_PREPROCESSED_URI}" || return 1
  download_r2_named "flux" "coco-preprocessed" "${MLPERF_FLUX_DATASET_PATH}/coco_preprocessed" "${MLPERF_FLUX_COCO_PREPROCESSED_URI}" || return 1
  download_r2_named "flux" "empty-encodings" "${MLPERF_FLUX_DATASET_PATH}/empty_encodings" "${MLPERF_FLUX_EMPTY_ENCODINGS_URI}" || return 1
}

run_benchmark() {
  local benchmark="$1"
  local command_text
  command_text=$(cat <<EOF
set -euo pipefail
cd "${REPO_ROOT}"
bash ./scripts/run_mlperf6_h200.sh --env-file "${ENV_FILE}" --execute run "${benchmark}"
EOF
)
  run_logged "${benchmark}" "run" "${command_text}"
}

generate_report() {
  local report_log="${MLPERF_ORCH_ROOT}/report.log"
  if python3 "${REPO_ROOT}/scripts/report_mlperf6_results.py" --env-file "${ENV_FILE}" --output "${REPORT_OUTPUT}" > "${report_log}" 2>&1; then
    record_status "pipeline" "report" "success" "${REPORT_OUTPUT}"
    log "Final report written to ${REPORT_OUTPUT}"
  else
    record_status "pipeline" "report" "failed" "${report_log}"
    log "Final report generation failed. See ${report_log}"
    return 1
  fi
}

on_exit() {
  local exit_code=$?
  generate_report || true
  exit "${exit_code}"
}

trap on_exit EXIT

log "Selected benchmarks: ${BENCHMARKS_CSV}"
log "Resolved llama2_lora mode: ${EFFECTIVE_MLPERF_LLAMA2_MODE}"

require_cmd "pipeline" "preflight" "docker" || handle_failure "pipeline" "preflight"
require_cmd "pipeline" "preflight" "git" || handle_failure "pipeline" "preflight"
require_cmd "pipeline" "preflight" "curl" || handle_failure "pipeline" "preflight"
require_cmd "pipeline" "preflight" "python3" || handle_failure "pipeline" "preflight"

if [[ "${SKIP_BOOTSTRAP}" -eq 0 ]]; then
  if ! run_logged "pipeline" "bootstrap" "cd \"${REPO_ROOT}\" && bash ./scripts/bootstrap_mlperf6_h200.sh \"${ENV_FILE}\""; then
    handle_failure "pipeline" "bootstrap"
  fi
else
  record_status "pipeline" "bootstrap" "skipped" "Bootstrap skipped by user"
fi

if [[ "${SKIP_DOWNLOADS}" -eq 0 ]]; then
  for benchmark in "${BENCHMARKS[@]}"; do
    case "${benchmark}" in
      llama31)
        download_llama31 || handle_failure "${benchmark}" "download"
        ;;
      llama2_lora)
        download_llama2_lora || handle_failure "${benchmark}" "download"
        ;;
      gpt_oss20b)
        download_gpt_oss20b || handle_failure "${benchmark}" "download"
        ;;
      flux)
        download_flux || handle_failure "${benchmark}" "download"
        ;;
      *)
        record_status "${benchmark}" "download" "failed" "Unknown benchmark"
        handle_failure "${benchmark}" "download"
        ;;
    esac
  done
else
  record_status "pipeline" "download" "skipped" "Downloads skipped by user"
fi

if [[ "${SKIP_RUNS}" -eq 0 ]]; then
  for benchmark in "${BENCHMARKS[@]}"; do
    run_benchmark "${benchmark}" || handle_failure "${benchmark}" "run"
  done
else
  record_status "pipeline" "run" "skipped" "Benchmark execution skipped by user"
fi

log "Pipeline completed. Report will be generated by the exit handler."

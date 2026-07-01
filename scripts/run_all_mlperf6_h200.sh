#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/configs/mlperf6-h200-4gpu.env"
# Default to hardware-perf (quick-run) mode: each benchmark runs a short
# time-boxed window and reports throughput, which is the purpose of this harness.
# Pass --full to do a real convergence run instead. Honor a pre-set env value.
export MLPERF_QUICK_RUN="${MLPERF_QUICK_RUN:-1}"
REPORT_OUTPUT=""
CONTINUE_ON_FAILURE=1
SKIP_BOOTSTRAP=0
SKIP_DOWNLOADS=0
SKIP_RUNS=0
# Default: run ALL four benchmarks. Order is deliberate:
#   - llama31 before gpt_oss20b: gpt_oss20b reuses llama31's downloaded C4 corpus
#   - flux LAST: it has by far the largest download (~2.23 TB), so the cheaper
#     benchmarks finish first and the big pull happens once everything else is done
BENCHMARKS_CSV="llama31,gpt_oss20b,llama2_lora,flux"

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
  --quick-run              Hardware-perf mode (DEFAULT): time-box each benchmark to
                           a short window (eval disabled) and report
                           throughput/step-time. NOT a valid MLPerf submission.
  --quick-run-seconds N    Per-benchmark perf window in seconds (default 300).
  --full                   Real convergence run (no time-box) instead of the
                           default hardware-perf mode. Use for a submission-style
                           run; takes hours per benchmark.
  -h, --help               Show this help

Behavior:
  - bootstraps the official mlcommons/training repo
  - for each benchmark in order: downloads its data, then runs it, before
    moving on to the next benchmark (download-one, run-one, repeat)
  - downloads public benchmark assets where official public URIs exist
  - reuses already-downloaded assets (completion markers, gpt_oss20b reuses the
    llama31 C4 corpus) and never deletes downloaded data
  - uses the gated upstream Llama2 downloader when MLPERF_LLAMA2_RCLONE_CONFIG is set
  - skips a benchmark's run when its own download stage failed
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
    --quick-run)
      export MLPERF_QUICK_RUN=1
      shift
      ;;
    --quick-run-seconds)
      export MLPERF_QUICK_RUN=1
      export MLPERF_QUICK_RUN_SECONDS="$2"
      shift 2
      ;;
    --full)
      export MLPERF_QUICK_RUN=0
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
  mkdir -p -- "$1" || return 1
}

# Resolve a path to its canonical absolute form, following symlinks when the
# target exists. Falls back to the literal path when no resolver is available so
# callers always get a non-empty string to compare against.
resolve_path() {
  local target="$1"
  if command -v realpath > /dev/null 2>&1; then
    realpath -- "${target}" 2>/dev/null && return 0
  fi
  if command -v readlink > /dev/null 2>&1; then
    readlink -f -- "${target}" 2>/dev/null && return 0
  fi
  printf '%s\n' "${target}"
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

  ensure_dir "$(dirname "${destination}")" || return 1
  local command_text
  # The MLCommons R2 downloader fetches every shard with `wget --continue`, then
  # verifies them with `md5sum -c`. On a flaky link a shard can land truncated or
  # corrupt at its full expected size; wget then treats it as complete and SKIPS
  # it on every re-run, so md5 verification fails forever and a plain retry never
  # converges (this is exactly how flux/cc12m-preprocessed failed: "Download
  # completed successfully!" followed by hundreds of "data-*.arrow: FAILED").
  # Wrap the download in a repair loop: on failure, parse the copied .md5 file,
  # delete only the shards that failed verification, and re-invoke -- wget then
  # re-fetches the now-missing shards and skips the good ones, so the dataset
  # completes across attempts. Only mark the download complete on full success.
  command_text=$(cat <<EOF
set -euo pipefail
mkdir -p "$(dirname "${destination}")"
cd "$(dirname "${destination}")"
r2_attempts=${MLPERF_R2_DOWNLOAD_RETRIES:-5}
r2_ok=0
for r2_attempt in \$(seq 1 "\${r2_attempts}"); do
  if bash <(curl -fsSL "${MLPERF_R2_DOWNLOADER_URL}") -d "$(basename "${destination}")" "${uri}"; then
    r2_ok=1
    break
  fi
  echo "r2 download/verify attempt \${r2_attempt}/\${r2_attempts} failed for $(basename "${destination}"); pruning corrupt shards before retry"
  md5_file=\$(ls "${destination}"/*.md5 2>/dev/null | head -n1 || true)
  if [[ -n "\${md5_file}" && -f "\${md5_file}" ]]; then
    ( cd "${destination}" && md5sum -c "\$(basename "\${md5_file}")" 2>/dev/null \\
        | awk -F': ' '/: FAILED\$/{print \$1}' \\
        | while IFS= read -r bad; do [[ -n "\${bad}" ]] && rm -f -- "\${bad}"; done ) || true
  fi
done
if [[ "\${r2_ok}" -ne 1 ]]; then
  echo "ERROR: r2 download for $(basename "${destination}") did not complete after \${r2_attempts} attempts" >&2
  exit 1
fi
touch "${completion_marker}"
EOF
)
  run_logged "${benchmark}" "download-${name}" "${command_text}"
}

sync_directory() {
  local source_dir="$1"
  local destination_dir="$2"

  # Fail closed on a missing/non-directory source.
  if [[ ! -d "${source_dir}" ]]; then
    log "sync_directory: source is not a directory: ${source_dir}"
    return 1
  fi
  # Never operate on an empty destination or the filesystem root.
  if [[ -z "${destination_dir}" || "${destination_dir}" == "/" ]]; then
    log "sync_directory: refusing unsafe destination: '${destination_dir}'"
    return 1
  fi

  # Reject identical or self-recursive source/destination so an additive copy can
  # never fold a directory into itself.
  local source_resolved="" dest_resolved=""
  source_resolved="$(resolve_path "${source_dir}")"
  dest_resolved="$(resolve_path "${destination_dir}")"
  if [[ -n "${source_resolved}" && -n "${dest_resolved}" ]]; then
    if [[ "${source_resolved}" == "${dest_resolved}" ]]; then
      log "sync_directory: source and destination are the same path: ${source_resolved}"
      return 1
    fi
    case "${dest_resolved}/" in
      "${source_resolved}/"*)
        log "sync_directory: destination ${dest_resolved} is inside source ${source_resolved}"
        return 1
        ;;
    esac
    case "${source_resolved}/" in
      "${dest_resolved}/"*)
        log "sync_directory: source ${source_resolved} is inside destination ${dest_resolved}"
        return 1
        ;;
    esac
  fi

  mkdir -p -- "${destination_dir}" || return 1
  # Additive only: never delete existing files so prior downloads stay reusable.
  # No rm -rf, no rsync --delete -- downloaded data is always preserved.
  if command -v rsync > /dev/null 2>&1; then
    rsync -a -- "${source_dir}/" "${destination_dir}/" || return 1
  else
    cp -a -- "${source_dir}/." "${destination_dir}/" || return 1
  fi
}

download_llama31() {
  download_r2_named "llama31" "dataset" "${MLPERF_LLAMA31_PREPROCESSED_PATH}" "${MLPERF_LLAMA31_DATASET_URI}" || return 1
  download_r2_named "llama31" "tokenizer" "${MLPERF_LLAMA31_TOKENIZER_PATH}" "${MLPERF_LLAMA31_TOKENIZER_URI}" || return 1
}

# Pre-stage the public model + GovReport dataset to the LOCAL paths the runner
# reads (so the run resolves to 'official' and reads them in place -- no 130 GB
# download inside the 5-minute container window). Runs in the llama2 image as the
# INVOKING USER so writes land on the root-squashed lustre mount. Triggered when
# HF_TOKEN is set (i.e. you ran `huggingface-cli login`). Idempotent:
# huggingface-cli download resumes/skips, the dataset prep is skipped if present.
prestage_llama2_from_hf() {
  local data_local="$1" model_local="$2"
  require_cmd "llama2_lora" "download" "docker" || return 1
  local img="${MLPERF_LLAMA2_DOCKER_IMAGE}" uid gid
  uid="$(id -u)"; gid="$(id -g)"
  local command_text
  command_text=$(cat <<EOF
set -euo pipefail
mkdir -p "${model_local}" "${data_local}"
docker pull "${img}" >/dev/null 2>&1 || true
docker run --rm --user ${uid}:${gid} \\
  -e HF_TOKEN="\${HF_TOKEN:-}" \\
  -v "${model_local}:/model_out" \\
  -v "${data_local}:/data_out" \\
  -v "${REPO_ROOT}:/repo:ro" \\
  "${img}" bash -lc '
set -euo pipefail
export HOME=/tmp HF_HOME=/tmp/hf_home HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="/tmp/.local/bin:\$PATH"
python3 -m pip install --user -q "huggingface_hub[cli]" "datasets>=2,<3" transformers sentencepiece >/dev/null 2>&1 || true
echo "Staging model ${MLPERF_LLAMA2_PUBLIC_MODEL_ID} -> local model dir (resumes/skips existing)"
# Newer huggingface_hub removed the huggingface-cli entrypoint in favor of hf
# ("huggingface-cli is deprecated and no longer works"). Prefer hf, fall back to
# huggingface-cli for older hubs.
HFC=hf; command -v hf >/dev/null 2>&1 || HFC=huggingface-cli
# The 70B model is ~130 GB; the writer can die mid-download. Each attempt resumes
# and only fetches missing shards, so retry; hard-fail if none succeed so the
# caller does not treat an empty model dir as a completed prestage.
prestage_ok=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if "\${HFC}" download "${MLPERF_LLAMA2_PUBLIC_MODEL_ID}" --local-dir /model_out --max-workers 4; then prestage_ok=1; break; fi
  echo "prestage model download attempt \${attempt} failed; resuming and retrying"
  sleep 5
done
if [ "\${prestage_ok}" -ne 1 ]; then echo "ERROR: prestage model download failed after retries" >&2; exit 1; fi
if [ ! -f /data_out/train-00000-of-00001.parquet ]; then
  echo "Staging GovReport dataset -> local dataset dir"
  python3 /repo/scripts/prepare_llama2_lora_smoke_dataset.py --dataset-name "${MLPERF_LLAMA2_SMOKE_DATASET_NAME}" --dataset-config "${MLPERF_LLAMA2_SMOKE_DATASET_CONFIG}" --output-dir /data_out --tokenizer-path /model_out --block-size 8192 --train-samples "\${MLPERF_LLAMA2_PRESTAGE_TRAIN_SAMPLES:-8000}" --validation-samples "\${MLPERF_LLAMA2_PRESTAGE_VAL_SAMPLES:-970}"
fi
echo "llama2 prestage complete"
'
EOF
)
  run_logged "llama2_lora" "download" "${command_text}"
}

download_llama2_lora() {
  local data_local="${MLPERF_LLAMA2_DATASET_PATH}/${MLPERF_LLAMA2_LOCAL_DATASET_SUBDIR}"
  # Stage the public HF model under the SAME dirname the in-container training
  # reads for non-official modes (LLAMA2_MODEL_SUBDIR = PUBLIC_MODEL_DIRNAME), so
  # the training container finds it at /models/<dir> and skips the /tmp download.
  local model_local="${MLPERF_LLAMA2_MODEL_ROOT}/${MLPERF_LLAMA2_PUBLIC_MODEL_DIRNAME}"
  # Already staged (dataset parquet + a non-empty model dir) -> nothing to do.
  if [[ -f "${data_local}/train-00000-of-00001.parquet" && -d "${model_local}" && -n "$(ls -A "${model_local}" 2>/dev/null)" ]]; then
    record_status "llama2_lora" "download" "skipped" "dataset+model already staged at ${data_local} and ${model_local}"
    return 0
  fi
  # With an HF login, auto-stage from HuggingFace to the local official paths.
  if [[ -n "${HF_TOKEN:-}" ]]; then
    if prestage_llama2_from_hf "${data_local}" "${model_local}"; then
      return 0
    fi
    log "llama2_lora HF prestage failed; falling back to mode-based handling"
  fi
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
  # gpt_oss20b consumes the same preprocessed C4 corpus as llama31. When that
  # corpus is already downloaded and the URIs match, reuse it via a symlink
  # instead of downloading ~79 GB a second time. Non-destructive and reversible.
  local data_path="${MLPERF_GPT_OSS_DATA_PATH}"
  local llama31_path="${MLPERF_LLAMA31_PREPROCESSED_PATH}"
  local llama31_marker="${llama31_path}/.mlperf-download-complete"
  local uris_match=0
  if [[ "${MLPERF_GPT_OSS_DATASET_URI}" == "${MLPERF_LLAMA31_DATASET_URI}" ]]; then
    uris_match=1
  fi

  if [[ -L "${data_path}" ]]; then
    # An existing symlink is only valid reuse when the URIs match, it resolves to
    # the llama31 preprocessed corpus, and that corpus finished downloading.
    # Anything else is stale/invalid: fail closed rather than feed the wrong
    # dataset into the benchmark.
    local link_resolved="" llama31_resolved=""
    link_resolved="$(resolve_path "${data_path}")"
    llama31_resolved="$(resolve_path "${llama31_path}")"
    if [[ "${uris_match}" -eq 1 \
          && -n "${link_resolved}" \
          && "${link_resolved}" == "${llama31_resolved}" \
          && -f "${llama31_marker}" ]]; then
      record_status "gpt_oss20b" "download-dataset" "skipped" "Reused llama31 preprocessed C4 via existing symlink"
    else
      record_status "gpt_oss20b" "download-dataset" "failed" "Stale/invalid symlink at ${data_path}; expected a symlink to ${llama31_path} with a matching dataset URI and a completed llama31 download. Remove the symlink and re-run."
      log "gpt_oss20b dataset symlink invalid: ${data_path} -> ${link_resolved:-<unresolved>}"
      return 1
    fi
  elif [[ ! -e "${data_path}" ]]; then
    # Nothing present yet. Reuse llama31's corpus when the URIs match and its
    # download completed; otherwise download gpt_oss20b's own copy.
    if [[ "${uris_match}" -eq 1 && -f "${llama31_marker}" ]]; then
      if ! ensure_dir "$(dirname "${data_path}")"; then
        record_status "gpt_oss20b" "download-dataset" "failed" "Failed to create parent dir for ${data_path}"
        log "gpt_oss20b dataset reuse failed: cannot create $(dirname "${data_path}")"
        return 1
      fi
      if ln -sfn "${llama31_path}" "${data_path}"; then
        record_status "gpt_oss20b" "download-dataset" "skipped" "Reused llama31 preprocessed C4 via symlink"
      else
        record_status "gpt_oss20b" "download-dataset" "failed" "Failed to create reuse symlink ${data_path} -> ${llama31_path}"
        log "gpt_oss20b dataset reuse symlink creation failed"
        return 1
      fi
    else
      download_r2_named "gpt_oss20b" "dataset" "${data_path}" "${MLPERF_GPT_OSS_DATASET_URI}" || return 1
    fi
  else
    # A real directory/file already exists. Preserve reusable behavior:
    # download_r2_named skips when its completion marker is present, otherwise it
    # resumes/fills in place. Never delete what is already on disk.
    download_r2_named "gpt_oss20b" "dataset" "${data_path}" "${MLPERF_GPT_OSS_DATASET_URI}" || return 1
  fi

  if [[ ! -d "${MLPERF_GPT_OSS_TOKENIZER_SOURCE}" ]]; then
    record_status "gpt_oss20b" "tokenizer" "failed" "Tokenizer source missing: ${MLPERF_GPT_OSS_TOKENIZER_SOURCE}"
    return 1
  fi

  if ! sync_directory "${MLPERF_GPT_OSS_TOKENIZER_SOURCE}" "${MLPERF_GPT_OSS_MODEL_PATH}"; then
    record_status "gpt_oss20b" "tokenizer" "failed" "Failed to sync tokenizer from ${MLPERF_GPT_OSS_TOKENIZER_SOURCE} to ${MLPERF_GPT_OSS_MODEL_PATH}"
    log "gpt_oss20b tokenizer sync failed"
    return 1
  fi
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

download_benchmark() {
  local benchmark="${1:-}"
  if [[ -z "${benchmark}" ]]; then
    record_status "unknown" "download" "failed" "Empty benchmark name passed to download_benchmark"
    log "download_benchmark called with an empty benchmark name"
    return 1
  fi
  case "${benchmark}" in
    llama31)     download_llama31 ;;
    llama2_lora) download_llama2_lora ;;
    gpt_oss20b)  download_gpt_oss20b ;;
    flux)        download_flux ;;
    *)
      record_status "${benchmark}" "download" "failed" "Unknown benchmark"
      log "download_benchmark called with an unknown benchmark: ${benchmark}"
      return 1
      ;;
  esac
}

# Process one benchmark fully before starting the next: download its data, then
# run it. This keeps active downloading/running scoped to a single benchmark at a
# time and lets results land incrementally instead of after every dataset is
# fetched. Note: persistent disk usage still accumulates across benchmarks --
# downloaded data is retained for reuse and never deleted, so this bounds the
# active working set, not the total on-disk footprint.
for benchmark in "${BENCHMARKS[@]}"; do
  log "=== Benchmark ${benchmark}: download then run ==="

  download_ok=1
  if [[ "${SKIP_DOWNLOADS}" -eq 0 ]]; then
    if ! download_benchmark "${benchmark}"; then
      download_ok=0
      handle_failure "${benchmark}" "download"
    fi
  else
    record_status "${benchmark}" "download" "skipped" "Downloads skipped by user"
  fi

  if [[ "${SKIP_RUNS}" -eq 1 ]]; then
    record_status "${benchmark}" "run" "skipped" "Benchmark execution skipped by user"
    continue
  fi

  if [[ "${download_ok}" -eq 0 ]]; then
    record_status "${benchmark}" "run" "skipped" "Download stage failed"
    log "Skipping ${benchmark}/run because its download stage failed"
    continue
  fi

  run_benchmark "${benchmark}" || handle_failure "${benchmark}" "run"

  # Regenerate the report after each benchmark so a completed benchmark's result
  # is captured even if a later benchmark (or the whole pipeline) fails. The
  # final exit-handler report is still written at the end.
  generate_report || log "Incremental report after ${benchmark} failed (continuing)"
done

log "Pipeline completed. Report will be generated by the exit handler."

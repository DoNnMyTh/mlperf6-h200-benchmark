#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/configs/mlperf6-h200-4gpu.env"
EXECUTE=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_mlperf6_h200.sh [--env-file PATH] show <benchmark|all>
  ./scripts/run_mlperf6_h200.sh [--env-file PATH] [--execute] run <benchmark|all>
  ./scripts/run_mlperf6_h200.sh [--env-file PATH] report [OUTPUT_PATH]

Benchmarks:
  llama31        Small LLM pretraining (Llama 3.1 8B)
  llama2_lora    Llama 2 70B LoRA fine-tuning
  gpt_oss20b     Small LLM MoE pretraining
  flux           Text-to-image generation
  all            All benchmark command plans

Options:
  --quick-run            Hardware-perf mode: time-box the benchmark to a short
                         sustained window (eval off) instead of full convergence.
  --quick-run-seconds N  Perf window seconds (default 300). Implies --quick-run.

Notes:
  - `show` prints the exact commands without running them.
  - `run` requires `--execute`; without it, the script refuses to launch work.
  - `report` writes a markdown summary using the current results directories.
  - quick-run yields throughput/step-time numbers, NOT a valid submission score.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --execute)
      EXECUTE=1
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: env file not found: ${ENV_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

# --- quick-run (hardware-perf) mode ----------------------------------------
# When MLPERF_QUICK_RUN=1 each benchmark runs a time-boxed PERF window instead of
# a full convergence run: eval is disabled, the step count is raised to a
# generous ceiling, and the training launch is wrapped in `timeout` so it stops
# after MLPERF_QUICK_RUN_SECONDS (default 300 = 5 min per benchmark). A timeout exit
# (124) is treated as success -- the run captured a throughput/step-time sample,
# not convergence. This yields hardware-performance numbers; it is NOT a valid
# MLPerf submission result (no target-loss convergence).
MLPERF_QUICK_RUN="${MLPERF_QUICK_RUN:-0}"
MLPERF_QUICK_RUN_SECONDS="${MLPERF_QUICK_RUN_SECONDS:-300}"
# Step ceilings are intentionally high: with a sustained window the timeout is
# the real bound, so these only stop a benchmark that is somehow faster than the
# window. Override per benchmark via env if a step bound is preferred.
MLPERF_QUICK_LLAMA31_STEPS="${MLPERF_QUICK_LLAMA31_STEPS:-1000000}"
MLPERF_QUICK_GPT_OSS_ITERS="${MLPERF_QUICK_GPT_OSS_ITERS:-1000000}"
MLPERF_QUICK_LLAMA2_STEPS="${MLPERF_QUICK_LLAMA2_STEPS:-1000000}"
MLPERF_QUICK_FLUX_STEPS="${MLPERF_QUICK_FLUX_STEPS:-1000000}"
# A large value that effectively disables periodic eval within the window. Use a
# power of two (2^30): llama31 derives val_check_interval = (eval_every / GBS) /
# GBS, which lightning requires to be an int or a whole float -- 2^30 stays whole
# for a power-of-two GBS (e.g. 32 -> 1048576.0), whereas 999999999 produced the
# fractional 976562.5 that lightning rejected. flux/gpt_oss only need "large".
MLPERF_QUICK_EVAL_DISABLE="${MLPERF_QUICK_EVAL_DISABLE:-1073741824}"

# Render-time helper: prefix for a quick-run launch. Rather than wrap the launch
# in `timeout` (whose SIGTERM->SIGKILL escalation hits the `docker run` CLI but
# leaves the CONTAINER running, holding the GPUs), start a background WATCHDOG
# that force-removes the container by name after exactly the perf window. Removing
# the container makes the foreground `docker run` exit immediately and frees the
# GPUs deterministically at the window, regardless of in-container signal
# handling. Pre-clean any stale same-named container first. Pass the name.
quick_timeout_prefix() {
  [[ "${MLPERF_QUICK_RUN}" == "1" ]] || return 0
  local cname="${1:-}"
  [[ -n "${cname}" ]] || return 0
  printf 'docker rm -f %s >/dev/null 2>&1 || true\n' "${cname}"
  printf '( sleep %s; echo "quick-run: %ss window reached, removing %s"; docker rm -f %s >/dev/null 2>&1 || true ) &\n' \
    "${MLPERF_QUICK_RUN_SECONDS}" "${MLPERF_QUICK_RUN_SECONDS}" "${cname}" "${cname}"
  # End with "; " (not a newline -- $() strips trailing newlines) so the caller's
  # `docker run`/`bash run_with_docker.sh` is separated from this assignment.
  printf 'QUICK_WATCHDOG=$! ; '
}
# Render-time helper: trailing handler for the quick-run launch. Capture the exit
# code, stop the watchdog, force-remove the container (belt-and-suspenders), and
# treat a watchdog/timeout kill as success. When the watchdog removes the
# container the foreground launch exits 137 (SIGKILL); a real crash exits 1/other.
quick_timeout_suffix() {
  [[ "${MLPERF_QUICK_RUN}" == "1" ]] || return 0
  local cname="${1:-}"
  printf ' || quick_ec=$?\n'
  printf 'kill "${QUICK_WATCHDOG}" 2>/dev/null || true\n'
  [[ -n "${cname}" ]] && printf 'docker rm -f %s >/dev/null 2>&1 || true\n' "${cname}"
  cat <<'SUF'
case "${quick_ec:-0}" in
  0|124|137|143) echo "quick-run: perf window reached (container stopped); treating as success" ;;
  *) exit "${quick_ec}" ;;
esac
SUF
}

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

ensure_upstream() {
  if [[ ! -d "${MLPERF_UPSTREAM_DIR}/.git" ]]; then
    echo "ERROR: MLCommons training repo not found at ${MLPERF_UPSTREAM_DIR}" >&2
    echo "Run ./scripts/bootstrap_mlperf6_h200.sh first." >&2
    exit 1
  fi
}

run_or_print() {
  local title="$1"
  local script_body="$2"

  echo "### ${title}"
  if [[ "${EXECUTE}" -eq 1 ]]; then
    bash -lc "${script_body}"
  else
    printf '%s\n' "${script_body}"
  fi
}

render_llama31() {
  # In quick-run, raise MAX_STEPS to a high ceiling and push eval out past the
  # window (the timeout bounds wall-clock). Emitted into the in-container block;
  # empty (a blank line) outside quick-run.
  local llama31_quick_exports=""
  if [[ "${MLPERF_QUICK_RUN}" == "1" ]]; then
    llama31_quick_exports="export MAX_STEPS=${MLPERF_QUICK_LLAMA31_STEPS}
export EVAL_EVERY=${MLPERF_QUICK_EVAL_DISABLE}
export START_EVAL_AT=${MLPERF_QUICK_EVAL_DISABLE}"
  fi
  cat <<EOF
set -euo pipefail
mkdir -p "${MLPERF_LLAMA31_RESULTS_PATH}" "${MLPERF_LLAMA31_CHECKPOINT_PATH}" "${MLPERF_LLAMA31_INDEX_PATH}"
cd "${MLPERF_UPSTREAM_DIR}/small_llm_pretraining/nemo"
# pretrain_llama31.py imports the full NeMo stack at module top -- wandb,
# lightning, and (lines 23-31) nemo.collections, nemo.lightning, nemo_run plus a
# local callbacks module. Without --run_slurm it builds a run.LocalExecutor()
# and launches training via torchrun, i.e. it trains IN-PROCESS in whatever
# environment runs the launcher. The bare host has none of that stack (the run
# failed first at 'import wandb', then 'import lightning', with nemo next), and
# installing NeMo + TransformerEngine + Megatron on the host is not viable. The
# built image already contains the entire stack, so run the launcher INSIDE the
# image with the GPUs attached. LocalExecutor spawns no nested container in local
# mode, so no docker socket is needed. Keep the idempotent Dockerfile wandb
# install so the in-image interpreter has wandb; WANDB_MODE=offline needs no key.
if ! grep -q 'pip install wandb' Dockerfile.h200; then
  printf '\nRUN pip install --no-cache-dir wandb\nENV WANDB_MODE=offline\n' >> Dockerfile.h200
  echo "Added wandb install + offline mode to Dockerfile.h200"
fi
docker build -t "${MLPERF_LLAMA31_IMAGE}" -f Dockerfile.h200 .
# Run the launcher inside the freshly built image. Every benchmark path lives
# under /scratch on this node, so a single bind mount makes all the absolute host
# paths below resolve identically inside the container; mount the results dir at
# /mlperf-outputs too so MLLog output persists past the --rm container.
$(quick_timeout_prefix mlperf-llama31)docker run --rm --name mlperf-llama31 --gpus all --ipc=host --network host \\
  --ulimit memlock=-1 --ulimit stack=67108864 \\
  -v /scratch:/scratch \\
  -v "${MLPERF_LLAMA31_RESULTS_PATH}:/mlperf-outputs" \\
  -w "${MLPERF_UPSTREAM_DIR}/small_llm_pretraining/nemo" \\
  "${MLPERF_LLAMA31_IMAGE}" \\
  bash -lc '
set -euo pipefail
source ./config_H200_1x8x1_8b.sh
export WANDB_MODE="\${WANDB_MODE:-offline}"
export USER="\${USER:-local}"
export HOST="\${HOST:-local}"
export ACCOUNT="\${ACCOUNT:-local}"
export PARTITION="\${PARTITION:-local}"
export REMOTE=0
export FROM_HF=0
export IMAGE="${MLPERF_LLAMA31_IMAGE}"
export GPUS_PER_NODE="${MLPERF_GPU_COUNT}"
export NNODES=1
export JOB_DIR="${MLPERF_LLAMA31_RESULTS_PATH}"
export PREPROCESSED_PATH="${MLPERF_LLAMA31_PREPROCESSED_PATH}"
export TOKENIZER_PATH="${MLPERF_LLAMA31_TOKENIZER_PATH}"
export CONTINUAL_CKPT="${MLPERF_LLAMA31_CHECKPOINT_PATH}"
export TMP_NPY_INDEX="${MLPERF_LLAMA31_INDEX_PATH}"
export GBS="${MLPERF_LLAMA31_GBS}"
export MBS="${MLPERF_LLAMA31_MBS}"
# The 8B run sits near the 140 GiB H200 limit (~130 GiB observed); reduce
# allocator fragmentation to lower the OOM risk.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${llama31_quick_exports}
# In-container we are root, so the upstream "mkdir /mlperf-outputs" succeeds, but
# keep the guard non-fatal for safety (the dir is also bind-mounted above).
sed -i "s@mkdir /mlperf-outputs; fi@mkdir -p /mlperf-outputs 2>/dev/null || true; fi@" run_llama31.sh
# pretrain_llama31.py runs the experiment with detach=True, so nemo_run launches
# the training job with log=False: the console shows only "Waiting for job ...
# [log=False]" while the GPUs are saturated, which looks hung. Flip it to
# detach=False so the job logs stream to stdout (and into the orchestrator log
# and the false-success grep below). Idempotent.
sed -i "s@exp.run(sequential=True, detach=True)@exp.run(sequential=True, detach=False)@" pretrain_llama31.py
# nemo_run LocalExecutor runs the training job with log=False: it writes the
# real per-step logs (global_batch_size / train_step_timing -> the throughput we
# need) into its experiment dir under /root/.nemo_run/.../-steps/, NOT to stdout.
# So the orchestrator stage log (and the report) never saw them, and they were
# lost when the --rm container exited. Stream that job log to stdout in the
# BACKGROUND: wait for the -steps job log to appear, then follow it with tail -F.
# The host tees our stdout to orchestration/llama31-run.log, so the throughput is
# captured persistently even when the quick-run timeout later kills the container.
( TAILED=" "; for _ in \$(seq 1 720); do
    for f in \$(find /root/.nemo_run/experiments -type f -path "*-steps/*" 2>/dev/null); do
      case "\${TAILED}" in
        *" \${f} "*) : ;;
        *) TAILED="\${TAILED}\${f} "; echo "[stream] following \${f}"; tail -n +1 -F "\${f}" 2>/dev/null & ;;
      esac
    done
    sleep 5
  done ) &
LLAMA31_TAILER=\$!
set +e
bash ./run_llama31.sh 2>&1 | tee /tmp/llama31_console.log
run_rc=\${PIPESTATUS[0]}
set -e
kill "\${LLAMA31_TAILER}" 2>/dev/null || true
echo "===== nemo_run experiment logs (tail) ====="
find /root/.nemo_run -type f \\( -name "*.log" -o -path "*-steps/*" \\) 2>/dev/null | while read -r f; do echo "--- \$f ---"; tail -n 200 "\$f"; done || true
if grep -qE "finished: FAILED|-steps FAILED|Task [0-9].* FAILED" /tmp/llama31_console.log; then
  echo "ERROR: llama31 nemo_run experiment reported FAILED (LocalExecutor exited 0 but the task failed)"
  exit 1
fi
[ "\${run_rc}" -eq 0 ] || exit "\${run_rc}"
'$(quick_timeout_suffix mlperf-llama31)
EOF
}

render_llama2_lora() {
  # Step/eval bounds. Full run keeps the smoke-test cap (1024); quick-run raises
  # the cap (timeout bounds wall-clock) and disables periodic eval.
  local llama2_max_steps="1024"
  local llama2_eval_steps="48"
  if [[ "${MLPERF_QUICK_RUN}" == "1" ]]; then
    llama2_max_steps="${MLPERF_QUICK_LLAMA2_STEPS}"
    llama2_eval_steps="${MLPERF_QUICK_EVAL_DISABLE}"
  fi
  cat <<EOF
set -euo pipefail
mkdir -p "${MLPERF_LLAMA2_RESULTS_PATH}"
cd "${MLPERF_UPSTREAM_DIR}/llama2_70b_lora"
cat > configs/h200_4gpu.yaml <<'YAML'
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_clipping: 0.3
  gradient_accumulation_steps: 1
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML
case "${EFFECTIVE_MLPERF_LLAMA2_MODE}" in
  official)
    LLAMA2_DATASET_SUBDIR="${MLPERF_LLAMA2_LOCAL_DATASET_SUBDIR}"
    LLAMA2_MODEL_SUBDIR="${MLPERF_LLAMA2_LOCAL_MODEL_SUBDIR}"
    LLAMA2_MODE_NOTE="Official MLPerf assets"
    ;;
  local-only)
    LLAMA2_DATASET_SUBDIR="${MLPERF_LLAMA2_LOCAL_DATASET_SUBDIR}"
    LLAMA2_MODEL_SUBDIR="${MLPERF_LLAMA2_PUBLIC_MODEL_DIRNAME}"
    LLAMA2_MODE_NOTE="Local dataset plus public HF model"
    ;;
  smoke-test)
    LLAMA2_DATASET_SUBDIR="${MLPERF_LLAMA2_SMOKE_DATASET_SUBDIR}"
    LLAMA2_MODEL_SUBDIR="${MLPERF_LLAMA2_PUBLIC_MODEL_DIRNAME}"
    LLAMA2_MODE_NOTE="Non-MLPerf smoke test with public dataset/model"
    ;;
  skip)
    echo "Skipping llama2_lora because effective mode is skip"
    exit 0
    ;;
  *)
    echo "ERROR: unsupported effective llama2_lora mode: ${EFFECTIVE_MLPERF_LLAMA2_MODE}" >&2
    exit 1
    ;;
esac

echo "Llama2 LoRA mode: \${LLAMA2_MODE_NOTE}"
if [[ "${EFFECTIVE_MLPERF_LLAMA2_MODE}" == "smoke-test" ]]; then
  echo "WARNING: smoke-test mode is not valid for MLPerf submission."
fi

docker pull "${MLPERF_LLAMA2_DOCKER_IMAGE}"
$(quick_timeout_prefix mlperf-llama2)docker run --rm --name mlperf-llama2 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
  -e HF_TOKEN="\${HF_TOKEN:-}" \\
  -e MLPERF_LLAMA2_MODE="${EFFECTIVE_MLPERF_LLAMA2_MODE}" \\
  -e MLPERF_LLAMA2_PUBLIC_MODEL_ID="${MLPERF_LLAMA2_PUBLIC_MODEL_ID}" \\
  -e MLPERF_LLAMA2_PUBLIC_MODEL_DIRNAME="${MLPERF_LLAMA2_PUBLIC_MODEL_DIRNAME}" \\
  -e MLPERF_LLAMA2_SMOKE_DATASET_NAME="${MLPERF_LLAMA2_SMOKE_DATASET_NAME}" \\
  -e MLPERF_LLAMA2_SMOKE_DATASET_CONFIG="${MLPERF_LLAMA2_SMOKE_DATASET_CONFIG}" \\
  -e MLPERF_LLAMA2_SMOKE_TRAIN_SAMPLES="${MLPERF_LLAMA2_SMOKE_TRAIN_SAMPLES}" \\
  -e MLPERF_LLAMA2_SMOKE_VAL_SAMPLES="${MLPERF_LLAMA2_SMOKE_VAL_SAMPLES}" \\
  -e LLAMA2_DATASET_SUBDIR="\${LLAMA2_DATASET_SUBDIR}" \\
  -e LLAMA2_MODEL_SUBDIR="\${LLAMA2_MODEL_SUBDIR}" \\
  -v "${MLPERF_UPSTREAM_DIR}/llama2_70b_lora:/workspace" \\
  -v "${MLPERF_LLAMA2_DATASET_PATH}:/workspace/dataset" \\
  -v "${MLPERF_LLAMA2_RESULTS_PATH}:/workspace/results" \\
  -v "${MLPERF_LLAMA2_MODEL_ROOT}:/models" \\
  -v "${MLPERF_HF_CACHE}:/root/.cache/huggingface" \\
  -v "${REPO_ROOT}:/repo" \\
  -w /workspace \\
  "${MLPERF_LLAMA2_DOCKER_IMAGE}" \\
  bash -lc '
set -euo pipefail
# Disable the hf-xet transfer backend: on this node it crashed mid-download with
# "Internal Writer Error: Background writer channel closed" while resuming the
# 70B model. Plain HTTPS download is slower but resumes reliably.
export HF_HUB_DISABLE_XET=1
# The accelerated hf_transfer (rust) writer intermittently dies on this node's
# lustre mount with "[Errno 14] Bad address" mid-download of the 70B shards.
# Disable it so writes go through plain Python http_get, which resumes reliably.
export HF_HUB_ENABLE_HF_TRANSFER=0
# huggingface caches downloads (datasets via load_dataset, model via
# huggingface-cli) under HF_HOME (default /root/.cache, bind-mounted from lustre
# and thus not writable by the squashed root). Point it at container-local /tmp
# so dataset prep and model download can cache.
export HF_HOME=/tmp/hf_home
echo "Resolved llama2_lora mode inside container: \${MLPERF_LLAMA2_MODE}"
# /workspace/dataset and /models are lustre bind mounts; the node root-squashes
# the container root user, so writes there fail with "Permission denied". Assets
# we GENERATE/DOWNLOAD must go to container-local /tmp; only official mode reads the
# mounted assets in place. Data dir is /tmp only when we prep it (non
# official/local); model dir is /tmp whenever we download it (non official).
LLAMA2_DATA_DIR="/workspace/dataset/\${LLAMA2_DATASET_SUBDIR}"
LLAMA2_MODEL_DIR="/models/\${LLAMA2_MODEL_SUBDIR}"
if [[ "\${MLPERF_LLAMA2_MODE}" != "official" ]]; then
  LLAMA2_MODEL_DIR="/tmp/llama2_model/\${LLAMA2_MODEL_SUBDIR}"
  if [[ "\${MLPERF_LLAMA2_MODE}" != "local-only" ]]; then
    LLAMA2_DATA_DIR="/tmp/llama2_data/\${LLAMA2_DATASET_SUBDIR}"
  fi
fi
echo "llama2 data dir: \${LLAMA2_DATA_DIR} | model dir: \${LLAMA2_MODEL_DIR}"
# pip reports "normal site-packages is not writeable" and installs console
# scripts (accelerate, huggingface-cli, ...) into the user site's bin, which is
# not on PATH -> "accelerate: command not found". Put it on PATH.
export PATH="\${HOME:-/root}/.local/bin:/root/.local/bin:\${PATH}"
pip install -r requirements.txt
# flash-attn: the upstream step force-builds 2.1.0 from source, but 2.1.0 (Sep
# 2023) predates CUDA 13 / Hopper sm_90 and will not compile on this H200 node
# (the build first died with "Python.h: No such file or directory", and even
# with headers the sm_90 / CUDA 13.2 compile is unsupported). Prefer a flash-attn
# already bundled in the container image -- nvcr/pytorch images ship one matched
# to the container torch+CUDA, which is what we actually want on Hopper. Only
# if none is importable, fall back to a source build: install the Python C dev
# headers for the running interpreter (python3.12-dev -> Python.h at
# /usr/include/python3.12/) and target sm_90.
if python3 -c "import flash_attn" >/dev/null 2>&1; then
  echo "Using flash-attn already present in image: \$(python3 -c "import flash_attn, sys; sys.stdout.write(flash_attn.__version__)")"
else
  echo "flash-attn not in image; building from source for sm_90"
  # This image keeps python under /usr/local (a source build), so the base distro
  # has no python3.12-dev package to apt-install -- that is why the previous
  # header step silently did nothing and the build still failed on Python.h.
  # Point the compiler at the interpreter OWN headers via python3-config (e.g.
  # /usr/local/include/python3.12, where this python keeps Python.h), and keep
  # apt as a backup for stock distro-python images.
  # Guard with || true: under set -euo pipefail a missing python3-config (or any
  # pipe failure here) would otherwise abort the whole llama2 run at this line.
  PY_INC=\$(python3-config --includes 2>/dev/null | sed "s/-I//g" || true)
  export CPATH="\${PY_INC// /:}:\${CPATH:-}"
  LLAMA2_PYV=\$(python3 -V 2>&1 | cut -d" " -f2 | cut -d. -f1,2)
  apt-get update -y >/dev/null 2>&1 \\
    && apt-get install -y --no-install-recommends python\${LLAMA2_PYV}-dev >/dev/null 2>&1 \\
    || apt-get install -y --no-install-recommends python3-dev >/dev/null 2>&1 || true
  # Non-fatal: this image has no Python.h anywhere (source-built /usr/local
  # python, no apt python3.12-dev), so the 2.1.0 build cannot succeed here. Let it
  # try, but do not abort the run -- smoke-test can fall back to no flash-attn.
  TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32 pip install flash-attn==2.1.0 --no-build-isolation \\
    || echo "flash-attn build failed; will run without it"
fi
# Use flash-attn only if it actually ended up importable. The HF trainer runs
# without it (standard SDPA attention) -- slower, fine for a smoke-test/perf run.
if python3 -c "import flash_attn" >/dev/null 2>&1; then
  FA_FLAG="--use_flash_attn"
else
  echo "WARNING: flash-attn unavailable; running llama2 smoke-test without it"
  FA_FLAG=""
fi
# Idempotent: a previous attempt in a reused image layer can leave this dir, so
# git clone fails with "destination path already exists". Remove it first.
rm -rf /tmp/mlperf-logging
git clone --depth 1 https://github.com/mlperf/logging.git /tmp/mlperf-logging
pip install -e /tmp/mlperf-logging
if [[ -n "\${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "\${HF_TOKEN}"
fi
if [[ "\${MLPERF_LLAMA2_MODE}" != "official" ]]; then
  # Download the public model for any non-official mode (local-only, smoke-test,
  # and the unresolved "auto" the container sometimes receives). Gating on the
  # exact strings skipped auto and failed with "model directory missing".
  # Always invoke the downloader. huggingface-cli download resumes and only
  # fetches missing files, so an earlier interrupted run that left an INCOMPLETE
  # model directory is repaired instead of being wrongly treated as complete
  # (the previous "dir exists -> skip" check produced a missing modeling_llama.py).
  # The 70B model is ~128 GB. On this node the writer intermittently dies mid
  # download (hf-xet "Background writer channel closed", then plain http_get
  # "[Errno 14] Bad address" writing to lustre). Each attempt resumes and only
  # fetches what is missing, so retry a few times with fewer parallel writers;
  # this completes the remaining shards across attempts.
  model_download_ok=0
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if huggingface-cli download "\${MLPERF_LLAMA2_PUBLIC_MODEL_ID}" --local-dir "\${LLAMA2_MODEL_DIR}" --max-workers 1; then
      model_download_ok=1
      break
    fi
    echo "llama2 model download attempt \${attempt} failed; resuming and retrying"
    sleep 5
  done
  if [[ "\${model_download_ok}" -ne 1 ]]; then
    echo "ERROR: llama2 model download did not complete after repeated attempts" >&2
    exit 1
  fi
  for required in config.json modeling_llama.py model.safetensors.index.json; do
    if [[ ! -f "\${LLAMA2_MODEL_DIR}/\${required}" ]]; then
      echo "ERROR: required model file still missing after download: \${LLAMA2_MODEL_DIR}/\${required}" >&2
      exit 1
    fi
  done
fi
# Prepare the public smoke dataset whenever the parquet is absent and we are not
# using official/local assets. Gating on exactly "smoke-test" was fragile: the
# container sometimes received the unresolved mode "auto" (so this was skipped
# and the run failed with "dataset parquet missing"). Keying off the missing
# parquet + non-official/local mode covers smoke-test, auto, and empty.
if [[ ! -f "\${LLAMA2_DATA_DIR}/train-00000-of-00001.parquet" \\
      && "\${MLPERF_LLAMA2_MODE}" != "official" \\
      && "\${MLPERF_LLAMA2_MODE}" != "local-only" ]]; then
  echo "Preparing public smoke dataset (parquet missing, mode=\${MLPERF_LLAMA2_MODE})"
  # Inline the prep so it never depends on a /repo helper file being present on
  # the host (a partial checkout left /repo/scripts/prepare_..._smoke_dataset.py
  # missing). Reads the smoke params from the env already forwarded into the
  # container. No single quotes and no dollar signs so it survives bash -lc.
  cat > /tmp/prep_llama2_smoke.py <<PYEOF
import sys
from datasets import load_dataset
name, config, out = sys.argv[1], sys.argv[2], sys.argv[3]
ntr, nval = int(sys.argv[4]), int(sys.argv[5])
import os
os.makedirs(out, exist_ok=True)
ds = load_dataset(name, config, trust_remote_code=True)
def _prep(split, n):
    if n > 0:
        split = split.select(range(min(n, len(split))))
    keep = {"input", "output"}
    drop = [c for c in split.column_names if c not in keep]
    if drop:
        split = split.remove_columns(drop)
    return split
_prep(ds["train"], ntr).to_parquet(out + "/train-00000-of-00001.parquet")
_prep(ds["validation"], nval).to_parquet(out + "/validation-00000-of-00001.parquet")
print("smoke dataset prepared at " + out)
PYEOF
  # Pass params as argv (bash expands them, including the non-exported shell var
  # LLAMA2_DATASET_SUBDIR that python os.environ could not see -> the KeyError).
  python3 /tmp/prep_llama2_smoke.py "\${MLPERF_LLAMA2_SMOKE_DATASET_NAME}" "\${MLPERF_LLAMA2_SMOKE_DATASET_CONFIG}" "\${LLAMA2_DATA_DIR}" "\${MLPERF_LLAMA2_SMOKE_TRAIN_SAMPLES:-128}" "\${MLPERF_LLAMA2_SMOKE_VAL_SAMPLES:-32}"
fi
if [[ ! -f "\${LLAMA2_DATA_DIR}/train-00000-of-00001.parquet" ]]; then
  echo "ERROR: dataset parquet missing for resolved mode \${MLPERF_LLAMA2_MODE}" >&2
  exit 1
fi
if [[ ! -d "\${LLAMA2_MODEL_DIR}" ]]; then
  echo "ERROR: model directory missing for resolved mode \${MLPERF_LLAMA2_MODE}" >&2
  exit 1
fi
SEED="${MLPERF_LLAMA2_SEED}"
# Use the module form so it works whether accelerate is on PATH or only importable.
python3 -m accelerate.commands.launch --config_file configs/h200_4gpu.yaml scripts/train.py \\
  --dataset_path "\${LLAMA2_DATA_DIR}" \\
  --model_path "\${LLAMA2_MODEL_DIR}" \\
  --max_seq_len 8192 \\
  --bf16 True \\
  --logging_steps 24 \\
  --eval_steps ${llama2_eval_steps} \\
  --output_dir "./results/llama-70b_scrolls_gov_report_r16_\${SEED}" \\
  --per_device_train_batch_size 1 \\
  --gradient_accumulation_steps 1 \\
  --lr_scheduler_type "cosine" \\
  --learning_rate 4e-4 \\
  --weight_decay 0.0001 \\
  --warmup_ratio 0 \\
  --max_grad_norm 0.3 \\
  --use_gradient_checkpointing True \\
  --target_eval_loss 0.925 \\
  --use_peft_lora True \\
  --lora_r 16 \\
  --lora_alpha 32 \\
  --lora_dropout 0.1 \\
  --max_steps ${llama2_max_steps} \\
  \${FA_FLAG} \\
  --seed "\${SEED}" \\
  --lora_target_modules "qkv_proj,o_proj"
'$(quick_timeout_suffix mlperf-llama2)
EOF
}

render_gpt_oss20b() {
  # In quick-run, append step/eval overrides to the generated config. They are
  # appended (not edited in place) so they win when run_with_docker.sh re-sources
  # the file, and so its 'compgen -e' env-forward still picks them up. The
  # timeout bounds wall-clock; the high iter ceiling just avoids finishing early.
  local gpt_quick_cfg=""
  if [[ "${MLPERF_QUICK_RUN}" == "1" ]]; then
    gpt_quick_cfg="cat >> config_H200_1x4x1.sh <<'QCFG'
export PRIMUS_TRAIN_ITERS=${MLPERF_QUICK_GPT_OSS_ITERS}
export PRIMUS_EVAL_INTERVAL=${MLPERF_QUICK_EVAL_DISABLE}
QCFG
echo \"quick-run: capped gpt_oss to ${MLPERF_QUICK_GPT_OSS_ITERS} iters, eval disabled\""
  fi
  cat <<EOF
set -euo pipefail
mkdir -p "${MLPERF_GPT_OSS_RESULTS_PATH}"
# Megatron builds a per-dataset index/shuffle cache. Without an explicit cache
# path it writes those "dataset materials" next to the data (/data), which fails
# with "OSError: ... 0 written / Failed to write dataset materials ... supply a
# directory ... via the path_to_cache attribute". The bind-mounted /results and
# /data both live on lustre, which root-squashes the container's root user, so a
# cache path there still cannot be written (0 bytes written). Point it at the
# container-local /tmp instead (always writable; the cache is regenerable, so
# losing it when the --rm container exits is fine).
cd "${MLPERF_UPSTREAM_DIR}/small_llm_moe_pretraining/primus"
cat > config_H200_1x4x1.sh <<'CFG'
#!/bin/bash
export DGXSYSTEM=H200_1x4x1
export GPUS_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29501

# Megatron-LM requires CUDA_DEVICE_MAX_CONNECTIONS=1 for correctly ordered
# comm/compute overlap. Without it the synthetic warmup's all-reduce raced
# across CUDA streams and aborted with "unhandled cuda error / Failed to CUDA
# calloc async 72 bytes" (preceded by AccumulateGrad stream-mismatch warnings
# from schedules.py). NCCL_DEBUG=WARN surfaces the NCCL error if it recurs.
# These reach the in-container training process: run_with_docker.sh auto-forwards
# every var this config exports -- it builds its "docker exec --env=" list from
# "compgen -e" after sourcing this file in a clean env, so any exported name here
# is passed through. Verified against the upstream run_with_docker.sh.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
# The 20B MoE runs right at the 140 GiB H200 limit and OOM'd in the expert
# forward (~131 GiB allocated, only ~389 MiB free). expandable_segments recovers
# reserved-but-unallocated fragmentation, which the allocator itself recommends.
# NOTE: this is a thin margin -- if it still OOMs, the real fix is a distributed
# optimizer / more model parallelism (TP>1) to shard optimizer state.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PRIMUS_PATH=/workspace/deps/Primus
export PYTHONPATH="\${PRIMUS_PATH}:\${PRIMUS_PATH}/third_party/Megatron-LM:\${PYTHONPATH:-}"
export EXP=/workspace/code/conf/gpt_oss_20B-pretrain-nvidia.yaml
export DATA_PATH=/data
export MODEL=/model

# micro-batch 2 OOMs by ~60 MiB on the 140 GiB H200 (133 GiB already resident);
# halving the micro-batch frees several GiB of activation memory so the 20B MoE
# fits. Throughput is still a valid hardware measurement at mbs=1.
export PRIMUS_MICRO_BATCH_SIZE=1
export PRIMUS_GLOBAL_BATCH_SIZE=16
export PRIMUS_LR=4.0e-4
export PRIMUS_MIN_LR=4.0e-5
export PRIMUS_TRAIN_ITERS=1200000
export PRIMUS_LR_WARMUP_ITERS=128
export PRIMUS_LR_DECAY_ITERS=\$((PRIMUS_TRAIN_ITERS-PRIMUS_LR_WARMUP_ITERS))

export EVAL_SAMPLES_INTERVAL=12288
export PRIMUS_EVAL_INTERVAL=\$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))
export EVAL_SAMPLES=1024
export PRIMUS_EVAL_ITERS=\$((EVAL_SAMPLES / PRIMUS_GLOBAL_BATCH_SIZE))

export PRIMUS_BF16=true
export PRIMUS_FP16=false
export PRIMUS_FP8=null
export PRIMUS_TURBO_ENABLED=false
export USE_TURBO_ATTENTION=false
export USE_TURBO_GROUPED_MLP=false
export USE_TURBO_DEEPEP=false
export TURBO_DEEPEP_NUM_CU=0
export TURBO_SYNC_FREE_MOE_STAGE=0
export PRIMUS_APPLY_ROPE_FUSION=false
export USE_ROCM_MEM_INFO=false
export OVERLAP_GRAD_REDUCE=true
export OVERLAP_PARAM_GATHER=true
export GRADIENT_ACCUMULATION_FUSION=false

export PRIMUS_TP=1
export PRIMUS_PP=1
export PRIMUS_EP=4

export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_output.log
export MLLOG_TRAIN_LOSS_LOG_FREQ=32
export MLLOG_TARGET_EVAL_LOSS=3.34
export MLLOG_SUBMISSION_BENCHMARK=gpt-oss-20b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=local
export MLLOG_SUBMISSION_PLATFORM=H200-4GPU

export HF_TOKEN="\${HF_TOKEN:-}"
CFG
chmod +x config_H200_1x4x1.sh
${gpt_quick_cfg}
# Upstream Dockerfile.nvidia clones Primus and runs 'git checkout main', then
# applies primus_evaluator.patch. Primus main has since moved (the evaluator fix
# was upstreamed in 8c5bc42d), so that patch no longer applies and the build
# fails. Pin Primus to ${MLPERF_GPT_OSS_PRIMUS_REF} -- the newest commit whose
# evaluator.py still matches the patch base (blob f7df2870) -- so both MLPerf
# patches apply cleanly. Idempotent: only the literal 'git checkout main' is
# rewritten, so re-runs over an already-pinned Dockerfile are a no-op.
if grep -q 'git checkout main' Dockerfile.nvidia; then
  sed -i "s|git checkout main|git checkout ${MLPERF_GPT_OSS_PRIMUS_REF}|" Dockerfile.nvidia
  echo "Pinned Primus to ${MLPERF_GPT_OSS_PRIMUS_REF} in Dockerfile.nvidia"
fi
# Upstream Dockerfile.nvidia hardcodes 'pip install primus_mllog-0.1.0-...whl',
# but the dir actually ships a different version (e.g. primus_mllog-0.1.20),
# so the build fails with "No such file". Rewrite the exact version to a glob so
# whichever primus_mllog wheel is present in the build context is installed.
if grep -q 'pip install primus_mllog-0.1.0-py3-none-any.whl' Dockerfile.nvidia; then
  sed -i 's|pip install primus_mllog-0.1.0-py3-none-any.whl|pip install primus_mllog-*-py3-none-any.whl|' Dockerfile.nvidia
  echo "Rewrote primus_mllog wheel install to a version glob in Dockerfile.nvidia"
fi
# Point Megatron's dataset index/shuffle cache at container-local /tmp (see the
# note above; lustre mounts are not writable by the container's squashed root).
# Inject data_cache_path into the model overrides block of the benchmark config,
# right after the *_data_path entries, matching their indentation. Idempotent:
# only added when not already present. The conf is re-cloned each run, so this
# patch reapplies every time.
if [ -f conf/gpt_oss_20B-pretrain-nvidia.yaml ]; then
  # Normalize (not just inject-if-absent): the conf persists across runs, so an
  # earlier run may have written a stale data_cache_path (e.g. the lustre
  # /results path, which the container's squashed root cannot write). Delete any
  # existing data_cache_path line, then set it to container-local /tmp.
  sed -i '/^[[:space:]]*data_cache_path:/d' conf/gpt_oss_20B-pretrain-nvidia.yaml
  sed -i 's#^\(\s*\)test_data_path:.*#&\n\1data_cache_path: /tmp/gpt_oss_dataset_cache#' conf/gpt_oss_20B-pretrain-nvidia.yaml
  echo "Set data_cache_path: /tmp/gpt_oss_dataset_cache in gpt_oss conf"
  # The conf suppresses Megatron's per-iteration console log (log_interval:
  # 99999999), so the only perf signal is MLLOG train_loss with no throughput --
  # the report cannot score it. Lower log_interval so Megatron emits its standard
  # "elapsed time per iteration (ms)" + "global batch size" line, which the report
  # parses into throughput. Harmless to perf; just more log lines.
  sed -i 's#^\(\s*\)log_interval:.*#\1log_interval: 5#' conf/gpt_oss_20B-pretrain-nvidia.yaml
  echo "Set log_interval: 5 in gpt_oss conf (so per-iteration throughput is logged)"
fi
docker build -t "${MLPERF_GPT_OSS_IMAGE}" -f Dockerfile.nvidia .
export DGXSYSTEM=H200_1x4x1
export CONT="${MLPERF_GPT_OSS_IMAGE}"
export DATADIR="${MLPERF_GPT_OSS_DATA_PATH}"
export MODELDIR="${MLPERF_GPT_OSS_MODEL_PATH}"
export LOGDIR="${MLPERF_GPT_OSS_RESULTS_PATH}"
export HF_TOKEN="\${HF_TOKEN:-}"
export NEXP="${MLPERF_GPT_OSS_NEXP}"
export CLEAR_CACHES="${MLPERF_GPT_OSS_CLEAR_CACHES}"
# run_with_docker.sh forwards the benchmark settings into the container as bare
# '--env=NAME' flags, i.e. it passes through THIS shell's value for each NAME.
# Those settings (NODE_RANK, MASTER_ADDR, MASTER_PORT, NNODES and the
# PRIMUS_*/MLLOG_* vars) live in config_H200_1x4x1.sh, so source it here to
# export them; otherwise NODE_RANK arrives empty and torchrun aborts with
# "argument --node-rank: invalid int value: ''".
set -a
source ./config_H200_1x4x1.sh
set +a
$(quick_timeout_prefix dev)bash ./run_with_docker.sh$(quick_timeout_suffix dev)
EOF
}

render_flux() {
  # In quick-run, cap steps high and push eval out past the window (the timeout
  # bounds wall-clock). One line so the surrounding \-continuation stays valid.
  local flux_quick_args=""
  if [[ "${MLPERF_QUICK_RUN}" == "1" ]]; then
    flux_quick_args="--training.steps=${MLPERF_QUICK_FLUX_STEPS} --eval.eval_freq=${MLPERF_QUICK_EVAL_DISABLE}"
  fi
  cat <<EOF
set -euo pipefail
mkdir -p "${MLPERF_FLUX_RESULTS_PATH}"
cd "${MLPERF_UPSTREAM_DIR}/text_to_image/torchtitan"
docker build -t "${MLPERF_FLUX_IMAGE}" -f Dockerfile .
$(quick_timeout_prefix mlperf-flux)docker run --rm --name mlperf-flux --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
  -v "${MLPERF_HF_CACHE}:/root/.cache" \\
  -v "${MLPERF_FLUX_DATASET_PATH}:/dataset" \\
  -v "${MLPERF_FLUX_MODEL_PATH}:/models" \\
  -v "${MLPERF_FLUX_RESULTS_PATH}:/results" \\
  -v "${MLPERF_UPSTREAM_DIR}/text_to_image/torchtitan:/workspace" \\
  -w /workspace \\
  "${MLPERF_FLUX_IMAGE}" \\
  bash -lc '
set -euo pipefail
# flux trained ~110 steps then died with no traceback (a SIGKILL, consistent with
# the host/GPU running out of memory near the 140 GiB limit). Reduce allocator
# fragmentation so it does not creep into an OOM kill mid-run.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -r requirements-mlperf.txt
pip install -r torchtitan/experiments/flux/requirements-flux.txt
CONFIG_FILE=./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf_preprocessed.toml \\
NGPU="${MLPERF_GPU_COUNT}" \\
./torchtitan/experiments/flux/run_train.sh \\
  --job.dump_folder=/results \\
  --training.dataset_path=/dataset/cc12m_preprocessed \\
  --eval.dataset_path=/dataset/coco_preprocessed \\
  --encoder.empty_encodings_path=/dataset/empty_encodings \\
  ${flux_quick_args} \\
  --training.seed=${MLPERF_FLUX_SEED}
'$(quick_timeout_suffix mlperf-flux)
EOF
}

run_target() {
  local target="$1"
  if [[ "${EXECUTE}" -eq 1 ]]; then
    ensure_upstream
  fi
  case "${target}" in
    llama31)
      run_or_print "llama31" "$(render_llama31)"
      ;;
    llama2_lora)
      run_or_print "llama2_lora" "$(render_llama2_lora)"
      ;;
    gpt_oss20b)
      run_or_print "gpt_oss20b" "$(render_gpt_oss20b)"
      ;;
    flux)
      run_or_print "flux" "$(render_flux)"
      ;;
    all)
      run_or_print "llama31" "$(render_llama31)"
      echo
      run_or_print "llama2_lora" "$(render_llama2_lora)"
      echo
      run_or_print "gpt_oss20b" "$(render_gpt_oss20b)"
      echo
      run_or_print "flux" "$(render_flux)"
      ;;
    *)
      echo "ERROR: unknown benchmark target: ${target}" >&2
      exit 1
      ;;
  esac
}

ACTION="${1:-}"
TARGET="${2:-}"

case "${ACTION}" in
  show)
    if [[ -z "${TARGET}" ]]; then
      usage
      exit 1
    fi
    EXECUTE=0
    run_target "${TARGET}"
    ;;
  run)
    if [[ -z "${TARGET}" ]]; then
      usage
      exit 1
    fi
    if [[ "${EXECUTE}" -ne 1 ]]; then
      echo "ERROR: refusing to launch work without --execute. Use 'show' to inspect commands first." >&2
      exit 1
    fi
    run_target "${TARGET}"
    ;;
  report)
    if [[ -n "${TARGET}" ]]; then
      python3 "${REPO_ROOT}/scripts/report_mlperf6_results.py" --env-file "${ENV_FILE}" --output "${TARGET}"
    else
      python3 "${REPO_ROOT}/scripts/report_mlperf6_results.py" --env-file "${ENV_FILE}"
    fi
    ;;
  *)
    usage
    exit 1
    ;;
esac

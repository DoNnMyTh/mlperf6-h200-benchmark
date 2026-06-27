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

Notes:
  - `show` prints the exact commands without running them.
  - `run` requires `--execute`; without it, the script refuses to launch work.
  - `report` writes a markdown summary using the current results directories.
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
  cat <<EOF
set -euo pipefail
mkdir -p "${MLPERF_LLAMA31_RESULTS_PATH}" "${MLPERF_LLAMA31_CHECKPOINT_PATH}" "${MLPERF_LLAMA31_INDEX_PATH}"
cd "${MLPERF_UPSTREAM_DIR}/small_llm_pretraining/nemo"
docker build -t "${MLPERF_LLAMA31_IMAGE}" -f Dockerfile.h200 .
source ./config_H200_1x8x1_8b.sh
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
export CUDA_HOME="${MLPERF_CUDA_HOME}"
bash ./run_llama31.sh
EOF
}

render_llama2_lora() {
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
docker pull "${MLPERF_LLAMA2_DOCKER_IMAGE}"
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
  -e HF_TOKEN="\${HF_TOKEN:-}" \\
  -v "${MLPERF_UPSTREAM_DIR}/llama2_70b_lora:/workspace" \\
  -v "${MLPERF_LLAMA2_DATASET_PATH}:/workspace/dataset" \\
  -v "${MLPERF_LLAMA2_RESULTS_PATH}:/workspace/results" \\
  -v "${MLPERF_LLAMA2_MODEL_ROOT}:/models" \\
  -v "${MLPERF_HF_CACHE}:/root/.cache/huggingface" \\
  -w /workspace \\
  "${MLPERF_LLAMA2_DOCKER_IMAGE}" \\
  bash -lc '
set -euo pipefail
pip install -r requirements.txt
pip install flash-attn==2.1.0 --no-build-isolation
git clone --depth 1 https://github.com/mlperf/logging.git /tmp/mlperf-logging
pip install -e /tmp/mlperf-logging
if [[ -n "\${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "\${HF_TOKEN}"
fi
SEED="${MLPERF_LLAMA2_SEED}"
accelerate launch --config_file configs/h200_4gpu.yaml scripts/train.py \\
  --dataset_path "./dataset/scrolls_gov_report_8k" \\
  --model_path "/models/Llama2-70b-fused-qkv-mlperf" \\
  --max_seq_len 8192 \\
  --bf16 True \\
  --logging_steps 24 \\
  --eval_steps 48 \\
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
  --max_steps 1024 \\
  --use_flash_attn \\
  --seed "\${SEED}" \\
  --lora_target_modules "qkv_proj,o_proj"
'
EOF
}

render_gpt_oss20b() {
  cat <<EOF
set -euo pipefail
mkdir -p "${MLPERF_GPT_OSS_RESULTS_PATH}"
cd "${MLPERF_UPSTREAM_DIR}/small_llm_moe_pretraining/primus"
cat > config_H200_1x4x1.sh <<'CFG'
#!/bin/bash
export DGXSYSTEM=H200_1x4x1
export GPUS_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29501

export PRIMUS_PATH=/workspace/deps/Primus
export PYTHONPATH="\${PRIMUS_PATH}:\${PRIMUS_PATH}/third_party/Megatron-LM:\${PYTHONPATH}"
export EXP=/workspace/code/conf/gpt_oss_20B-pretrain-nvidia.yaml
export DATA_PATH=/data
export MODEL=/model

export PRIMUS_MICRO_BATCH_SIZE=2
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
docker build -t "${MLPERF_GPT_OSS_IMAGE}" -f Dockerfile.nvidia .
export DGXSYSTEM=H200_1x4x1
export CONT="${MLPERF_GPT_OSS_IMAGE}"
export DATADIR="${MLPERF_GPT_OSS_DATA_PATH}"
export MODELDIR="${MLPERF_GPT_OSS_MODEL_PATH}"
export LOGDIR="${MLPERF_GPT_OSS_RESULTS_PATH}"
export HF_TOKEN="\${HF_TOKEN:-}"
export NEXP="${MLPERF_GPT_OSS_NEXP}"
export CLEAR_CACHES="${MLPERF_GPT_OSS_CLEAR_CACHES}"
bash ./run_with_docker.sh
EOF
}

render_flux() {
  cat <<EOF
set -euo pipefail
mkdir -p "${MLPERF_FLUX_RESULTS_PATH}"
cd "${MLPERF_UPSTREAM_DIR}/text_to_image/torchtitan"
docker build -t "${MLPERF_FLUX_IMAGE}" -f Dockerfile .
docker run --rm --gpus all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
  -v "${MLPERF_HF_CACHE}:/root/.cache" \\
  -v "${MLPERF_FLUX_DATASET_PATH}:/dataset" \\
  -v "${MLPERF_FLUX_MODEL_PATH}:/models" \\
  -v "${MLPERF_FLUX_RESULTS_PATH}:/results" \\
  -v "${MLPERF_UPSTREAM_DIR}/text_to_image/torchtitan:/workspace" \\
  -w /workspace \\
  "${MLPERF_FLUX_IMAGE}" \\
  bash -lc '
set -euo pipefail
pip install -r requirements-mlperf.txt
pip install -r torchtitan/experiments/flux/requirements-flux.txt
CONFIG_FILE=./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf_preprocessed.toml \\
NGPU="${MLPERF_GPU_COUNT}" \\
./torchtitan/experiments/flux/run_train.sh \\
  --job.dump_folder=/results \\
  --training.dataset_path=/dataset/cc12m_preprocessed/* \\
  --eval.dataset_path=/dataset/coco_preprocessed/* \\
  --encoder.empty_encodings_path=/dataset/empty_encodings \\
  --training.seed=${MLPERF_FLUX_SEED}
'
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

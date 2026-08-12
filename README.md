# MLPerf 6.0 H200 Benchmark Harness

Runs the official [MLCommons `training`](https://github.com/mlcommons/training)
v6.0 reference benchmarks on a single Linux node with 4x NVIDIA H200 NVL GPUs.

The harness bootstraps the upstream reference code, downloads each benchmark's
official assets, runs the benchmarks one at a time, and writes a single markdown
report.

The target host profile (`t3ihpc07`) is captured in
`configs/mlperf6-h200-4gpu.env`. Edit that file to retarget another node.

## Contents

- [Benchmarks](#benchmarks)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [How to run](#how-to-run)
- [Where models, weights, and data come from](#where-models-weights-and-data-come-from)
- [Authenticity, provenance, and deviations](#authenticity-provenance-and-deviations)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)

---

## Benchmarks

| Key | MLPerf area | Model | Dataset |
| --- | --- | --- | --- |
| `llama31` | LLM pretraining (small) | Llama 3.1 8B | preprocessed C4 |
| `gpt_oss20b` | MoE LLM pretraining (small) | GPT-OSS 20B | same preprocessed C4 corpus |
| `llama2_lora` | LLM fine-tuning | Llama 2 70B + LoRA r16 | SCROLLS GovReport |
| `flux` | Text-to-image | FLUX.1-schnell | CC12M (eval: COCO) |

---

## Prerequisites

### Host

- Linux. `run_all_mlperf6_h200.sh` refuses to start anywhere else.
- 4x NVIDIA H200 (143,771 MiB each on the reference node), driver 595.45.04.
- Docker with the NVIDIA container runtime (`--gpus all` must work).
- `git`, `curl`, `python3` on PATH. The orchestrator preflights all four.

### Disk

~2.5 TB free under `MLPERF_DATA_ROOT` for a full four-benchmark run. See
[Download footprint](#download-footprint). Nothing is ever deleted between runs,
so plan for the cumulative total, not the largest single benchmark.

### Credentials

| Variable | Needed for | Effect if unset |
| --- | --- | --- |
| `HF_TOKEN` | Llama 2 70B weight prestage from Hugging Face | The 70B prestage is skipped; `llama2_lora` falls back to mode-based handling and will fail unless official assets are already on disk |
| `MLPERF_LLAMA2_RCLONE_CONFIG` | MLCommons member-gated Llama 2 assets | `llama2_lora` resolves to a public-mirror mode instead of `official` |

Everything else downloads anonymously over public HTTPS.

---

## Quick start

```bash
# 1. Clone the upstream MLCommons reference code
./scripts/bootstrap_mlperf6_h200.sh

# 2. Read the exact commands before running anything
./scripts/run_mlperf6_h200.sh show all

# 3. Run the whole pipeline (hardware-perf mode, ~1 h per benchmark)
export HF_TOKEN=hf_...            # required for Llama 2 70B weights
./scripts/run_all_mlperf6_h200.sh
```

The report lands at `MLPERF_FINAL_REPORT_PATH`
(default `${MLPERF_RESULTS_ROOT}/mlperf6-h200-final-report.md`).

> **The default mode is `--quick-run`, which is a hardware-performance
> measurement, not a valid MLPerf submission result.** See
> [quick-run vs full](#quick-run-vs-full).

---

## How to run

### Step 1 — configure

All paths, image tags, dataset URIs, batch sizes, and step budgets live in
`configs/mlperf6-h200-4gpu.env`. Every variable uses `${X:-default}`, so an
exported environment variable always overrides the file.

The paths you are most likely to change:

```bash
MLPERF_WORK_ROOT      # upstream checkout + scratch     (default /scratch/.../work)
MLPERF_DATA_ROOT      # datasets + model weights        (default /scratch/.../data)
MLPERF_RESULTS_ROOT   # logs, MLLOG output, the report  (default /scratch/.../results)
MLPERF_HF_CACHE       # Hugging Face cache
```

### Step 2 — bootstrap

```bash
./scripts/bootstrap_mlperf6_h200.sh [ENV_FILE]
```

Creates every root directory, then clones
`https://github.com/mlcommons/training.git` into `MLPERF_UPSTREAM_DIR`
(`--depth 1 --recurse-submodules`) or fast-forwards an existing checkout, plus
an explicit submodule init for `text_to_image/torchtitan`.

Reference code only — no weights, no datasets. Safe and idempotent to re-run.

### Step 3 — inspect before executing

```bash
./scripts/run_mlperf6_h200.sh show llama31
./scripts/run_mlperf6_h200.sh show all
```

`show` prints the exact shell that *would* run — every `docker build`,
`docker run`, bind mount, environment variable, patch, and training flag — and
executes nothing. `run` without `--execute` refuses to launch:

```text
ERROR: refusing to launch work without --execute. Use 'show' to inspect commands first.
```

This is the audit path. Every modification the harness makes to upstream code is
visible in `show` output before anything runs.

### Step 4a — run one benchmark

```bash
./scripts/run_mlperf6_h200.sh --execute run llama31
```

Runs *only* the training. It does **not** download data, write the status
ledger, or generate the report. Use it when the assets are already staged.

### Step 4b — run the full pipeline

```bash
./scripts/run_all_mlperf6_h200.sh [options]
```

Preflight → bootstrap → for each benchmark `download → run → report` → final
report from an `EXIT` trap (so a report is written even when stages fail).

```text
preflight            docker | git | curl | python3
  |
  +-- bootstrap      clone/update mlcommons/training
  |
  +-- FOR each benchmark: llama31, gpt_oss20b, llama2_lora, flux
  |        download_<bench>  ->  run_<bench>  ->  generate_report (incremental)
  |
  +-- EXIT trap      generate_report (always)
```

#### Options

| Flag | Effect |
| --- | --- |
| `--env-file PATH` | Use an alternate env file |
| `--report-output PATH` | Override the final report path |
| `--benchmarks CSV` | Subset, e.g. `--benchmarks llama31,flux` |
| `--skip-bootstrap` | Reuse the existing upstream checkout |
| `--skip-downloads` | Skip all download stages |
| `--skip-runs` | Download only — use to pre-stage every dataset first |
| `--fail-fast` | Stop at the first failing stage (default: continue) |
| `--quick-run` | Hardware-perf mode (**default**) |
| `--quick-run-seconds N` | Perf window per benchmark (default 3600) |
| `--full` | Real convergence run, no time box |

**Why the order is fixed.** The default is
`llama31,gpt_oss20b,llama2_lora,flux`. `llama31` runs first because
`gpt_oss20b` reuses its C4 corpus; `flux` runs last because it is by far the
largest download (~2.23 TB), so cheaper benchmarks finish first.

**Download-one / run-one.** Assets for one benchmark are fetched, that benchmark
runs, and only then does the next begin. Results land incrementally, so a
completed benchmark is already in the report if a later one dies. This bounds the
*active* working set — the *persistent* footprint still accumulates, because
downloaded data is retained for reuse.

### Common recipes

```bash
# Pre-stage every dataset overnight, run nothing
./scripts/run_all_mlperf6_h200.sh --skip-runs

# Then run everything against the staged data
./scripts/run_all_mlperf6_h200.sh --skip-downloads

# One benchmark, end to end, stop on first failure
./scripts/run_all_mlperf6_h200.sh --benchmarks flux --fail-fast

# Short 10-minute perf sample per benchmark
./scripts/run_all_mlperf6_h200.sh --quick-run-seconds 600

# Convergence run (hours to days per benchmark)
./scripts/run_all_mlperf6_h200.sh --full

# Regenerate the report from existing logs, run nothing
./scripts/run_mlperf6_h200.sh report /tmp/report.md
```

### quick-run vs full

| | quick-run (**default**) | `--full` |
| --- | --- | --- |
| Purpose | Hardware throughput / step time | Convergence to target quality |
| Wall clock | `MLPERF_QUICK_RUN_SECONDS`, default 3600 s per benchmark | Unbounded |
| Eval | Disabled (interval pushed to 2^30) | Enabled |
| Step budget | ~10^6 ceiling; the time window is the real bound | llama2 1024 steps, flux 247,000 steps |
| llama2 data | Small smoke subset (`-quick` folder) | Whole GovReport set (`-full` folder) |
| Valid submission | **No** — no target-loss convergence | Submission-*style*; see [deviations](#deviations-from-upstream) |

Quick and full prepare their llama2 datasets into **separate** folders, so
switching modes never reuses the wrong dataset.

**How the time box works.** Not `timeout` — its `SIGTERM`→`SIGKILL` escalation
kills the `docker run` CLI but leaves the *container* alive holding the GPUs.
Instead a background watchdog force-removes the container by name at the window,
which makes the foreground launch exit immediately and frees the GPUs
deterministically. Exit codes 124/137/143 are treated as success (a throughput
sample was captured); any other non-zero code is a real crash and propagates.

### Resuming

Re-running is cheap and non-destructive:

- Each completed download writes a `.mlperf-download-complete` marker; a marker
  present means the stage is skipped instantly.
- Partial downloads resume (`wget --continue`, `hf download` resume).
- **Nothing on disk is ever deleted** — no `rm -rf`, no `rsync --delete`.
  Directory syncs are additive only.

---

## Where models, weights, and data come from

Every asset is pulled from an official or explicitly named public source. No
weights are vendored in this repository, and none are re-hosted.

### Datasets — MLCommons R2 storage

Fetched with the official
[`mlcommons/r2-downloader`](https://github.com/mlcommons/r2-downloader)
(`MLPERF_R2_DOWNLOADER_URL`) against published `.uri` manifests on
`training.mlcommons-storage.org`:

| Benchmark | Asset | Manifest (`.uri`) | Size |
| --- | --- | --- | --- |
| `llama31` | preprocessed C4 corpus | `llama-3-1-8b-preprocessed-c4-dataset.uri` | ~79 GB |
| `llama31` | 8B tokenizer / model | `llama-3-1-8b-tokenizer.uri` | ~30 GB |
| `gpt_oss20b` | *reuses the llama31 C4 corpus* | — | 0 |
| `flux` | CC12M preprocessed (train) | `flux-1-cc12m-preprocessed.uri` | ~2.17 TB |
| `flux` | COCO preprocessed (eval) | `flux-1-coco-preprocessed.uri` | ~60 GB |
| `flux` | empty text encodings | `flux-1-empty-encodings.uri` | ~2 MB |

**Integrity.** The downloader fetches each shard with `wget --continue` and then
verifies the set with `md5sum -c` against the manifest's `.md5`. A download is
only marked complete when that verification passes.

**Corruption repair.** On a flaky link a shard can land corrupt at its full
expected size; `wget --continue` then treats it as complete and skips it forever,
so md5 verification fails permanently and a plain retry never converges. The
harness wraps the downloader in a repair loop
(`MLPERF_R2_DOWNLOAD_RETRIES`, default 5): on failure it parses the `.md5`,
deletes **only** the shards that failed verification, and re-invokes. wget
refetches exactly those and skips the good ones.

### `gpt_oss20b` — dataset reuse, not a second download

`gpt_oss20b` consumes the same preprocessed C4 corpus as `llama31`, so it is
symlinked rather than downloaded again:

```bash
ln -sfn "${MLPERF_LLAMA31_PREPROCESSED_PATH}" "${MLPERF_GPT_OSS_DATA_PATH}"
```

The symlink is only created — and on later runs only accepted — when the two
dataset URIs match, the link resolves to the llama31 corpus, **and** llama31's
completion marker exists. Anything stale or unexpected is a hard failure, never
a silent fallback to the wrong dataset. The tokenizer is then copied from the
llama31 tokenizer directory.

### `llama2_lora` — model weights

Two paths, selected by `MLPERF_LLAMA2_MODE` (default `auto`):

**1. Official (MLCommons member-gated).** Requires an rclone config containing
an `[mlc-llama2]` remote. Searched, in order:

```text
$MLPERF_LLAMA2_RCLONE_CONFIG
~/.config/mlperf/llama2-rclone.conf
~/.config/rclone/rclone.conf
```

Found → the harness delegates entirely to the upstream downloader:

```bash
bash ./scripts/download_data.sh --data_dir=... --model_dir=... --rclone_config=...
```

**2. Public mirror (default when no rclone config exists).** Weights come from
Hugging Face:

| Item | Source | Notes |
| --- | --- | --- |
| 70B weights (~130 GB) | `regisss/llama2-70b-fused-qkv-mlperf` (`MLPERF_LLAMA2_PUBLIC_MODEL_ID`) | The fused-QKV MLPerf conversion |
| Tokenizer | `NousResearch/Llama-2-70b-hf` → `NousResearch/Llama-2-7b-hf` → `meta-llama/Llama-2-70b-hf` | First that succeeds. The fused-QKV weights repo ships **no** tokenizer files |
| Dataset | `tau/scrolls`, config `gov_report` | Public GovReport, tokenized and packed locally |

Prestage details that matter operationally:

- Runs **outside** the timed benchmark window, so a 130 GB fetch never eats the
  perf window. The training container fails fast if the model is not staged.
- Runs as the invoking host user (`--user $(id -u):$(id -g)`), because the lustre
  mount root-squashes the container root — a root-owned download fails with
  `Permission denied`.
- `HF_HUB_DISABLE_XET=1` and `HF_HUB_ENABLE_HF_TRANSFER=0`. Both accelerated
  backends died mid-transfer on this node (`Background writer channel closed`;
  `[Errno 14] Bad address`). Plain HTTPS is slower but resumes reliably.
- 10 retry attempts, each resuming and fetching only missing shards. Hard-fails
  if all fail, so an empty model directory is never mistaken for success.
- Integrity relies on the Hugging Face hub's own transfer verification; the
  harness additionally asserts that `config.json`, `modeling_llama.py`, and
  `model.safetensors.index.json` exist before training starts.

### Container images

| Benchmark | Image | Origin |
| --- | --- | --- |
| `llama31` | `local/mlperf6-llama31-h200` | Built from upstream `small_llm_pretraining/nemo/Dockerfile.h200` |
| `gpt_oss20b` | `local/mlperf6-gpt-oss20b-h200` | Built from upstream `small_llm_moe_pretraining/primus/Dockerfile.nvidia` |
| `flux` | `local/mlperf6-flux-h200` | Built from upstream `text_to_image/torchtitan/Dockerfile` |
| `llama2_lora` | `nvcr.io/nvidia/pytorch:23.09-py3` | Pulled from NGC, as the upstream reference specifies |

### Download footprint

| Benchmark | Download-stage assets | Size |
| --- | --- | --- |
| `llama31` | C4 (~79 GB) + tokenizer/model (~30 GB) | ~109 GB |
| `gpt_oss20b` | reuses llama31 C4 via symlink + synced tokenizer | ~0 GB |
| `llama2_lora` | 70B weights via prestage; dataset gated/tiny | ~128 GB |
| `flux` | CC12M (~2.17 TB) + COCO (~60 GB) + encodings (~2 MB) | ~2.23 TB |
| **Total** | | **~2.46 TB** |

---

## Authenticity, provenance, and deviations

**Short version:** all training code, model definitions, optimizers, loss
functions, and datasets are the unmodified upstream MLCommons reference. The
harness modifies only build plumbing, I/O paths, logging, and system-size
configuration — with **three exceptions that do affect result validity**, listed
explicitly below. Read this section before quoting any number from this harness.

### What is guaranteed unmodified

- **Training code.** Every benchmark executes from a fresh
  `git clone https://github.com/mlcommons/training.git`. This repository contains
  no fork, no vendored copy, and no patch to any model, optimizer, loss, or data
  pipeline source file.
- **Datasets.** All corpora are the official MLCommons preprocessed artifacts,
  md5-verified against the published manifests. No dataset is re-generated,
  resampled, or truncated — except the `llama2_lora` public-substitute path,
  which is labelled as such (see below).
- **Llama 2 LoRA hyperparameters.** In `--full` mode the harness passes flags
  identical to the upstream reference `run_llama_70B_scrolls_r16.sh`:

  | Parameter | Upstream | This harness (`--full`) |
  | --- | --- | --- |
  | `max_seq_len` | 8192 | 8192 |
  | `logging_steps` / `eval_steps` | 24 / 48 | 24 / 48 |
  | `per_device_train_batch_size` | 1 | 1 |
  | `gradient_accumulation_steps` | 1 | 1 |
  | `learning_rate` / scheduler | 4e-4 / cosine | 4e-4 / cosine |
  | `weight_decay` / `warmup_ratio` | 0.0001 / 0 | 0.0001 / 0 |
  | `max_grad_norm` | 0.3 | 0.3 |
  | `target_eval_loss` | 0.925 | 0.925 |
  | LoRA r / alpha / dropout | 16 / 32 / 0.1 | 16 / 32 / 0.1 |
  | `lora_target_modules` | `qkv_proj,o_proj` | `qkv_proj,o_proj` |
  | `max_steps` | 1024 | 1024 |

- **Llama 2 accelerate config.** `configs/h200_4gpu.yaml` is upstream
  `configs/default_config.yaml` with exactly one change: `num_processes` 8 → 4,
  matching the GPU count. ZeRO stage 3, `gradient_clipping: 0.3`, bf16, and all
  other keys are unchanged.
- **FLUX config.** Uses upstream
  `flux_schnell_mlperf_preprocessed.toml` as-is. Only dataset paths, step count,
  and seed are passed on the command line.
- **Auditability.** `show <benchmark>` prints every command, patch, and flag
  before anything executes. All patches are idempotent `sed`/append operations
  applied to the freshly cloned tree, so nothing accumulates across runs.

### Deviations from upstream

Grouped by whether they can affect a reported result.

#### A. Build and runtime plumbing — no effect on training math

| Change | Reason |
| --- | --- |
| Append `pip install wandb` + `WANDB_MODE=offline` to `Dockerfile.h200` | The NeMo launcher imports `wandb` at module top; offline mode needs no API key |
| `sed` the `mkdir /mlperf-outputs` guard in `run_llama31.sh` | Directory is bind-mounted; make the guard non-fatal |
| `sed` `exp.run(detach=True)` → `detach=False` in `pretrain_llama31.py` | With `detach=True` the console shows only "Waiting for job … [log=False]" while GPUs are saturated — indistinguishable from a hang. Logging only |
| Background tailer for `/root/.nemo_run/**/-steps/*` | `LocalExecutor` writes per-step logs to its experiment dir, not stdout; they would be lost when the `--rm` container exits |
| Pin Primus to `9788c180` instead of `git checkout main` | Upstream `Dockerfile.nvidia:20` checks out `main`, which has drifted past the base `primus_evaluator.patch` was authored against, so the build fails. This is the newest commit where the official MLPerf patches still apply — i.e. it restores upstream intent |
| Rewrite `pip install primus_mllog-0.1.0-…whl` to a version glob | Upstream `Dockerfile.nvidia:36` hardcodes 0.1.0 but the directory ships `primus_mllog-0.1.20`; the build fails with "No such file" |
| Force `data_cache_path: /tmp/gpt_oss_dataset_cache` | Megatron writes its index/shuffle cache next to the data; lustre root-squash makes that fail with `Failed to write dataset materials … 0 written`. The cache is regenerable |
| Set `log_interval: 5` in the gpt_oss conf | Upstream sets `99999999`, suppressing the per-iteration line the report parses for throughput. Extra log lines only |
| Redirect compliance log, checkpoints, and HF cache to `/tmp` | The bind-mounted lustre paths are not writable by the root-squashed container root |
| `PYTORCH_CUDA_ALLOC_CONF` allocator flags | Fragmentation mitigation near the 140 GiB limit. Allocator behaviour only, not numerics |

#### B. System-size configuration — legitimate, but changes comparability

| Change | Detail |
| --- | --- |
| `llama31` GPU count | Upstream ships `config_H200_1x8x1_8b.sh` (8 GPUs). The harness sources it but sets `GPUS_PER_NODE=4`, `GBS=32`, `MBS=2` for this 4-GPU node |
| `gpt_oss20b` config authored locally | Upstream ships only `config_B200_1x8x1.sh` and `config_MI355X_1x8x1.sh` — no H200 config. `config_H200_1x4x1.sh` is written by this harness: micro-batch 1, global batch 16, TP 1 / PP 1 / EP 4 |
| `flux` step budget | `--full` sets 247,000 steps; the upstream toml default is `steps = 30_000`, far short of the ~15.8M samples FLUX needs to converge |

Batch size and parallelism are legitimate submitter choices, but results are only
comparable against reference convergence points (RCPs) for the same scale.

#### C. Changes that affect result validity — read carefully

1. **quick-run is not a submission result.** It is the default. Training is time
   boxed and evaluation is disabled, so no target loss is ever reached. It yields
   throughput and step time only. Any "projected time-to-train" in the report is
   an arithmetic extrapolation (`REF_SAMPLES / throughput`), **not** a measured
   convergence, and not a substitute for the required 10-run measurement.

2. **flash-attn version metadata is spoofed for `llama2_lora`.** The upstream
   step force-builds flash-attn 2.1.0 from source; 2.1.0 predates CUDA 13 /
   Hopper sm_90 and does not compile on this node. The harness instead uses the
   flash-attn already in the NGC image (**2.0.4**) and rewrites its dist-info
   `Version:` field to `2.1.0`, because transformers gates FA2 on
   `importlib.metadata.version("flash_attn") >= 2.1.0` while the fused-QKV
   model's `utils.py` hardcodes `attn_implementation=flash_attention_2`.
   The kernels that actually execute are 2.0.4 — API-compatible with the
   transformers 4.38 llama FA2 path, but **this is a deviation from the reference
   software stack and must be resolved before any submission.**

3. **Non-official `llama2_lora` assets.** Without an MLCommons rclone config the
   benchmark runs against the public HF weight mirror and a locally tokenized
   `tau/scrolls` GovReport dataset. The prep writes
   `SMOKE_TEST_METADATA.json` with `submission_valid: false` into the dataset
   directory so it can never be mistaken for official input. In quick-run this is
   additionally only a small subset; a subset cannot converge (it overfits —
   `train_loss` → ~0.02 while `eval_loss` diverges 1.15 → 2.46), which is why
   `--full` uses the whole GovReport set.

### For a submission-grade run

1. `--full` (never quick-run).
2. Official MLCommons Llama 2 assets via `MLPERF_LLAMA2_RCLONE_CONFIG`
   (`MLPERF_LLAMA2_MODE=official`).
3. Resolve the flash-attn 2.1.0 requirement properly rather than via the metadata
   bump.
4. Validate scale-appropriate RCPs, and perform the required repeated runs.

---

## Outputs

| Path | Content |
| --- | --- |
| `${MLPERF_ORCH_ROOT}/pipeline-status.tsv` | Per-stage ledger: `benchmark / stage / status / note` |
| `${MLPERF_ORCH_ROOT}/orchestrator.log` | Timestamped stage transitions |
| `${MLPERF_ORCH_ROOT}/<bench>-<stage>.log` | Full output of each download and run |
| `${MLPERF_<BENCH>_RESULTS_PATH}/` | MLLOG output and per-benchmark artifacts |
| `MLPERF_FINAL_REPORT_PATH` | Final markdown report |

The report parses `RESULT,<bench>,,<seconds>,…` for convergence runs. Quick-runs
emit no such line, so it scrapes throughput per framework instead — MLLOG
`"throughput"`, NeMo `train_step_timing in s:`, Megatron
`elapsed time per iteration (ms)`, HF Trainer tqdm `Ns/it` — averages the tail 50
samples (warmup is the head, steady state the tail), and derives
`throughput = global_batch_size / step_time` when only step time is available.

Reference sample counts for the projection are set in the env file
(`MLPERF_LLAMA31_REF_SAMPLES`, `MLPERF_FLUX_REF_SAMPLES`, …). Set to `0` to
disable the projection; the report then prints `(set ref samples)`.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ERROR: refusing to launch work without --execute` | Intentional. Use `show` first, then add `--execute` |
| `must be run on the Linux benchmark server` | `run_all` is Linux-only. `show` works anywhere |
| `MLCommons training repo not found` | Run `./scripts/bootstrap_mlperf6_h200.sh` first |
| llama2 run fails with `70B model not staged` | The prestage did not finish. Set `HF_TOKEN` and re-run the download stage; the fetch is deliberately outside the run window |
| `dataset parquet missing` | Dataset prep did not run. Check the resolved mode in the stage log |
| gpt_oss `download-dataset failed … Stale/invalid symlink` | Fails closed by design. Remove the symlink at `MLPERF_GPT_OSS_DATA_PATH` and re-run |
| Downloads verify-fail repeatedly | The repair loop retries `MLPERF_R2_DOWNLOAD_RETRIES` times; raise it for a very flaky link |
| A benchmark shows `run skipped` | Its download stage failed. The run is skipped rather than launched against missing data |
| OOM near 140 GiB | Known thin margin on gpt_oss and llama31. Next levers: distributed optimizer, shorter sequence length, smaller micro-batch |

---

## Repository layout

```text
configs/mlperf6-h200-4gpu.env      all paths, image tags, URIs, run defaults
scripts/
  bootstrap_mlperf6_h200.sh        clone/update upstream mlcommons/training
  run_mlperf6_h200.sh              per-benchmark command renderers; show / run / report
  run_all_mlperf6_h200.sh          orchestrator: downloads, ordering, status ledger
  report_mlperf6_results.py        log parsing, throughput, projection, markdown report
generated/                         local report output (git-ignored)
```

`results/`, `work/`, and `artifacts/` are git-ignored: run output, downloaded
data, and host snapshots stay out of version control.

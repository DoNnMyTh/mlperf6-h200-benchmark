# MLPerf 6.0 H200 Benchmark Bootstrap

This repository is the staging area for an MLPerf 6.0 benchmark run on a Linux
server with 4x NVIDIA H200 GPUs.

The first phase is system discovery. We capture a benchmark-oriented hardware
and software snapshot, review it, and use that output to build the benchmark
runner and final report in the next phase.

## Target benchmark areas

- LLM pretraining (small): Llama 3.1 8B on C4
- LLM fine-tuning: Llama 2 70B LoRA on SCROLLS GovReport
- MoE LLM recommendation system (small): GPT-OSS 20B DLRM-DCNv2 on Criteo Multi-Hot
- Text-to-image generation: FLUX.1 on CC12M

## Repository layout

- `scripts/collect_system_config.sh`: captures a Linux server snapshot into `artifacts/system/`
- `scripts/commit_system_snapshot.sh`: runs the collector and creates a local git commit for that snapshot
- `artifacts/system/`: committed system snapshots used to drive the next automation step

## Safety notes

- Review every generated snapshot before pushing it to a public repository.
- The collector intentionally avoids broad environment dumps and IP address collection.
- Host labels, PCIe layout, package versions, driver versions, and GPU topology can still be sensitive.
- The commit helper never pushes and refuses to run when the repo is already dirty.

## Bootstrap commands

Run these once in this repository:

```bash
git init -b main
git add .
git commit -m "Bootstrap MLPerf 6.0 H200 benchmark repo"
```

If you want the bootstrap published to GitHub under your authenticated account:

```bash
gh repo create DoNnMyTh/mlperf6-h200-benchmark \
  --public \
  --source . \
  --remote origin
git push -u origin main
```

## Collect a system snapshot on the target Linux server

From the repository root:

```bash
chmod +x scripts/*.sh
./scripts/collect_system_config.sh
```

The collector prints the created snapshot directory, for example:

```text
artifacts/system/my-host-20260627T120000Z
```

Review the generated files before committing or publishing them.

## Collect and create a local commit for the snapshot

After the bootstrap commit exists and the repo is clean:

```bash
./scripts/commit_system_snapshot.sh
```

This helper:

- verifies the repository is clean before it starts
- runs the collector
- stages only the new snapshot directory
- creates a local commit
- never pushes

To publish the reviewed snapshot later:

```bash
git push
```

## What the collector captures

- OS release and kernel details
- CPU topology and memory inventory
- block devices, mounts, and filesystem capacity
- PCIe device inventory
- GPU inventory, driver details, and `nvidia-smi topo -m`
- InfiniBand and RDMA details when available
- container/runtime/toolchain versions when available
- a small allowlist of benchmark-relevant environment variables
- a manifest of which commands were present, missing, or failed

## What the collector avoids

- full `env` output
- IP address dumps
- automatic uploads or pushes

## H200 benchmark toolkit

For the 4x H200 host profile captured from `t3ihpc07`, this repository now
includes a local benchmark toolkit:

- `configs/mlperf6-h200-4gpu.env`: editable paths and defaults for the H200 node
- `scripts/bootstrap_mlperf6_h200.sh`: clones or updates the official `mlcommons/training` repo with the required submodule
- `scripts/run_mlperf6_h200.sh`: prints or runs benchmark commands for `llama31`, `llama2_lora`, `gpt_oss20b`, `flux`, or `all`
- `scripts/run_all_mlperf6_h200.sh`: bootstraps, downloads, runs all selected benchmarks, and writes one final report
- `scripts/report_mlperf6_results.py`: summarizes benchmark logs into markdown

Suggested sequence on the H200 node:

```bash
./scripts/bootstrap_mlperf6_h200.sh
./scripts/run_mlperf6_h200.sh show all
./scripts/run_mlperf6_h200.sh --execute run llama31
./scripts/run_mlperf6_h200.sh report generated/mlperf6-h200-report.md
```

Fully automated end-to-end flow:

```bash
./scripts/run_all_mlperf6_h200.sh
```

This script:

- bootstraps the upstream MLCommons training repo
- processes one benchmark at a time: it downloads that benchmark's data, runs
  it, and only then moves on to the next benchmark (download-one, run-one,
  repeat) — so peak disk usage is bounded to one benchmark's assets and results
  land incrementally instead of after every dataset is fetched
- downloads public assets for Llama 3.1, GPT-OSS 20B, and FLUX.1
- reuses anything already downloaded (per-directory completion markers) and
  never deletes downloaded data; `gpt_oss20b` reuses the llama31 C4 corpus via a
  symlink instead of re-downloading it
- uses the official gated Llama 2 70B LoRA downloader when `MLPERF_LLAMA2_RCLONE_CONFIG` is set
- skips a benchmark's run when its own download stage failed
- writes a final markdown report to `MLPERF_FINAL_REPORT_PATH`

### Download footprint

Sizes below are the public download-stage assets pulled by the orchestrator,
measured from the MLCommons R2 manifests. They land under `MLPERF_DATA_ROOT`
(default `/scratch/...`, which on this node has ~321 TB free).

| Benchmark | Download-stage assets | Approx size |
| --- | --- | --- |
| `llama31` | preprocessed C4 corpus (~79 GB) + 8B tokenizer/model (~30 GB) | **~109 GB** |
| `gpt_oss20b` | reuses the llama31 C4 corpus + synced tokenizer | **~0 GB extra** |
| `flux` | CC12M preprocessed (~2.17 TB) + COCO preprocessed (~60 GB) + empty encodings (~2 MB) | **~2.23 TB** |
| `llama2_lora` | 70B model pulled at run time (~128 GB) + dataset (gated, small; smoke-test subset is tiny) | **~128 GB** |

- **The default run executes all four benchmarks**, in the order
  `llama31,gpt_oss20b,llama2_lora,flux`. `flux` (the largest download, ~2.23 TB)
  runs last so the cheaper benchmarks complete first; `llama31` runs before
  `gpt_oss20b` because gpt_oss20b reuses its C4 corpus.
- **Total downloads ≈ 2.46 TB** (~109 GB llama31 + ~0 gpt_oss20b reuse +
  ~128 GB llama2 70B model at run time + ~2.23 TB flux).
- Because the pipeline is download-one/run-one, only one benchmark's assets are
  being fetched at any moment; with the `--skip-runs` flag you can pre-stage all
  data first instead.
- Re-running is cheap: completion markers make already-downloaded datasets skip
  instantly, and nothing on disk is deleted between runs.

Important caveat for Llama 2 70B LoRA:

- MLCommons member-only assets are required
- set `MLPERF_LLAMA2_RCLONE_CONFIG` in `configs/mlperf6-h200-4gpu.env` to the provided `rclone.conf`
- without that file, the pipeline will report the Llama 2 download stage as failed while still generating the final report

Llama 2 LoRA modes:

- `MLPERF_LLAMA2_MODE=official`: uses MLCommons-gated dataset and model, valid for official review workflows
- `MLPERF_LLAMA2_MODE=local-only`: uses a locally present authorized dataset plus the public Hugging Face model mirror
- `MLPERF_LLAMA2_MODE=smoke-test`: uses the public Hugging Face model mirror plus a small public `tau/scrolls` GovReport subset for local debugging only
- `MLPERF_LLAMA2_MODE=skip`: skips the Llama 2 benchmark entirely

For `local-only`, place the local dataset under:

- `${MLPERF_LLAMA2_DATASET_PATH}/${MLPERF_LLAMA2_LOCAL_DATASET_SUBDIR}`

For `smoke-test`, the runner will automatically materialize:

- the public model from `MLPERF_LLAMA2_PUBLIC_MODEL_ID`
- a small parquet dataset under `${MLPERF_LLAMA2_DATASET_PATH}/${MLPERF_LLAMA2_SMOKE_DATASET_SUBDIR}`

The generated run commands are based on the official `mlcommons/training`
repository and adapted for a single node with 4x NVIDIA H200 NVL GPUs.

# MLPerf 6.0 H200 Benchmark Bootstrap

This repository drives an MLPerf 6.0 benchmark run on a Linux server with
4x NVIDIA H200 GPUs. It bootstraps the upstream MLCommons reference code,
downloads each benchmark's assets, runs the benchmarks, and writes one report.

The target host profile (`t3ihpc07`) is already captured in
`configs/mlperf6-h200-4gpu.env`; edit that file to retarget another node.

## Target benchmark areas

- LLM pretraining (small): Llama 3.1 8B on C4
- LLM fine-tuning: Llama 2 70B LoRA on SCROLLS GovReport
- MoE LLM recommendation system (small): GPT-OSS 20B DLRM-DCNv2 on Criteo Multi-Hot
- Text-to-image generation: FLUX.1 on CC12M

## Repository layout

- `configs/mlperf6-h200-4gpu.env`: all paths, image tags, dataset URIs, and run defaults
- `scripts/`: the benchmark toolkit (see "H200 benchmark toolkit" below)
- `generated/`: local report output (git-ignored)
- `tools/powermon/`: standalone Linux power/temperature recorder (CSV + graphs + report); see [tools/powermon/README.md](tools/powermon/README.md)

## Safety notes

- Run output, downloaded data, and reports stay outside version control (`generated/`, `results/`, `work/`, `artifacts/`, `powermon_runs/` are git-ignored).
- Host labels, PCIe layout, package versions, driver versions, and GPU topology can be sensitive. Review anything you publish from this repo.

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

## H200 benchmark toolkit

For the 4x H200 host profile captured from `cluster`, this repository provides
a local benchmark toolkit:

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
  repeat) — so active downloading/running is scoped to a single benchmark at a
  time and results land incrementally instead of after every dataset is fetched.
  This bounds the active working set, not the persistent on-disk footprint:
  downloaded data is retained for reuse and never deleted, so total disk usage
  accumulates across benchmarks
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

Rows are in the default run order (`llama31`, `gpt_oss20b`, `llama2_lora`,
`flux`), not sorted by size.

| Benchmark | Download-stage assets | Approx size |
| --- | --- | --- |
| `llama31` | preprocessed C4 corpus (~79 GB) + 8B tokenizer/model (~30 GB) | **~109 GB** |
| `gpt_oss20b` | reuses the llama31 C4 corpus + synced tokenizer | **~0 GB extra** |
| `llama2_lora` | 70B model pulled at run time (~128 GB) + dataset (gated, small; smoke-test subset is tiny) | **~128 GB** |
| `flux` | CC12M preprocessed (~2.17 TB) + COCO preprocessed (~60 GB) + empty encodings (~2 MB) | **~2.23 TB** |

- **The default run executes all four benchmarks**, in the order
  `llama31,gpt_oss20b,llama2_lora,flux`. `flux` (the largest download, ~2.23 TB)
  runs last so the cheaper benchmarks complete first; `llama31` runs before
  `gpt_oss20b` because gpt_oss20b reuses its C4 corpus.
- **Total downloads ≈ 2.46 TB** (~109 GB llama31 + ~0 gpt_oss20b reuse +
  ~128 GB llama2 70B model at run time + ~2.23 TB flux).
- Because the pipeline is download-one/run-one, only one benchmark's assets are
  being fetched at any moment, so the *active* working set stays small. The
  *persistent* footprint still grows as each benchmark's data is retained for
  reuse — peak disk usage is the sum of everything kept, not one benchmark's
  assets. With the `--skip-runs` flag you can pre-stage all data first instead.
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

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


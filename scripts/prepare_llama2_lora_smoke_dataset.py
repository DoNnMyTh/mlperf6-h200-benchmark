#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def trim_dataset(split, limit: int):
    if limit <= 0:
        return split
    return split.select(range(min(limit, len(split))))


def keep_required_columns(split):
    keep = {"input", "output"}
    drop = [column for column in split.column_names if column not in keep]
    if drop:
        split = split.remove_columns(drop)
    return split


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a public smoke-test dataset for the Llama2 LoRA runner."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_split = keep_required_columns(trim_dataset(dataset["train"], args.train_samples))
    validation_split = keep_required_columns(
        trim_dataset(dataset["validation"], args.validation_samples)
    )

    train_split.to_parquet(str(output_dir / "train-00000-of-00001.parquet"))
    validation_split.to_parquet(str(output_dir / "validation-00000-of-00001.parquet"))

    metadata = {
        "mode": "smoke-test",
        "submission_valid": False,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "train_samples": len(train_split),
        "validation_samples": len(validation_split),
        "note": "Public substitute dataset for local debugging only.",
    }
    (output_dir / "SMOKE_TEST_METADATA.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


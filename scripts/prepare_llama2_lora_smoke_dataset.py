#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from itertools import chain
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def _load_tokenizer(path):
    # The MLPerf fused-qkv model ships a custom config (CustomLlamaConfig) that
    # AutoTokenizer cannot map to a tokenizer class, so load the standard Llama2
    # tokenizer directly (reads tokenizer.json/.model, ignores the model config).
    try:
        from transformers import LlamaTokenizerFast

        return LlamaTokenizerFast.from_pretrained(path)
    except Exception:
        pass
    try:
        from transformers import LlamaTokenizer

        return LlamaTokenizer.from_pretrained(path)
    except Exception:
        pass
    return AutoTokenizer.from_pretrained(path, use_fast=True)


def _build_split(split, limit: int, tokenizer, block_size: int, config: str):
    if limit > 0:
        split = split.select(range(min(limit, len(split))))
    texts = []
    for row in split:
        if "gov_report" in config:
            text = (
                "### Summarize the following text:\n "
                + str(row["input"])
                + "\n ### Summary:\n "
                + str(row["output"])
                + (tokenizer.eos_token or "")
            )
        else:
            text = (
                "### "
                + str(row["input"])
                + "\n ### The answer is:\n "
                + str(row["output"])
                + (tokenizer.eos_token or "")
            )
        texts.append(text)
    input_ids = tokenizer(texts).input_ids
    flat = list(chain(*input_ids))
    total = (len(flat) // block_size) * block_size
    if total == 0:
        raise SystemExit(
            "prep produced fewer than one block of tokens; raise the sample count"
        )
    chunks = [flat[i : i + block_size] for i in range(0, total, block_size)]
    return Dataset.from_dict(
        {"input_ids": chunks, "labels": [list(c) for c in chunks]}
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a public smoke-test dataset for the Llama2 LoRA runner. "
            "train.py loads the parquet straight into the HF Trainer with no "
            "tokenizer, so it must already contain tokenized+packed "
            "input_ids/labels (mirrors the reference create_datasets)."
        )
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="Local model dir holding the Llama2 tokenizer (e.g. the prestaged model).",
    )
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(args.tokenizer_path)
    dataset = load_dataset(
        args.dataset_name, args.dataset_config, trust_remote_code=True
    )
    train_split = _build_split(
        dataset["train"],
        args.train_samples,
        tokenizer,
        args.block_size,
        args.dataset_config,
    )
    validation_split = _build_split(
        dataset["validation"],
        args.validation_samples,
        tokenizer,
        args.block_size,
        args.dataset_config,
    )

    train_split.to_parquet(str(output_dir / "train-00000-of-00001.parquet"))
    validation_split.to_parquet(str(output_dir / "validation-00000-of-00001.parquet"))

    metadata = {
        "mode": "smoke-test",
        "submission_valid": False,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "block_size": args.block_size,
        "train_sequences": len(train_split),
        "validation_sequences": len(validation_split),
        "format": "tokenized+packed input_ids/labels",
        "note": "Public substitute dataset for local debugging only.",
    }
    (output_dir / "SMOKE_TEST_METADATA.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

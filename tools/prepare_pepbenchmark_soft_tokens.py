# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Materialize PepBenchmark rows with ESM-derived soft-token embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

DEFAULT_DATASET_REPO = "jiahuizhang/PepBenchData"
DEFAULT_DATASET_ROOT = "PepBenchData-50"
DEFAULT_TASK = "hemolytic"
DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--split-file", default="hybrid_split.json")
    parser.add_argument("--seed-key", default="seed_0")
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--esm-model", default=DEFAULT_ESM_MODEL)
    parser.add_argument("--num-soft-tokens", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2688)
    parser.add_argument("--projection-seed", type=int, default=1234)
    parser.add_argument("--projection-scale", type=float, default=0.01)
    parser.add_argument(
        "--label-format",
        choices=("binary", "choice"),
        default="binary",
        help="Use raw 0/1 answers or A/B answers for the multichoice verifier.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_single_column_csv(path: str) -> list[str]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or len(reader.fieldnames) != 1:
            raise ValueError(f"Expected one-column CSV at {path}")
        key = reader.fieldnames[0]
        return [row[key].strip() for row in reader]


def load_task_data(
    dataset_repo: str, dataset_root: str, task: str, split_file: str
) -> tuple[list[str], list[str], dict[str, Any]]:
    task_root = f"{dataset_root}/{task}"
    fasta_path = hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        filename=f"{task_root}/fasta.csv",
    )
    label_path = hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        filename=f"{task_root}/label.csv",
    )
    split_path = hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        filename=f"{task_root}/{split_file}",
    )

    sequences = read_single_column_csv(fasta_path)
    labels = read_single_column_csv(label_path)
    if len(sequences) != len(labels):
        raise ValueError(
            f"Sequence count {len(sequences)} does not match label count {len(labels)}"
        )

    with open(split_path) as f:
        split_doc = json.load(f)

    return sequences, labels, split_doc


def get_split_indices(
    split_doc: dict[str, Any], seed_key: str, split_name: str
) -> list[int]:
    if seed_key not in split_doc:
        raise ValueError(f"Missing split seed {seed_key}")
    seed_splits = split_doc[seed_key]
    if split_name not in seed_splits:
        raise ValueError(f"Missing split {split_name} for seed {seed_key}")
    return [int(idx) for idx in seed_splits[split_name]]


def balanced_sample_indices(
    candidate_indices: list[int],
    labels: list[str],
    num_samples: int,
    seed: int,
) -> list[int]:
    if num_samples <= 0:
        return []

    rng = random.Random(seed)
    buckets: dict[str, list[int]] = {}
    for idx in candidate_indices:
        buckets.setdefault(labels[idx], []).append(idx)
    if not buckets:
        return []

    for bucket in buckets.values():
        rng.shuffle(bucket)

    label_order = sorted(buckets)
    base_count = num_samples // len(label_order)
    remainder = num_samples % len(label_order)

    selected: list[int] = []
    for offset, label in enumerate(label_order):
        take = base_count + int(offset < remainder)
        selected.extend(buckets[label][:take])

    if len(selected) < num_samples:
        selected_set = set(selected)
        leftovers = [
            idx
            for label in label_order
            for idx in buckets[label]
            if idx not in selected_set
        ]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: num_samples - len(selected)])

    rng.shuffle(selected)
    return selected[:num_samples]


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false")
    return torch.device(requested_device)


def make_projection(
    esm_hidden_size: int,
    hidden_size: int,
    projection_seed: int,
    projection_scale: float,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(projection_seed)
    projection = torch.randn(esm_hidden_size, hidden_size, generator=generator)
    return projection * (projection_scale / math.sqrt(esm_hidden_size))


def extract_residue_hidden(
    sequence_hidden: torch.Tensor, attention_mask: torch.Tensor, sequence_length: int
) -> torch.Tensor:
    valid_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if len(valid_positions) >= sequence_length + 2:
        residue_positions = valid_positions[1 : sequence_length + 1]
    else:
        residue_positions = valid_positions[:sequence_length]
    if len(residue_positions) == 0:
        raise ValueError("Could not find residue token positions for an empty sequence")
    return sequence_hidden[residue_positions]


def pool_residue_hidden(
    residue_hidden: torch.Tensor, num_soft_tokens: int
) -> torch.Tensor:
    if residue_hidden.shape[0] == num_soft_tokens:
        return residue_hidden
    pooled = F.adaptive_avg_pool1d(
        residue_hidden.T.unsqueeze(0), output_size=num_soft_tokens
    )
    return pooled.squeeze(0).T.contiguous()


def format_answer(label: str, label_format: str) -> str:
    if label_format == "binary":
        return label
    if label_format == "choice":
        if label == "0":
            return "A"
        if label == "1":
            return "B"
        raise ValueError(f"choice label format expects 0/1 labels, got {label}")
    raise ValueError(f"Unknown label format {label_format}")


def build_prompt(task: str, sequence: str, label_format: str) -> str:
    if task == "hemolytic":
        question = "Is the following peptide sequence hemolytic?"
        positive = "hemolytic"
        negative = "non-hemolytic"
    else:
        question = f"Does the following peptide sequence satisfy task {task}?"
        positive = "positive"
        negative = "negative"
    if label_format == "binary":
        return (
            f"{question}\n"
            f"Sequence: {sequence}\n\n"
            f"Answer with exactly 1 for {positive} or 0 for {negative}."
        )
    if label_format == "choice":
        return (
            f"{question}\n"
            f"Sequence: {sequence}\n\n"
            f"Options:\n"
            f"A. {negative}\n"
            f"B. {positive}\n\n"
            f"Reply exactly as 'Answer: A' or 'Answer: B'."
        )
    raise ValueError(f"Unknown label format {label_format}")


def encode_soft_tokens(
    sequences: list[str],
    model_name: str,
    num_soft_tokens: int,
    hidden_size: int,
    projection_seed: int,
    projection_scale: float,
    batch_size: int,
    device: torch.device,
) -> list[list[list[float]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    esm_hidden_size = int(model.config.hidden_size)
    projection = make_projection(
        esm_hidden_size=esm_hidden_size,
        hidden_size=hidden_size,
        projection_seed=projection_seed,
        projection_scale=projection_scale,
    )

    all_soft_tokens: list[list[list[float]]] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        encoded = tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded_on_device = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded_on_device)

        hidden = outputs.last_hidden_state.detach().float().cpu()
        attention_mask = encoded["attention_mask"].cpu()
        for row_idx, sequence in enumerate(batch_sequences):
            residue_hidden = extract_residue_hidden(
                sequence_hidden=hidden[row_idx],
                attention_mask=attention_mask[row_idx],
                sequence_length=len(sequence),
            )
            pooled = pool_residue_hidden(residue_hidden, num_soft_tokens)
            soft_tokens = pooled @ projection
            all_soft_tokens.append(soft_tokens.tolist())

    return all_soft_tokens


def make_rows(
    indices: list[int],
    split_name: str,
    task: str,
    sequences: list[str],
    labels: list[str],
    soft_tokens: list[list[list[float]]],
    esm_model: str,
    label_format: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, soft_token_embeddings in zip(indices, soft_tokens):
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": build_prompt(task, sequences[idx], label_format),
                    },
                    {
                        "role": "assistant",
                        "content": format_answer(labels[idx], label_format),
                    },
                ],
                "soft_token_embeddings": soft_token_embeddings,
                "metadata": {
                    "pepbenchmark_index": idx,
                    "task": task,
                    "split": split_name,
                    "label": labels[idx],
                    "answer": format_answer(labels[idx], label_format),
                    "label_format": label_format,
                    "sequence": sequences[idx],
                    "esm_model": esm_model,
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sequences, labels, split_doc = load_task_data(
        dataset_repo=args.dataset_repo,
        dataset_root=args.dataset_root,
        task=args.task,
        split_file=args.split_file,
    )
    train_indices = balanced_sample_indices(
        get_split_indices(split_doc, args.seed_key, "train"),
        labels,
        args.train_samples,
        args.sample_seed,
    )
    validation_indices = balanced_sample_indices(
        get_split_indices(split_doc, args.seed_key, "valid"),
        labels,
        args.validation_samples,
        args.sample_seed + 1,
    )

    device = choose_device(args.device)
    selected_indices = train_indices + validation_indices
    selected_sequences = [sequences[idx] for idx in selected_indices]
    soft_tokens = encode_soft_tokens(
        sequences=selected_sequences,
        model_name=args.esm_model,
        num_soft_tokens=args.num_soft_tokens,
        hidden_size=args.hidden_size,
        projection_seed=args.projection_seed,
        projection_scale=args.projection_scale,
        batch_size=args.batch_size,
        device=device,
    )

    train_soft_tokens = soft_tokens[: len(train_indices)]
    validation_soft_tokens = soft_tokens[len(train_indices) :]
    train_rows = make_rows(
        train_indices,
        "train",
        args.task,
        sequences,
        labels,
        train_soft_tokens,
        args.esm_model,
        args.label_format,
    )
    validation_rows = make_rows(
        validation_indices,
        "validation",
        args.task,
        sequences,
        labels,
        validation_soft_tokens,
        args.esm_model,
        args.label_format,
    )

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    manifest = {
        "dataset_repo": args.dataset_repo,
        "dataset_root": args.dataset_root,
        "task": args.task,
        "split_file": args.split_file,
        "seed_key": args.seed_key,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "esm_model": args.esm_model,
        "num_soft_tokens": args.num_soft_tokens,
        "hidden_size": args.hidden_size,
        "projection_seed": args.projection_seed,
        "projection_scale": args.projection_scale,
        "label_format": args.label_format,
        "device": str(device),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
    }
    with (args.output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {len(train_rows)} train rows to {train_path}")
    print(f"Wrote {len(validation_rows)} validation rows to {validation_path}")


if __name__ == "__main__":
    main()

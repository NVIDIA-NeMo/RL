#!/usr/bin/env python3
"""Dataset loading helpers for standalone math checkpoint evaluation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def load_gsm8k(num_samples: int, seed: int) -> list[dict[str, str]]:
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="test")
    samples = [
        {
            "question": row["question"],
            "gold": extract_hash_answer(row["answer"]) or row["answer"].strip(),
            "raw_answer": row["answer"],
        }
        for row in rows
    ]
    return subsample(samples, num_samples, seed)


def normalize_benchmark(name: str) -> str:
    aliases = {
        "gsm8k": "gsm8k",
        "aime24": "aime2024",
        "aime2024": "aime2024",
        "aime25": "aime2025",
        "aime2025": "aime2025",
    }
    try:
        return aliases[name.lower()]
    except KeyError as e:
        raise ValueError(f"Unsupported benchmark: {name}") from e


def load_aime(variant: str, num_samples: int, seed: int) -> list[dict[str, str]]:
    nemo_samples = load_nemo_skills_aime(variant)
    if nemo_samples is not None:
        samples = nemo_samples
        print(f"Loaded AIME {variant} from Nemo Skills dataset")
    else:
        samples = load_hf_aime(variant)
        print(f"Loaded AIME {variant} from Hugging Face fallback dataset")

    return subsample(samples, num_samples, seed)


def load_nemo_skills_aime(variant: str) -> list[dict[str, str]] | None:
    dataset_name = {"2024": "aime24", "2025": "aime25"}.get(variant)
    if dataset_name is None:
        raise ValueError(f"Unsupported AIME variant: {variant}")

    local_data_path = (
        Path(__file__).resolve().parent / "data" / f"nemo_skills_{dataset_name}_test.jsonl"
    )
    if local_data_path.exists():
        return load_nemo_skills_aime_jsonl(local_data_path, dataset_name)

    explicit_data_dir = os.environ.get("NEMO_SKILLS_AIME_DATA_DIR")
    if explicit_data_dir:
        explicit_data_path = Path(explicit_data_dir) / f"{dataset_name}" / "test.jsonl"
        if explicit_data_path.exists():
            return load_nemo_skills_aime_jsonl(explicit_data_path, dataset_name)

    packaged_data_path = (
        Path("/opt/nemo_rl_venv/lib/python3.12/site-packages")
        / "nemo_skills"
        / "dataset"
        / dataset_name
        / "test.jsonl"
    )
    if packaged_data_path.exists():
        return load_nemo_skills_aime_jsonl(packaged_data_path, dataset_name)

    try:
        import nemo_skills
    except ImportError:
        return None

    dataset_path = Path(nemo_skills.__file__).resolve().parent / "dataset" / dataset_name / "test.jsonl"
    if not dataset_path.exists():
        return None

    return load_nemo_skills_aime_jsonl(dataset_path, dataset_name)


def load_nemo_skills_aime_jsonl(dataset_path: Path, dataset_name: str) -> list[dict[str, str]]:
    print(f"Using Nemo Skills AIME dataset file: {dataset_path}")
    samples: list[dict[str, str]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for row_id, line in enumerate(f):
            row = json.loads(line)
            question = row.get("problem") or row.get("question")
            answer = row.get("expected_answer") or row.get("answer")
            if question is None or answer is None:
                raise ValueError(f"Malformed Nemo Skills row {row_id} in {dataset_path}")
            samples.append(
                {
                    "question": str(question),
                    "gold": str(answer).strip(),
                    "raw_answer": str(row.get("reference_solution") or answer),
                    "source_id": str(row.get("id") or f"{dataset_name}-{row_id}"),
                }
            )
    return samples


def load_hf_aime(variant: str) -> list[dict[str, str]]:
    from datasets import concatenate_datasets, load_dataset

    if variant == "2024":
        rows = load_dataset("HuggingFaceH4/aime_2024", split="train")
        question_key = "problem"
    elif variant == "2025":
        rows_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        rows_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        rows = concatenate_datasets([rows_i, rows_ii])
        question_key = "question"
    else:
        raise ValueError(f"Unsupported AIME variant: {variant}")

    return [
        {
            "question": str(row[question_key]),
            "gold": str(row["answer"]).strip(),
            "raw_answer": str(row["answer"]),
        }
        for row in rows
    ]


def load_benchmark_samples(benchmark: str, num_samples: int, seed: int) -> list[dict[str, str]]:
    normalized = normalize_benchmark(benchmark)
    if normalized == "gsm8k":
        return load_gsm8k(num_samples, seed)
    if normalized == "aime2024":
        return load_aime("2024", num_samples, seed)
    if normalized == "aime2025":
        return load_aime("2025", num_samples, seed)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def subsample(
    samples: list[dict[str, str]], num_samples: int, seed: int
) -> list[dict[str, str]]:
    if not (0 < num_samples < len(samples)):
        return samples

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(samples)), num_samples))
    return [samples[i] for i in indices]

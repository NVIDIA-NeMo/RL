#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Convert R2E-Gym/R2E-Gym-Subset parquet shards to swe_agents JSONL.

The swe_agents R2E path expects a Gym row whose metadata contains the original
R2E instance serialized as ``instance_dict``. At rollout time the agent server
writes that instance to ``/root/dataset/data.jsonl`` and invokes the R2E local
evaluation harness with the generated ``instance_id``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DATASET_NAME = "R2E-Gym/R2E-Gym-Subset"
DEFAULT_SPLIT = "train"
DEFAULT_AGENT_REF_NAME = "swe_agents_train"
DEFAULT_MODEL = "default"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_CONTAINER_IMAGE_DIR = "/swe-bench-images"
DEFAULT_REPO_FORMATTER = "/swe-bench-repos/{repo}"
OUTPUT_JSONL_FILENAME = "r2e_gym_subset_full.jsonl"
METRICS_JSON_FILENAME = "r2e_gym_subset_full_conversion_metrics.json"
BASE_COMMIT_CACHE_FILENAME = "r2e_gym_subset_train_base_commits.json"
TRAIN_JSONL_FILENAME = "benchmark_r2e_gym_easy_train.jsonl"
VAL_JSONL_FILENAME = "benchmark_r2e_gym_easy_val.jsonl"

REPO_NAME_MAP = {
    "pandas": "pandas-dev/pandas",
    "numpy": "numpy/numpy",
    "pillow": "python-pillow/Pillow",
    "orange3": "biolab/orange3",
    "datalad": "datalad/datalad",
    "coveragepy": "nedbat/coveragepy",
    "aiohttp": "aio-libs/aiohttp",
    "tornado": "tornadoweb/tornado",
    "pyramid": "pylons/pyramid",
    "scrapy": "scrapy/scrapy",
}

R2E_COLUMNS = [
    "repo_name",
    "docker_image",
    "commit_hash",
    "parsed_commit_content",
    "execution_result_content",
    "modified_files",
    "modified_entity_summaries",
    "relevant_files",
    "num_non_test_files",
    "num_non_test_func_methods",
    "num_non_test_lines",
    "prompt",
    "problem_statement",
    "expected_output_json",
]

COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{40}$")
INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9a-f]{40}$")


@dataclass(frozen=True)
class ConversionPaths:
    output_jsonl: Path
    metrics_json: Path
    base_commit_cache_json: Path
    repo_cache_dir: Path
    train_jsonl: Path
    val_jsonl: Path


@dataclass(frozen=True)
class SplitStats:
    train_rows: int = 0
    val_rows: int = 0
    neither_rows: int = 0
    train_ids: int = 0
    val_ids: int = 0


@dataclass
class ConversionStats:
    rows: int = 0
    repo_counts: Counter[str] = field(default_factory=Counter)
    expected_status_counts: Counter[str] = field(default_factory=Counter)
    max_expected_tests: int = 0
    min_expected_tests: int | None = None
    duplicate_instance_ids: list[str] = field(default_factory=list)
    seen_instance_ids: set[str] = field(default_factory=set)

    def observe(self, source_row: dict[str, Any], instance_id: str) -> None:
        self.rows += 1
        self.repo_counts[source_row["repo_name"]] += 1

        if instance_id in self.seen_instance_ids:
            self.duplicate_instance_ids.append(instance_id)
        self.seen_instance_ids.add(instance_id)

        expected = parse_json_object(source_row["expected_output_json"], "expected_output_json")
        expected_count = len(expected)
        self.max_expected_tests = max(self.max_expected_tests, expected_count)
        self.min_expected_tests = (
            expected_count if self.min_expected_tests is None else min(self.min_expected_tests, expected_count)
        )
        self.expected_status_counts.update(str(status) for status in expected.values())

    def to_json(self, output_jsonl: Path) -> dict[str, Any]:
        return {
            "dataset_name": DATASET_NAME,
            "output_jsonl": str(output_jsonl),
            "rows": self.rows,
            "repo_counts": dict(self.repo_counts.most_common()),
            "expected_status_counts": dict(self.expected_status_counts.most_common()),
            "min_expected_tests": self.min_expected_tests,
            "max_expected_tests": self.max_expected_tests,
            "duplicate_instance_ids": self.duplicate_instance_ids,
        }


def parse_json_object(value: str, field_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to an object, got {type(parsed).__name__}")
    return parsed


def json_dumps(value: Any) -> str:
    return json.dumps(value)


def build_conversion_paths(output_dir: Path, cache_dir: Path) -> ConversionPaths:
    return ConversionPaths(
        output_jsonl=output_dir / OUTPUT_JSONL_FILENAME,
        metrics_json=output_dir / METRICS_JSON_FILENAME,
        base_commit_cache_json=cache_dir / BASE_COMMIT_CACHE_FILENAME,
        repo_cache_dir=cache_dir,
        train_jsonl=output_dir / TRAIN_JSONL_FILENAME,
        val_jsonl=output_dir / VAL_JSONL_FILENAME,
    )


def repo_from_repo_name(repo_name: str) -> str:
    try:
        return REPO_NAME_MAP[repo_name]
    except KeyError as exc:
        raise ValueError(f"No GitHub repo mapping configured for repo_name {repo_name!r}") from exc


def git_repo_from_repo_name(repo_name: str) -> str:
    return repo_from_repo_name(repo_name)


def build_instance_id(repo_name: str, commit_hash: str) -> str:
    if not COMMIT_HASH_RE.fullmatch(commit_hash):
        raise ValueError(f"commit_hash {commit_hash!r} is not a 40-character lowercase SHA-1")

    owner, repo = repo_from_repo_name(repo_name).split("/", 1)
    instance_id = f"{owner}__{repo}-{commit_hash}"
    if not INSTANCE_ID_RE.fullmatch(instance_id):
        raise ValueError(f"Generated invalid instance_id: {instance_id}")
    return instance_id


def expected_container_id(instance_id: str) -> str:
    return re.sub(r"[^_]+__([^-]+)-", lambda match: match.group(1).lower() + "_final_", instance_id)


def derive_base_commit(parsed_commit_content: dict[str, Any]) -> str | None:
    old_commit_hash = parsed_commit_content.get("old_commit_hash")
    if not isinstance(old_commit_hash, str) or not old_commit_hash:
        return None
    return old_commit_hash.removesuffix("^")


def commit_created_at(parsed_commit_content: dict[str, Any]) -> str:
    value = parsed_commit_content.get("commit_date", "")
    return value if isinstance(value, str) else ""


def build_container_formatter(docker_image: str, container_image_dir: str) -> str:
    image_name = docker_image.replace("/", "_").replace(":", "_")
    return f"{container_image_dir.rstrip('/')}/{image_name}.sif"


class BaseCommitResolver:
    def __init__(self, cache_json: Path | None = None, repo_cache_dir: Path | None = None) -> None:
        self.cache_json = cache_json
        self.repo_cache_dir = repo_cache_dir
        self.cache: dict[str, str] = {}
        self.dirty = False

        if cache_json and cache_json.exists():
            loaded = json.loads(cache_json.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"Base commit cache must be a JSON object: {cache_json}")
            self.cache = {str(k): str(v) for k, v in loaded.items()}

    def resolve(self, repo_name: str, commit_hash: str) -> str | None:
        if commit_hash in self.cache:
            return self.cache[commit_hash]
        if self.repo_cache_dir is None:
            return None

        repo_path = self._ensure_repo(repo_name)
        parent = self._parent_from_repo(repo_path, commit_hash)
        self.cache[commit_hash] = parent
        self.dirty = True
        return parent

    def save(self) -> None:
        if not self.cache_json or not self.dirty:
            return
        self.cache_json.parent.mkdir(parents=True, exist_ok=True)
        self.cache_json.write_text(json.dumps(dict(sorted(self.cache.items())), indent=2) + "\n", encoding="utf-8")

    def _ensure_repo(self, repo_name: str) -> Path:
        assert self.repo_cache_dir is not None
        repo = git_repo_from_repo_name(repo_name)
        repo_path = self.repo_cache_dir / f"{repo.replace('/', '__')}.git"
        if repo_path.exists():
            return repo_path

        repo_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--bare",
                f"https://github.com/{repo}.git",
                str(repo_path),
            ],
            check=True,
        )
        return repo_path

    def _parent_from_repo(self, repo_path: Path, commit_hash: str) -> str:
        exists = subprocess.run(
            ["git", "-C", str(repo_path), "cat-file", "-e", f"{commit_hash}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if exists.returncode != 0:
            subprocess.run(
                ["git", "-C", str(repo_path), "fetch", "--filter=blob:none", "origin", commit_hash],
                check=True,
            )

        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-list", "--parents", "-n", "1", commit_hash],
            check=True,
            capture_output=True,
            text=True,
        )
        parts = result.stdout.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Commit {commit_hash} in {repo_path} has no parent")
        return parts[1]


def validate_source_row(source_row: dict[str, Any], row_number: int) -> None:
    missing = [column for column in R2E_COLUMNS if column not in source_row]
    if missing:
        raise ValueError(f"Row {row_number}: missing columns {missing}")

    for column in R2E_COLUMNS:
        if source_row[column] is None:
            raise ValueError(f"Row {row_number}: column {column!r} is null")

    parsed_commit = parse_json_object(source_row["parsed_commit_content"], "parsed_commit_content")
    execution_result = parse_json_object(source_row["execution_result_content"], "execution_result_content")
    parse_json_object(source_row["expected_output_json"], "expected_output_json")

    commit_hash = source_row["commit_hash"]
    if parsed_commit.get("new_commit_hash") != commit_hash:
        raise ValueError(
            f"Row {row_number}: parsed_commit_content.new_commit_hash "
            f"{parsed_commit.get('new_commit_hash')!r} does not match commit_hash {commit_hash!r}"
        )
    if execution_result.get("new_commit_hash") != commit_hash:
        raise ValueError(
            f"Row {row_number}: execution_result_content.new_commit_hash "
            f"{execution_result.get('new_commit_hash')!r} does not match commit_hash {commit_hash!r}"
        )

    repo_name = source_row["repo_name"]
    docker_image = source_row["docker_image"]
    expected_suffix = f"/{repo_name}_final:{commit_hash}"
    if not docker_image.endswith(expected_suffix):
        raise ValueError(
            f"Row {row_number}: docker_image {docker_image!r} does not end with expected suffix {expected_suffix!r}"
        )


def build_instance_dict(
    source_row: dict[str, Any],
    instance_id: str,
    split: str,
    repo: str,
    base_commit: str | None,
    row_index: int,
    container_image_dir: str,
    repo_formatter: str,
    parsed_commit_content: dict[str, Any],
) -> dict[str, Any]:
    commit_hash = source_row["commit_hash"]
    instance_dict: dict[str, Any] = {
        "FAIL_TO_PASS": [],
        "PASS_TO_PASS": [],
        "created_at": commit_created_at(parsed_commit_content),
        "hints_text": "",
        "version": "",
        "instance_id": instance_id,
        "base_commit": base_commit,
        "repo": repo,
        "problem_statement": source_row["problem_statement"],
    }
    for column in R2E_COLUMNS:
        if column != "problem_statement":
            instance_dict[column] = source_row[column]

    instance_dict["container_formatter"] = build_container_formatter(source_row["docker_image"], container_image_dir)
    instance_dict["container_id"] = row_index
    instance_dict["dataset_name"] = DATASET_NAME
    instance_dict["split"] = split
    instance_dict["resolved_commit"] = commit_hash
    instance_dict["repo_formatter"] = repo_formatter
    return instance_dict


def convert_row(
    source_row: dict[str, Any],
    *,
    row_number: int,
    row_index: int,
    split: str,
    model: str,
    temperature: float,
    top_p: float,
    agent_ref_name: str | None,
    base_commit: str | None,
    container_image_dir: str,
    repo_formatter: str,
) -> dict[str, Any]:
    validate_source_row(source_row, row_number)

    repo_name = source_row["repo_name"]
    commit_hash = source_row["commit_hash"]
    instance_id = build_instance_id(repo_name, commit_hash)
    repo = repo_from_repo_name(repo_name)
    parsed_commit_content = parse_json_object(source_row["parsed_commit_content"], "parsed_commit_content")
    if base_commit is None:
        base_commit = derive_base_commit(parsed_commit_content)
    created_at = commit_created_at(parsed_commit_content)
    instance_dict = build_instance_dict(
        source_row,
        instance_id,
        split,
        repo,
        base_commit,
        row_index,
        container_image_dir,
        repo_formatter,
        parsed_commit_content,
    )

    row: dict[str, Any] = {}
    row["responses_create_params"] = {
        "input": [],
        "metadata": {
            "instance_id": instance_id,
            "base_commit": base_commit,
            "dataset_name": DATASET_NAME,
            "split": split,
            "problem_statement": source_row["problem_statement"],
            "hints_text": "",
            "version": "",
            "created_at": created_at,
            "instance_dict": json_dumps(instance_dict),
            "repo": repo,
        },
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
    }

    if agent_ref_name:
        row["agent_ref"] = {"type": "responses_api_agents", "name": agent_ref_name}

    return row


def iter_parquet_rows(dataset_dir: Path, batch_size: int) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the R2E-Gym parquet shards") from exc

    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No train-*.parquet shards found under {data_dir}")

    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(columns=R2E_COLUMNS, batch_size=batch_size):
            names = batch.schema.names
            columns = [batch.column(index).to_pylist() for index in range(len(names))]
            for values in zip(*columns):
                yield dict(zip(names, values))


def load_instance_ids(ids_path: Path, label: str) -> set[str]:
    instance_ids: set[str] = set()
    duplicates: list[str] = []

    with ids_path.open(encoding="utf-8") as ids_file:
        for line_number, line in enumerate(ids_file, start=1):
            instance_id = line.strip()
            if not instance_id:
                continue
            if not INSTANCE_ID_RE.fullmatch(instance_id):
                raise ValueError(
                    f"{label} ids file {ids_path} line {line_number}: invalid instance_id {instance_id!r}"
                )
            if instance_id in instance_ids:
                duplicates.append(instance_id)
            instance_ids.add(instance_id)

    if duplicates:
        raise ValueError(f"{label} ids file {ids_path} contains duplicate instance_ids: {duplicates[:5]}")
    return instance_ids


def converted_row_instance_id(row: dict[str, Any], row_number: int) -> str:
    try:
        instance_id = row["responses_create_params"]["metadata"]["instance_id"]
    except KeyError as exc:
        raise ValueError(f"Full JSONL row {row_number}: missing metadata.instance_id") from exc
    if not isinstance(instance_id, str):
        raise ValueError(f"Full JSONL row {row_number}: metadata.instance_id must be a string")
    return instance_id


def split_jsonl_by_instance_ids(
    *,
    full_jsonl: Path,
    train_ids_path: Path,
    val_ids_path: Path,
    train_jsonl: Path,
    val_jsonl: Path,
) -> SplitStats:
    train_ids = load_instance_ids(train_ids_path, "train")
    val_ids = load_instance_ids(val_ids_path, "val")
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(f"Train and val ids overlap: {overlap[:5]}")

    train_tmp = train_jsonl.with_suffix(train_jsonl.suffix + ".tmp")
    val_tmp = val_jsonl.with_suffix(val_jsonl.suffix + ".tmp")
    train_found: set[str] = set()
    val_found: set[str] = set()
    train_rows = 0
    val_rows = 0
    neither_rows = 0

    try:
        train_jsonl.parent.mkdir(parents=True, exist_ok=True)
        val_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with (
            full_jsonl.open(encoding="utf-8") as full_file,
            train_tmp.open("w", encoding="utf-8") as train_file,
            val_tmp.open("w", encoding="utf-8") as val_file,
        ):
            for row_number, line in enumerate(full_file, start=1):
                row = json.loads(line)
                instance_id = converted_row_instance_id(row, row_number)
                if instance_id in train_ids:
                    row["agent_ref"]["name"] = "swe_agents_train"
                    train_file.write(json.dumps(row, separators=(",", ":")))
                    train_file.write("\n")
                    train_found.add(instance_id)
                    train_rows += 1
                elif instance_id in val_ids:
                    row["agent_ref"]["name"] = "swe_agents_val"
                    val_file.write(json.dumps(row, separators=(",", ":")))
                    val_file.write("\n")
                    val_found.add(instance_id)
                    val_rows += 1
                else:
                    neither_rows += 1

        missing_train = sorted(train_ids - train_found)
        missing_val = sorted(val_ids - val_found)
        if missing_train or missing_val:
            raise ValueError(
                "Split ids were not found in the full JSONL: "
                f"missing_train={missing_train[:5]}, missing_val={missing_val[:5]}"
            )

        train_tmp.replace(train_jsonl)
        val_tmp.replace(val_jsonl)
    except Exception:
        train_tmp.unlink(missing_ok=True)
        val_tmp.unlink(missing_ok=True)
        raise

    return SplitStats(
        train_rows=train_rows,
        val_rows=val_rows,
        neither_rows=neither_rows,
        train_ids=len(train_ids),
        val_ids=len(val_ids),
    )


def validate_converted_row(row: dict[str, Any], row_number: int) -> None:
    allowed_top_keys = {"responses_create_params", "agent_ref"}
    unexpected_top_keys = set(row) - allowed_top_keys
    if unexpected_top_keys:
        raise ValueError(f"Converted row {row_number}: unexpected top-level keys {sorted(unexpected_top_keys)}")

    params = row.get("responses_create_params")
    if not isinstance(params, dict):
        raise ValueError(f"Converted row {row_number}: responses_create_params must be an object")
    if params.get("input") != []:
        raise ValueError(f"Converted row {row_number}: responses_create_params.input must be [] for swe_agents")
    if "max_output_tokens" in params:
        raise ValueError(f"Converted row {row_number}: max_output_tokens should be omitted for MLPerf R2E format")

    metadata = params.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Converted row {row_number}: responses_create_params.metadata must be an object")

    expected_metadata_keys = {
        "instance_id",
        "base_commit",
        "dataset_name",
        "split",
        "problem_statement",
        "hints_text",
        "version",
        "created_at",
        "instance_dict",
        "repo",
    }
    if set(metadata) != expected_metadata_keys:
        raise ValueError(
            f"Converted row {row_number}: metadata keys {sorted(metadata)} "
            f"do not match expected keys {sorted(expected_metadata_keys)}"
        )

    for key in expected_metadata_keys:
        if key not in metadata:
            raise ValueError(f"Converted row {row_number}: metadata missing {key!r}")

    if "R2E-Gym" not in metadata["dataset_name"]:
        raise ValueError(f"Converted row {row_number}: dataset_name must contain 'R2E-Gym'")

    instance_dict = parse_json_object(metadata["instance_dict"], "metadata.instance_dict")
    if instance_dict.get("instance_id") != metadata["instance_id"]:
        raise ValueError(f"Converted row {row_number}: instance_dict.instance_id does not match metadata.instance_id")
    if instance_dict.get("base_commit") != metadata["base_commit"]:
        raise ValueError(f"Converted row {row_number}: instance_dict.base_commit does not match metadata.base_commit")
    if instance_dict.get("repo") != metadata["repo"]:
        raise ValueError(f"Converted row {row_number}: instance_dict.repo does not match metadata.repo")
    if not isinstance(instance_dict.get("container_id"), int):
        raise ValueError(f"Converted row {row_number}: instance_dict.container_id must be an integer")
    if instance_dict.get("FAIL_TO_PASS") != [] or instance_dict.get("PASS_TO_PASS") != []:
        raise ValueError(f"Converted row {row_number}: FAIL_TO_PASS and PASS_TO_PASS must be empty lists")


def convert_dataset(
    *,
    dataset_dir: Path,
    output_jsonl: Path,
    metrics_json: Path | None,
    base_commit_cache_json: Path | None,
    repo_cache_dir: Path | None,
    split: str,
    model: str,
    temperature: float,
    top_p: float,
    agent_ref_name: str | None,
    container_image_dir: str,
    repo_formatter: str,
    batch_size: int,
    limit: int | None,
) -> ConversionStats:
    stats = ConversionStats()
    base_commit_resolver = BaseCommitResolver(cache_json=base_commit_cache_json, repo_cache_dir=repo_cache_dir)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as output_file:
        for row_index, source_row in enumerate(iter_parquet_rows(dataset_dir, batch_size)):
            if limit is not None and row_index >= limit:
                break

            row_number = row_index + 1
            base_commit = base_commit_resolver.resolve(source_row["repo_name"], source_row["commit_hash"])
            converted = convert_row(
                source_row,
                row_number=row_number,
                row_index=row_index,
                split=split,
                model=model,
                temperature=temperature,
                top_p=top_p,
                agent_ref_name=agent_ref_name,
                base_commit=base_commit,
                container_image_dir=container_image_dir,
                repo_formatter=repo_formatter,
            )
            validate_converted_row(converted, row_number)
            stats.observe(source_row, converted["responses_create_params"]["metadata"]["instance_id"])
            output_file.write(json_dumps(converted) + "\n")

    if stats.duplicate_instance_ids:
        raise ValueError(f"Generated duplicate instance_ids: {stats.duplicate_instance_ids[:5]}")

    base_commit_resolver.save()

    if metrics_json:
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps(stats.to_json(output_jsonl), indent=2) + "\n", encoding="utf-8")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("~/Datasets/R2E-Gym__R2E-Gym-Subset").expanduser(),
        help="Local R2E-Gym__R2E-Gym-Subset directory containing data/train-*.parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            f"Destination directory for {OUTPUT_JSONL_FILENAME} and "
            f"{METRICS_JSON_FILENAME}."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help=(
            f"Cache directory for {BASE_COMMIT_CACHE_FILENAME} and blobless bare "
            "GitHub repo clones used to resolve true parent SHAs."
        ),
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Split value to put in metadata.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="responses_create_params.model value.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--container-image-dir",
        default=DEFAULT_CONTAINER_IMAGE_DIR,
        help="Directory prefix used for instance_dict.container_formatter.",
    )
    parser.add_argument(
        "--repo-formatter",
        default=DEFAULT_REPO_FORMATTER,
        help="Value for instance_dict.repo_formatter.",
    )
    parser.add_argument(
        "--agent-ref-name",
        default=DEFAULT_AGENT_REF_NAME,
        help="Top-level agent_ref name. Use --no-agent-ref to omit it.",
    )
    parser.add_argument("--no-agent-ref", action="store_true", help="Do not emit top-level agent_ref.")
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=None,
        help=f"Optional one-instance-id-per-line file used to create {TRAIN_JSONL_FILENAME}.",
    )
    parser.add_argument(
        "--val-ids",
        type=Path,
        default=None,
        help=f"Optional one-instance-id-per-line file used to create {VAL_JSONL_FILENAME}.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Parquet streaming batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows for smoke tests.")
    args = parser.parse_args()
    if (args.train_ids is None) != (args.val_ids is None):
        parser.error("--train-ids and --val-ids must be provided together")
    return args


def main() -> None:
    args = parse_args()
    agent_ref_name = None if args.no_agent_ref else args.agent_ref_name
    paths = build_conversion_paths(args.output_dir.expanduser(), args.cache_dir.expanduser())

    stats = convert_dataset(
        dataset_dir=args.dataset_dir.expanduser(),
        output_jsonl=paths.output_jsonl,
        metrics_json=paths.metrics_json,
        base_commit_cache_json=paths.base_commit_cache_json,
        repo_cache_dir=paths.repo_cache_dir,
        split=args.split,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        agent_ref_name=agent_ref_name,
        container_image_dir=args.container_image_dir,
        repo_formatter=args.repo_formatter,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(
        f"Wrote {stats.rows} rows to {paths.output_jsonl}. "
        f"Repos: {dict(stats.repo_counts.most_common())}. "
        f"Expected tests per row: min={stats.min_expected_tests}, max={stats.max_expected_tests}."
    )

    if args.train_ids and args.val_ids:
        split_stats = split_jsonl_by_instance_ids(
            full_jsonl=paths.output_jsonl,
            train_ids_path=args.train_ids.expanduser(),
            val_ids_path=args.val_ids.expanduser(),
            train_jsonl=paths.train_jsonl,
            val_jsonl=paths.val_jsonl,
        )
        print(
            f"Wrote {split_stats.train_rows} train rows to {paths.train_jsonl} and "
            f"{split_stats.val_rows} val rows to {paths.val_jsonl}. "
            f"Skipped {split_stats.neither_rows} rows not listed in either split."
        )


if __name__ == "__main__":
    main()

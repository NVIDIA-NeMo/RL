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

import json
from pathlib import Path

import datasets
import pytest
from datasets import Dataset

from nemo_rl.data.datasets.response_datasets.nemogym_dataset import NemoGymDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


def test_jsonl_rows_are_cached_as_raw_text(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "train.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "one"}]},
        {"different_nested_shape": {"value": 2}},
    ]
    data_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))

    cache_dir = tmp_path / "hf_datasets_cache"
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(cache_dir))

    first = load_dataset_from_path(str(data_path), preserve_jsonl_rows=True)
    second = load_dataset_from_path(str(data_path), preserve_jsonl_rows=True)

    assert first.column_names == ["text"]
    assert [json.loads(row) for row in first["text"]] == rows
    assert first.cache_files == second.cache_files
    assert first.cache_files
    assert Path(first.cache_files[0]["filename"]).is_file()
    assert cache_dir in Path(first.cache_files[0]["filename"]).parents


def test_nemogym_dataset_accepts_preconverted_parquet(tmp_path: Path) -> None:
    data_path = tmp_path / "train.parquet"
    rows = ['{"sample": 1}', '{"sample": 2}']
    Dataset.from_dict({"extra_env_info": rows}).to_parquet(str(data_path))

    dataset = NemoGymDataset(str(data_path))

    assert dataset.dataset["extra_env_info"] == rows
    assert dataset.dataset.column_names == ["extra_env_info", "task_name"]
    assert len(set(dataset.dataset["task_name"])) == 1
    # Only JSONL files are line-scannable, so no source identity is recorded.
    assert dataset.agent_name_sources is None


def test_nemogym_dataset_accepts_save_to_disk_dataset(tmp_path: Path) -> None:
    data_path = tmp_path / "train_arrow"
    rows = ['{"sample": 1}', '{"sample": 2}']
    Dataset.from_dict({"extra_env_info": rows}).save_to_disk(str(data_path))

    dataset = NemoGymDataset(str(data_path))

    assert dataset.dataset["extra_env_info"] == rows


@pytest.mark.parametrize("file_name", ["train.json", "train.ndjson", "train"])
def test_nemogym_dataset_reads_json_lines_regardless_of_suffix(
    tmp_path: Path, file_name: str
) -> None:
    data_path = tmp_path / file_name
    rows = [{"agent_ref": {"name": "a"}}, {"agent_ref": {"name": "b"}}]
    data_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))

    dataset = NemoGymDataset(str(data_path))

    assert [json.loads(row) for row in dataset.dataset["extra_env_info"]] == rows


def test_nemogym_dataset_records_source_for_jsonl_symlink(tmp_path: Path) -> None:
    # Mirrors the Hugging Face hub cache layout: snapshots/.../train.jsonl -> blobs/<sha>.
    blob_path = tmp_path / "blobs" / "0123abcd"
    blob_path.parent.mkdir()
    blob_path.write_text(f"{json.dumps({'agent_ref': {'name': 'a'}})}\n")
    data_path = tmp_path / "train.jsonl"
    data_path.symlink_to(blob_path)

    dataset = NemoGymDataset(str(data_path))

    assert dataset.agent_name_sources is not None
    assert {source.path for source in dataset.agent_name_sources} == {
        str(blob_path.resolve())
    }


def test_nemogym_dataset_rejects_preconverted_dataset_without_raw_text(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "train.parquet"
    Dataset.from_dict({"parsed": [1]}).to_parquet(str(data_path))

    try:
        NemoGymDataset(str(data_path))
    except ValueError as error:
        assert "'extra_env_info' or 'text'" in str(error)
    else:
        raise AssertionError("Expected an invalid pre-converted schema to fail")

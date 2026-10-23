# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
from pathlib import Path

from nemo_rl.data.datasets.utils import load_dataset_from_path


def test_load_dataset_from_path_recognizes_megatron_jsonl_packed(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "sample.jsonl.packed"
    record = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = load_dataset_from_path(str(data_path))

    assert len(dataset) == 1
    assert dataset[0] == record


def test_load_dataset_from_path_preserves_packed_jsonl_rows(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "sample.jsonl.packed"
    row = '{"messages": [{"role": "user", "content": "hello"}]}'
    data_path.write_text(row + "\n", encoding="utf-8")

    dataset = load_dataset_from_path(str(data_path), preserve_jsonl_rows=True)

    assert dataset[0]["text"] == row

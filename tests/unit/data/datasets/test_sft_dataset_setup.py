# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples import run_sft


def test_setup_data_rejects_duplicate_megatron_sft_packed_entries(
    monkeypatch,
) -> None:
    packed_dataset = SimpleNamespace(
        dataset=object(),
        val_dataset=None,
        task_name="megatron_sft_packed",
        task_spec=object(),
        processor=Mock(),
        preprocessor=None,
    )
    data_config = {
        "train": [{"path": "first.jsonl"}, {"path": "second.jsonl"}],
        "add_bos": False,
        "add_eos": False,
        "add_generation_prompt": False,
        "max_input_seq_length": 8,
    }
    monkeypatch.setattr(
        run_sft,
        "load_response_dataset",
        Mock(side_effect=[packed_dataset, packed_dataset]),
    )
    monkeypatch.setattr(
        run_sft,
        "merge_datasets",
        Mock(side_effect=AssertionError("duplicate registration was not rejected")),
    )

    with pytest.raises(
        ValueError,
        match="multiple megatron_sft_packed datasets",
    ):
        run_sft.setup_data(object(), data_config)

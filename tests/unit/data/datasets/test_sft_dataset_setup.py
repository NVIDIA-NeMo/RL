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
        "train": [
            {"dataset_name": "megatron_sft_packed", "path": "first.jsonl"},
            {"dataset_name": "megatron_sft_packed", "path": "second.jsonl"},
        ],
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
        run_sft.setup_data(
            object(),
            data_config,
            {"megatron_cfg": {"enabled": True, "context_parallel_size": 2}},
        )


def test_setup_data_passes_policy_cp_only_to_packed_datasets(monkeypatch) -> None:
    packed_dataset = SimpleNamespace(
        dataset=[{"task_name": "megatron_sft_packed"}],
        val_dataset=None,
        task_name="megatron_sft_packed",
        task_spec=object(),
        processor=Mock(),
        preprocessor=None,
    )
    loader = Mock(return_value=packed_dataset)
    monkeypatch.setattr(run_sft, "load_response_dataset", loader)
    monkeypatch.setattr(run_sft, "merge_datasets", lambda datasets: datasets[0])
    monkeypatch.setattr(
        run_sft, "AllTaskProcessedDataset", lambda *args, **kwargs: [object()]
    )
    data_config = {
        "train": {
            "dataset_name": "megatron_sft_packed",
            "megatron_sft": {"prompt_format": "identity"},
        },
        "validation": None,
        "add_bos": False,
        "add_eos": False,
        "add_generation_prompt": False,
        "max_input_seq_length": 8,
    }

    run_sft.setup_data(
        object(),
        data_config,
        {"megatron_cfg": {"enabled": True, "context_parallel_size": 4}},
    )

    assert loader.call_args.kwargs == {"context_parallel_size": 4}


def test_setup_data_does_not_pass_context_size_to_regular_dataset(monkeypatch) -> None:
    regular_dataset = SimpleNamespace(
        dataset=[{"task_name": "squad"}],
        val_dataset=None,
        task_name="squad",
        task_spec=object(),
        processor=Mock(),
        preprocessor=None,
    )
    loader = Mock(return_value=regular_dataset)
    monkeypatch.setattr(run_sft, "load_response_dataset", loader)
    monkeypatch.setattr(run_sft, "merge_datasets", lambda datasets: datasets[0])
    monkeypatch.setattr(
        run_sft, "AllTaskProcessedDataset", lambda *args, **kwargs: [object()]
    )
    data_config = {
        "train": {"dataset_name": "squad"},
        "validation": None,
        "add_bos": False,
        "add_eos": False,
        "add_generation_prompt": False,
        "max_input_seq_length": 8,
    }

    run_sft.setup_data(
        object(),
        data_config,
        {"megatron_cfg": {"enabled": True, "context_parallel_size": 4}},
    )

    assert loader.call_args.kwargs == {}

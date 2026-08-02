"""Regression tests for custom preference dataset resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nemo_rl.data.datasets.preference_datasets import load_preference_dataset


def test_load_preference_dataset_resolves_external_dotted_path(monkeypatch):
    config = {"dataset_name": "my_module.MyDataset", "data_path": "dummy.jsonl"}
    mock_cls = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.data.datasets.preference_datasets.resolve_external_dataset_class",
        lambda name: mock_cls,
    )

    dataset = load_preference_dataset(config)

    mock_cls.assert_called_once_with(**config)
    mock_cls.return_value.set_task_spec.assert_called_once_with(config)

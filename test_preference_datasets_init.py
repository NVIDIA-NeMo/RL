"""Regression tests for nemo_rl.data.datasets.preference_datasets."""

from __future__ import annotations

import pytest

from nemo_rl.data.datasets.preference_datasets import load_preference_dataset


def test_load_preference_dataset_raises_for_unsupported_name():
    config = {"dataset_name": "UnknownDataset"}
    with pytest.raises(
        ValueError,
        match="Unsupported dataset_name='UnknownDataset'",
    ):
        load_preference_dataset(config)

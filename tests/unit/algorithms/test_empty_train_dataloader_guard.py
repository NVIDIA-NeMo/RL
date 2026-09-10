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
from unittest.mock import MagicMock

import pytest
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.utils import check_train_dataloader_not_empty

# Every algorithm setup builds its train dataloader with drop_last=True, so a
# batch size larger than the dataset yields a zero-length dataloader and the
# training loop silently never runs (#921). These tests cover the shared guard
# the setups call right after constructing the dataloader, plus a regression
# test through the real SFT setup path the issue reported.


def _loader(dataset, batch_size: int, drop_last: bool = True):
    return StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
    )


def test_raises_when_batch_size_exceeds_dataset():
    """bs=16 over 8 samples with drop_last=True is the reported silent no-op."""
    dataset = list(range(8))
    with pytest.raises(ValueError, match="zero batches") as excinfo:
        check_train_dataloader_not_empty(
            _loader(dataset, 16),
            dataset=dataset,
            batch_size=16,
            batch_size_source="policy.train_global_batch_size",
        )
    message = str(excinfo.value)
    assert "policy.train_global_batch_size=16" in message
    assert "8 samples" in message
    assert "gradient update" in message


def test_passes_when_dataset_covers_at_least_one_batch():
    dataset = list(range(16))
    check_train_dataloader_not_empty(
        _loader(dataset, 8),
        dataset=dataset,
        batch_size=8,
        batch_size_source="policy.train_global_batch_size",
    )


def test_partial_epoch_is_not_flagged():
    """dataset >= batch with a dropped remainder is normal, not an error."""
    dataset = list(range(20))
    check_train_dataloader_not_empty(
        _loader(dataset, 8),
        dataset=dataset,
        batch_size=8,
        batch_size_source="policy.train_global_batch_size",
    )


def test_without_drop_last_a_small_dataset_still_passes():
    """A partial batch is kept when drop_last=False, so nothing to flag."""
    dataset = list(range(8))
    check_train_dataloader_not_empty(
        _loader(dataset, 16, drop_last=False),
        dataset=dataset,
        batch_size=16,
        batch_size_source="policy.train_global_batch_size",
    )


def test_iterable_dataset_without_len_is_skipped():
    """Iterable-style datasets have no length, on the dataset or its
    dataloader; the guard must skip them entirely rather than raise
    ``TypeError`` probing either one."""

    class _Stream(IterableDataset):
        def __iter__(self):
            yield from range(4)

    stream = _Stream()
    dataloader = StatefulDataLoader(stream, batch_size=16)
    check_train_dataloader_not_empty(
        dataloader,
        dataset=stream,
        batch_size=16,
        batch_size_source="policy.train_global_batch_size",
    )


def test_context_names_the_dataloader():
    """Multi-dataloader setups (GRPO tasks) must say which dataloader is empty."""
    dataset = list(range(2))
    with pytest.raises(ValueError, match="for task 'math'"):
        check_train_dataloader_not_empty(
            _loader(dataset, 4),
            dataset=dataset,
            batch_size=4,
            batch_size_source="num_prompts_per_dataloader * batch_multiplier",
            context=" for task 'math'",
        )


# ---------------------------------------------------------------------------
# Regression coverage through the real SFT setup path reported in #921.
# ---------------------------------------------------------------------------


def _sft_master_config(train_global_batch_size: int):
    """Minimal master config for driving sft.setup() up to the guard."""
    master_config = MagicMock()
    master_config.sft.seed = 42
    master_config.policy = {"train_global_batch_size": train_global_batch_size}
    master_config.data = {"shuffle": False, "num_workers": 0}
    master_config.checkpointing = {}
    # Use concrete cluster settings so the missing segment_size defaults to None,
    # skipping Ray topology discovery before the mocked cluster sentinel.
    master_config.cluster = {"num_nodes": 1, "gpus_per_node": 1}
    return master_config


class _ReachedClusterConstruction(Exception):
    """Sentinel raised by the mocked cluster class: setup made it past the
    guard and reached infrastructure construction."""


def _patch_sft_infrastructure(monkeypatch):
    """Mock everything before/after the data section so no Ray or model
    infrastructure is touched; return the pieces whose usage we assert on."""
    import nemo_rl.algorithms.sft as sft_module

    monkeypatch.setattr(sft_module, "Logger", MagicMock())
    monkeypatch.setattr(sft_module, "load_dataloader_state", MagicMock())
    checkpoint_manager_cls = MagicMock()
    checkpoint_manager_cls.return_value.get_latest_checkpoint_path.return_value = None
    checkpoint_manager_cls.return_value.load_training_info.return_value = None
    monkeypatch.setattr(sft_module, "CheckpointManager", checkpoint_manager_cls)
    cluster_cls = MagicMock()
    policy_cls = MagicMock()
    monkeypatch.setattr(sft_module, "RayVirtualCluster", cluster_cls)
    monkeypatch.setattr(sft_module, "Policy", policy_cls)
    return sft_module, cluster_cls, policy_cls


def test_sft_setup_fails_fast_on_oversized_batch(monkeypatch):
    """The reported case (#921): 128 samples, batch size 1024. setup() must
    raise before constructing the cluster or the policy."""
    sft_module, cluster_cls, policy_cls = _patch_sft_infrastructure(monkeypatch)

    with pytest.raises(ValueError, match="zero batches"):
        sft_module.setup(
            master_config=_sft_master_config(train_global_batch_size=1024),
            tokenizer=MagicMock(),
            train_dataset=list(range(128)),
            val_dataset=None,
        )

    cluster_cls.assert_not_called()
    policy_cls.assert_not_called()


def test_sft_setup_accepts_dataset_equal_to_batch_size(monkeypatch):
    """dataset size == batch size yields exactly one batch and must pass the
    guard. The mocked cluster class raises a sentinel, so this test can only
    pass if setup ran the real guard without raising and then reached cluster
    construction; any earlier failure surfaces as its own exception."""
    sft_module, cluster_cls, _ = _patch_sft_infrastructure(monkeypatch)
    cluster_cls.side_effect = _ReachedClusterConstruction()

    real_guard = sft_module.check_train_dataloader_not_empty
    guard_spy = MagicMock(wraps=real_guard)
    monkeypatch.setattr(sft_module, "check_train_dataloader_not_empty", guard_spy)

    with pytest.raises(_ReachedClusterConstruction):
        sft_module.setup(
            master_config=_sft_master_config(train_global_batch_size=128),
            tokenizer=MagicMock(),
            train_dataset=list(range(128)),
            val_dataset=None,
        )

    guard_spy.assert_called_once()

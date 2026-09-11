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

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

# sft_dataloader imports megatron.energon and sft_worker imports megatron.core,
# both of which ship only in the `mcore` extra. importorskip must run before the
# imports below: the mcore mark is applied in pytest_collection_modifyitems,
# too late to prevent a collection error.
pytest.importorskip("megatron.energon")
pytest.importorskip("megatron.core")

pytestmark = pytest.mark.mcore

from nemo_rl.data.energon.config import (  # noqa: E402
    EnergonLoaderConfig,
    EnergonSourceConfig,
)
from nemo_rl.data.energon.sft_dataloader import (  # noqa: E402
    _identity_fingerprint,
    _loader_identity,
    _v2_topology,
    _worker_config,
    build_energon_sft_loader,
)
from nemo_rl.data_plane import KVBatchMeta  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.distributed.ray_actor_environment_registry import (  # noqa: E402
    get_actor_python_env,
)


def _v2_fingerprint(
    *,
    source: EnergonSourceConfig,
    loader_config: EnergonLoaderConfig,
    adapter_fingerprint: str,
    split_role: str,
    logical_rank: int,
    logical_world_size: int,
    placement_fingerprint: str,
    batch_size: int = 1,
) -> str:
    """Hash a V2 loader identity the way ``build_energon_sft_loader`` does."""
    return _identity_fingerprint(
        _loader_identity(
            source=source,
            loader_config=loader_config,
            adapter_fingerprint=adapter_fingerprint,
            split_role=split_role,
            batch_size=batch_size,
            shuffle=split_role == "train",
            topology=_v2_topology(
                loader_config=loader_config,
                placement_fingerprint=placement_fingerprint,
                logical_rank=logical_rank,
                logical_world_size=logical_world_size,
            ),
            packing_algorithm=None,
            max_sequences_per_bin=None,
            sequence_length_pad_multiple=1,
            only_unmask_final=False,
        )
    )


def test_worker_config_uses_logical_data_rank() -> None:
    worker = _worker_config(
        EnergonLoaderConfig(model_family="qwen", num_workers=3),
        logical_rank=2,
        logical_world_size=4,
    )

    assert worker.rank == 2
    assert worker.world_size == 4
    assert worker.num_workers == 3


def test_v2_fingerprint_identifies_each_logical_shard() -> None:
    source = EnergonSourceConfig(path="/dataset", split="train", virtual_epoch_length=8)
    loader = EnergonLoaderConfig(model_family="qwen")
    common = {
        "source": source,
        "loader_config": loader,
        "adapter_fingerprint": "processor",
        "split_role": "train",
        "logical_world_size": 2,
        "placement_fingerprint": "placement",
    }

    rank_zero = _v2_fingerprint(logical_rank=0, **common)
    rank_one = _v2_fingerprint(logical_rank=1, **common)

    assert rank_zero != rank_one
    assert rank_zero == _v2_fingerprint(logical_rank=0, **common)

    nemotron = loader.model_copy(update={"model_family": "nemotron"})
    assert rank_zero != _v2_fingerprint(
        logical_rank=0,
        **{**common, "loader_config": nemotron},
    )


def test_v2_loader_applies_cache_pool_and_gc_controls() -> None:
    adapter = MagicMock(fingerprint="processor")
    task_encoder = MagicMock()
    task_encoder.cookers = [MagicMock(need_cache=True)]
    dataset = object()
    cache_pool = object()
    raw_loader = MagicMock()

    with (
        patch(
            "nemo_rl.data.energon.sft_dataloader.build_processor_adapter",
            return_value=adapter,
        ),
        patch(
            "nemo_rl.data.energon.sft_dataloader._task_encoder",
            return_value=task_encoder,
        ),
        patch(
            "nemo_rl.data.energon.sft_dataloader.get_train_dataset",
            return_value=dataset,
        ) as get_train_dataset,
        patch(
            "nemo_rl.data.energon.sft_dataloader.FileStoreCachePool",
            return_value=cache_pool,
        ) as cache_pool_type,
        patch(
            "nemo_rl.data.energon.sft_dataloader.get_savable_loader",
            return_value=raw_loader,
        ) as get_savable_loader,
    ):
        build_energon_sft_loader(
            data_config={
                "shuffle": True,
                "energon": {
                    "model_family": "qwen",
                    "cache_pool_max_gbytes": 8,
                    "cache_pool_num_workers": 3,
                    "gc_collect_every_n_steps": 1234,
                    "task_encoder": {
                        "packing": {
                            "name": "balanced_greedy_knapsack",
                            "buffer_size": 5000,
                            "options": {
                                "max_sequence_length": 128,
                                "sequence_length_pad_multiple": 8,
                                "balanced_knapsack_delta": 5,
                            },
                        }
                    },
                },
            },
            source=EnergonSourceConfig(
                path="/dataset", split="train", virtual_epoch_length=8
            ),
            processor=MagicMock(tokenizer=MagicMock()),
            batch_size=2,
            max_sequence_length=128,
            split_role="train",
            logical_rank=0,
            logical_world_size=1,
            placement_fingerprint="placement",
            only_unmask_final=False,
        )

    cache_pool_type.assert_called_once_with(
        method="raw", num_workers=3, max_cache_size_gbytes=8.0
    )
    assert get_train_dataset.call_args.kwargs["packing_buffer_size"] == 5000
    assert get_savable_loader.call_args.kwargs["cache_pool"] is cache_pool
    assert get_savable_loader.call_args.kwargs["gc_collect_every_n_steps"] == 1234


def test_sft_v2_worker_uses_megatron_worker_environment() -> None:
    assert get_actor_python_env(
        "nemo_rl.data.energon.sft_worker.SFTMegatronPolicyWorker"
    ) == get_actor_python_env(
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    )


def test_sft_v2_worker_defers_processor_construction() -> None:
    from nemo_rl.data.energon.sft_worker import SFTMegatronPolicyWorker
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker_cls = SFTMegatronPolicyWorker.__ray_metadata__.modified_class
    with (
        patch.object(MegatronPolicyWorkerImpl, "__init__", return_value=None),
        patch("nemo_rl.algorithms.utils.get_tokenizer") as get_tokenizer,
    ):
        worker = worker_cls(
            {"tokenizer": {"use_processor": True}}, tokenizer=MagicMock()
        )

    get_tokenizer.assert_not_called()
    assert worker._sft_processor is None


def test_sft_v2_worker_publishes_sequence_alignment() -> None:
    from nemo_rl.data.energon.sft_worker import SFTMegatronPolicyWorker

    worker_cls = SFTMegatronPolicyWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker._sft_loader = MagicMock()
    worker._sft_loader_iterator = iter([{"message_log": []}])
    worker._sft_active_envelope = None
    worker._sft_logical_rank = 0
    worker._sft_logical_world_size = 2
    worker._sft_next_batch_index = 0
    worker._ld_on = False
    worker._ld_phase = None
    worker._ld_t0 = 0.0
    worker._ld_durations = {}
    worker._ld_watchdog = None
    worker.tokenizer = MagicMock(pad_token_id=0)
    worker._dp_client = MagicMock()

    prepared = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "input_lengths": torch.tensor([3, 2], dtype=torch.int32),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "sample_mask": torch.ones(2),
            "source_ids": ["source-a", "source-b"],
        }
    )
    policy_fields = [key for key in prepared if key != "source_ids"]
    worker._dp_client.put_samples.return_value = KVBatchMeta(
        partition_id="sft_v2_dp0_batch0",
        task_name=None,
        sample_ids=["sft_v2_dp0_batch0_row0", "sft_v2_dp0_batch0_row1"],
        fields=policy_fields,
        sequence_lengths=[3, 2],
        extra_info={"generation": 7},
    )

    with patch(
        "nemo_rl.data.energon.sft_worker.prepare_sft_batch",
        return_value=prepared,
    ):
        envelope = worker.load_next_sft_batch(
            only_unmask_final=False,
            make_sequence_length_divisible_by=4,
        )

    assert envelope.meta.extra_info == {
        "generation": 7,
        "pad_to_multiple": 4,
    }
    assert envelope.source_ids == ("source-a", "source-b")
    assert envelope.field_names == tuple(policy_fields)
    assert "source_ids" not in worker._dp_client.put_samples.call_args.kwargs["fields"]
    assert worker._dp_client.put_samples.call_args.kwargs["tags"] == [
        {"source_id": "source-a"},
        {"source_id": "source-b"},
    ]
    assert set(envelope.load_phase_seconds) == {
        "iter",
        "prepare",
        "post-prepare",
        "tensordict",
        "publish",
        "publish_setup",
        "publish_register_partition",
        "publish_source_tags",
        "publish_put_samples",
        "publish_batch_metadata",
    }
    assert all(value >= 0.0 for value in envelope.load_phase_seconds.values())
    assert envelope.load_seconds >= sum(
        envelope.load_phase_seconds[phase]
        for phase in ("iter", "prepare", "post-prepare", "tensordict", "publish")
    )

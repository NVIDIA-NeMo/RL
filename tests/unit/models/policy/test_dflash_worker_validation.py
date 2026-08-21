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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.megatron.draft.training import resolve_draft_speculator
from nemo_rl.models.policy.draft_config import DFlashDraftConfig, DSparkDraftConfig
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
    _all_reduce_draft_normalization_counts,
    _validate_draft_training_entrypoint,
    _validate_draft_training_setup,
)

pytestmark = pytest.mark.mcore


def test_optional_draft_config_does_not_start_update_probe() -> None:
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.cfg = {}
    worker.draft_model = None

    assert worker._maybe_start_draft_update_probe() is None


def _provider():
    provider = resolve_draft_speculator(
        DFlashDraftConfig(
            enabled=True,
            gamma=3,
            anchors_per_sample=1,
            mask_token_id=7,
            target_hidden_state_layer_ids=[1, 3],
        )
    )
    assert provider is not None
    return provider


def _dspark_provider():
    provider = resolve_draft_speculator(
        DSparkDraftConfig(
            enabled=True,
            block_size=3,
            anchors_per_sample=2,
            mask_token_id=7,
            target_hidden_state_layer_ids=[1, 3],
        )
    )
    assert provider is not None
    return provider


def test_dflash_setup_rejects_layout_and_target_mismatches_together() -> None:
    config = {
        "megatron_cfg": {"use_fused_linear_logprobs": True},
        "sequence_packing": {"enabled": True},
        "generation": {"vllm_kwargs": {"speculative_config": {"method": "eagle3"}}},
    }
    model_cfg = SimpleNamespace(
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        sequence_parallel=True,
        num_layers=3,
    )

    with pytest.raises(ValueError) as error:
        _validate_draft_training_setup(
            draft_provider=_provider(),
            config=config,
            model_cfg=model_cfg,
        )

    message = str(error.value)
    assert "pipeline_model_parallel_size must be 1" in message
    assert "use_fused_linear_logprobs must be disabled" in message
    assert "target_hidden_state_layer_ids exceed the target model: 3" in message
    assert "generation speculative method must be dflash" in message


def test_dflash_setup_allows_training_without_generation() -> None:
    _validate_draft_training_setup(
        draft_provider=_provider(),
        config={"sequence_packing": {"enabled": False}, "generation": None},
        model_cfg=SimpleNamespace(
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=False,
            num_layers=4,
        ),
    )


def test_dflash_setup_rejects_unsupported_target_tp() -> None:
    with pytest.raises(
        ValueError,
        match="tensor_model_parallel_size must be 2",
    ):
        _validate_draft_training_setup(
            draft_provider=_provider(),
            config={"sequence_packing": {"enabled": False}, "generation": None},
            model_cfg=SimpleNamespace(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                num_layers=4,
            ),
        )


def test_dflash_normalization_counts_move_to_nccl_device_before_reduce() -> None:
    counts = MagicMock()
    counts.device.type = "cpu"
    reduced_counts = MagicMock()
    counts.to.return_value = reduced_counts
    group = object()

    with (
        patch("torch.distributed.get_backend", return_value="nccl"),
        patch("torch.cuda.current_device", return_value=3),
        patch("torch.distributed.all_reduce") as all_reduce,
    ):
        result = _all_reduce_draft_normalization_counts(counts, group=group)

    counts.to.assert_called_once_with(device=3)
    all_reduce.assert_called_once_with(reduced_counts, group=group)
    assert result is reduced_counts


@pytest.mark.parametrize("context_parallel_size", [2, 4])
def test_dflash_setup_allows_packed_cp_with_target_sp(
    context_parallel_size: int,
) -> None:
    _validate_draft_training_setup(
        draft_provider=_provider(),
        config={"sequence_packing": {"enabled": True}, "generation": None},
        model_cfg=SimpleNamespace(
            pipeline_model_parallel_size=1,
            context_parallel_size=context_parallel_size,
            sequence_parallel=True,
            virtual_pipeline_model_parallel_size=None,
            num_layers=4,
        ),
    )


def test_dflash_setup_requires_nemo_owned_packing_for_cp() -> None:
    with pytest.raises(ValueError, match="requires sequence_packing.enabled=true"):
        _validate_draft_training_setup(
            draft_provider=_provider(),
            config={"sequence_packing": {"enabled": False}, "generation": None},
            model_cfg=SimpleNamespace(
                pipeline_model_parallel_size=1,
                context_parallel_size=2,
                sequence_parallel=True,
                virtual_pipeline_model_parallel_size=None,
                num_layers=4,
            ),
        )


def test_dflash_setup_requires_packing_to_reconstruct_target_sp() -> None:
    with pytest.raises(ValueError, match="sequence_parallel requires sequence_packing"):
        _validate_draft_training_setup(
            draft_provider=_provider(),
            config={"sequence_packing": {"enabled": False}, "generation": None},
            model_cfg=SimpleNamespace(
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=True,
                virtual_pipeline_model_parallel_size=None,
                num_layers=4,
            ),
        )


def test_dflash_setup_rejects_vpp_and_generation_cp() -> None:
    config = {
        "sequence_packing": {"enabled": True},
        "generation": {
            "mcore_generation_config": {
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 2,
            }
        },
    }
    with pytest.raises(ValueError) as error:
        _validate_draft_training_setup(
            draft_provider=_provider(),
            config=config,
            model_cfg=SimpleNamespace(
                pipeline_model_parallel_size=1,
                context_parallel_size=2,
                sequence_parallel=True,
                virtual_pipeline_model_parallel_size=2,
                num_layers=4,
            ),
        )

    assert "virtual_pipeline_model_parallel_size must be 1" in str(error.value)
    assert "generation context_parallel_size must be 1" in str(error.value)


def test_packed_cp_draft_training_requires_split_entrypoint() -> None:
    with pytest.raises(ValueError, match="split begin/train_microbatch/finish API"):
        _validate_draft_training_entrypoint(
            draft_provider=_provider(),
            context_parallel_size=2,
            split_api=False,
        )

    _validate_draft_training_entrypoint(
        draft_provider=_provider(),
        context_parallel_size=4,
        split_api=True,
    )


def test_dspark_setup_allows_packed_cp4_target_sp_and_matching_generation() -> None:
    _validate_draft_training_setup(
        draft_provider=_dspark_provider(),
        config={
            "sequence_packing": {"enabled": True},
            "generation": {"vllm_kwargs": {"speculative_config": {"method": "dspark"}}},
        },
        model_cfg=SimpleNamespace(
            pipeline_model_parallel_size=1,
            context_parallel_size=4,
            sequence_parallel=True,
            virtual_pipeline_model_parallel_size=None,
            num_layers=4,
        ),
    )


def test_dspark_setup_rejects_mismatched_generation_method() -> None:
    with pytest.raises(
        ValueError, match="generation speculative method must be dspark"
    ):
        _validate_draft_training_setup(
            draft_provider=_dspark_provider(),
            config={
                "sequence_packing": {"enabled": True},
                "generation": {
                    "vllm_kwargs": {"speculative_config": {"method": "dflash"}}
                },
            },
            model_cfg=SimpleNamespace(
                pipeline_model_parallel_size=1,
                context_parallel_size=2,
                sequence_parallel=True,
                virtual_pipeline_model_parallel_size=None,
                num_layers=4,
            ),
        )

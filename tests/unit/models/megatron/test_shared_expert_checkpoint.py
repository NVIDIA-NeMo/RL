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
"""Expert parameter reordering must not restore positional optimizer moments."""

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo_rl.models.megatron import sm_partition


@pytest.fixture
def checkpoint_io(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Substitute checkpoint I/O so format validation also runs without Megatron."""
    io = SimpleNamespace(
        resolve=Mock(return_value=(12, False)),
        name=Mock(return_value="/checkpoint/iter_0000012"),
        common=Mock(return_value={"optimizer": {}}),
        metadata=Mock(return_value={"distrib_optim_sharding_type": "fully_reshardable"}),
    )
    modules = {
        "megatron.bridge.training.checkpointing": SimpleNamespace(
            _DIRECT_ITERATION_DIR_SENTINEL=-2, _resolve_checkpoint_iteration=io.resolve
        ),
        "megatron.bridge.training.utils.checkpoint_utils": SimpleNamespace(get_checkpoint_name=io.name),
        "megatron.core": SimpleNamespace(dist_checkpointing=SimpleNamespace(
            load_common_state_dict=io.common, load_content_metadata=io.metadata
        )),
        "megatron.core.optimizer.distrib_optimizer": SimpleNamespace(DistributedOptimizer=SimpleNamespace(
            checkpoint_fully_reshardable_formats={"fully_reshardable", "fully_sharded_model_space", "fsdp_dtensor"}
        )),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return io


def _config(**overrides: object) -> SimpleNamespace:
    values = dict(load_optim=True, finetune=False, load="/checkpoint", ckpt_step=None,
                  dist_ckpt_optim_fully_reshardable=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("saved_format", ["fully_reshardable", "fully_sharded_model_space"])
@pytest.mark.parametrize("direct_iteration", [False, True])
def test_named_optimizer_resume_uses_resolved_checkpoint(
    checkpoint_io: SimpleNamespace, saved_format: str, direct_iteration: bool
) -> None:
    checkpoint_io.metadata.return_value = {"distrib_optim_sharding_type": saved_format}
    config = _config(ckpt_step=12)
    if direct_iteration:
        config.load = "/checkpoint/iter_0000012"
        checkpoint_io.resolve.return_value = (-2, False)
    sm_partition.configure_shared_expert_checkpoint(config)
    checkpoint_io.resolve.assert_called_once_with(config.load, 12)
    checkpoint_io.common.assert_called_once_with("/checkpoint/iter_0000012")
    checkpoint_io.metadata.assert_called_once_with(preloaded_state_dict={"optimizer": {}})
    assert config.dist_ckpt_optim_fully_reshardable
    if direct_iteration:
        checkpoint_io.name.assert_not_called()
    else:
        checkpoint_io.name.assert_called_once_with("/checkpoint", 12, release=False)


@pytest.mark.parametrize("saved_format", ["dp_reshardable", "dp_zero_gather_scatter", None])
def test_positional_optimizer_resume_is_rejected_before_reordering(
    checkpoint_io: SimpleNamespace, saved_format: str | None
) -> None:
    checkpoint_io.metadata.return_value = {"distrib_optim_sharding_type": saved_format}
    config = _config()
    with pytest.raises(ValueError, match="Bucket-position optimizer state cannot be restored"):
        sm_partition.configure_shared_expert_checkpoint(config)
    assert not config.dist_ckpt_optim_fully_reshardable


@pytest.mark.parametrize("overrides", [dict(load_optim=False), dict(finetune=True), dict(load=None)])
def test_fresh_optimizer_uses_named_saves_without_reading_checkpoint(overrides: dict[str, object]) -> None:
    config = _config(**overrides)
    sm_partition.configure_shared_expert_checkpoint(config)
    assert config.dist_ckpt_optim_fully_reshardable


@pytest.mark.parametrize("resolution", [(-1, False), (0, True)])
def test_missing_or_release_checkpoint_does_not_restore_optimizer(
    checkpoint_io: SimpleNamespace, resolution: tuple[int, bool]
) -> None:
    checkpoint_io.resolve.return_value = resolution
    config = _config()
    sm_partition.configure_shared_expert_checkpoint(config)
    checkpoint_io.common.assert_not_called()
    assert config.dist_ckpt_optim_fully_reshardable


def test_weights_only_checkpoint_allows_fresh_optimizer(checkpoint_io: SimpleNamespace) -> None:
    checkpoint_io.common.return_value = {"model": {}}
    config = _config()
    sm_partition.configure_shared_expert_checkpoint(config)
    checkpoint_io.metadata.assert_not_called()
    assert config.dist_ckpt_optim_fully_reshardable


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs Megatron and a GPU")
@pytest.mark.parametrize("source_reordered", [False, True])
def test_native_optimizer_checkpoint_preserves_moments_across_expert_order(
    tmp_path: Path, source_reordered: bool
) -> None:
    # These optional dependencies belong to the GPU/Megatron test lane.
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.dist_checkpointing import ShardedTensor
    from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig

    class Experts(MegatronModule):
        def __init__(self, reordered: bool) -> None:
            super().__init__(TransformerConfig(
                num_layers=1, hidden_size=16, num_attention_heads=1,
                bf16=True, params_dtype=torch.bfloat16, gradient_accumulation_fusion=False,
            ))
            for linear_name in ("linear_fc1", "linear_fc2"):
                linear = torch.nn.Module()
                self.add_module(linear_name, linear)
                for index in range(3):
                    linear.register_parameter(f"weight{index}", torch.nn.Parameter(
                        torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
                    ))
                if reordered:
                    assert sm_partition._reverse_expert_parameter_order(linear, num_experts=3)

        def sharded_state_dict(self) -> dict[str, ShardedTensor]:
            return {name: ShardedTensor.from_rank_offsets(name, parameter)
                    for name, parameter in self.named_parameters()}

    def build(reordered: bool) -> tuple[Experts, DistributedOptimizer]:
        model = Experts(reordered)
        ddp = DistributedDataParallel(model.config, DistributedDataParallelConfig(
            use_distributed_optimizer=True, pad_buckets_for_high_nccl_busbw=False,
        ), model)
        optimizer = get_megatron_optimizer(OptimizerConfig(
            bf16=True, params_dtype=torch.bfloat16, use_distributed_optimizer=True, lr=0.001,
        ), [ddp])
        if isinstance(optimizer, ChainedOptimizer):
            assert len(optimizer.chained_optimizers) == 1
            optimizer = optimizer.chained_optimizers[0]
        assert isinstance(optimizer, DistributedOptimizer)
        for group in optimizer.optimizer.param_groups:
            for parameter in group["params"]:
                optimizer.optimizer.state[parameter]["exp_avg"] = torch.zeros_like(parameter)
                optimizer.optimizer.state[parameter]["exp_avg_sq"] = torch.zeros_like(parameter)
        return model, optimizer

    assert not torch.distributed.is_initialized(), "run this test in an isolated process"
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1,
        timeout=timedelta(seconds=120),
    )
    try:
        parallel_state.initialize_model_parallel()
        source_model, source_optimizer = build(source_reordered)
        expected = {}
        with torch.no_grad():
            for name, parameter in source_model.named_parameters():
                code = 10 * int(name.startswith("linear_fc2")) + int(name[-1]) + 1
                tensors = source_optimizer._get_main_param_and_optimizer_states(parameter)
                for offset, key in enumerate(("param", "exp_avg", "exp_avg_sq")):
                    tensors[key].fill_(code + offset / 4)
                expected[name] = {key: value.clone() for key, value in tensors.items()}
        metadata = {"distrib_optim_sharding_type": "fully_reshardable"}
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        dist_checkpointing.save({"optimizer": source_optimizer.sharded_state_dict(
            source_model.sharded_state_dict(), metadata=metadata,
        )}, str(checkpoint_dir), content_metadata=metadata)
        sm_partition.configure_shared_expert_checkpoint(_config(load=str(checkpoint_dir)))
        target_model, target_optimizer = build(not source_reordered)
        target_state = {"optimizer": target_optimizer.sharded_state_dict(
            target_model.sharded_state_dict(), metadata=metadata, is_loading=True,
        )}
        restored = dist_checkpointing.load(target_state, str(checkpoint_dir))
        target_optimizer.load_state_dict(restored["optimizer"])
        for name, parameter in target_model.named_parameters():
            actual = target_optimizer._get_main_param_and_optimizer_states(parameter)
            for key, value in expected[name].items():
                assert torch.equal(actual[key], value), (name, key)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

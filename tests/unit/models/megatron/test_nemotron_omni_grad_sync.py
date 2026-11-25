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

"""Two-rank check that Nemotron Omni gradients are reduced across DP and TP ranks.

The Nemotron Omni provider deep-copies its config into the model it builds, so
gradient-sync callbacks installed on the provider never reach the config that
Megatron's scheduler reads. Training then runs without error while each rank
updates from its own local gradient. This test trains one step through NeMo-RL's
training path and checks that the optimizer applied the reduced gradient.
"""

import functools
import gc
from types import SimpleNamespace

import pytest
import torch

# Collected by every unit-test shard; skip where the Megatron extra is missing.
pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

from megatron.bridge.training.setup import _update_model_config_funcs  # noqa: E402
from megatron.core import parallel_state  # noqa: E402
from megatron.core.distributed import DistributedDataParallelConfig  # noqa: E402
from megatron.core.optimizer import (  # noqa: E402
    OptimizerConfig,
    get_megatron_optimizer,
)
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.tensor_parallel import (  # noqa: E402
    param_is_not_tensor_parallel_duplicate,
)
from megatron.core.tensor_parallel.random import (  # noqa: E402
    model_parallel_cuda_manual_seed,
)
from megatron.core.utils import get_model_config  # noqa: E402

from nemo_rl.algorithms.loss.loss_functions import NLLLossFn  # noqa: E402
from nemo_rl.data.multimodal_utils import PackedTensor  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.models.megatron.data import get_microbatch_iterator  # noqa: E402
from nemo_rl.models.megatron.setup import (  # noqa: E402
    check_grad_sync_callbacks,
    install_grad_sync_callbacks,
)
from nemo_rl.models.megatron.train import (  # noqa: E402
    LossPostProcessor,
    megatron_forward_backward,
)
from tests.unit.models.megatron.test_nemotron_omni_model import (  # noqa: E402
    _TinyOmniProvider,
)

pytestmark = pytest.mark.mcore

# The distributed optimizer only supports Adam. Adam's first step moves a
# weight by -lr * g / (|g| + eps), which is about lr * sign(g) for the usual
# tiny eps and would hide a wrong gradient magnitude. With eps far above |g| it
# is linear in g, about -(lr / eps) * g, so the update shows which gradient was
# used. A large lr / eps keeps the typical 1e-8..1e-6 gradients here well above
# fp32 rounding of the weights, which would swallow -g on weights near 1.
_ADAM_EPS = 1.0e3
_LR = 1.0e6 * _ADAM_EPS
# Large enough that the grad norm is computed but never clips. Random-init
# gradients here can be large, so a merely big value like 1e9 is not enough.
_NO_CLIP = 1.0e30


def _build_model(tensor_parallel_size: int):
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_parallel_size
    )
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)

    provider = _TinyOmniProvider(
        tensor_model_parallel_size=tensor_parallel_size,
        context_parallel_size=1,
        sequence_parallel=tensor_parallel_size > 1,
        # Mamba and MoE language layers, all trainable. No language attention
        # layer: TE has no fp32 attention backend for packed sequences, and
        # this test needs fp32 for exact comparisons.
        hybrid_layer_pattern="ME",
        num_moe_experts=4,
        moe_ffn_hidden_size=64,
        moe_router_topk=2,
        moe_shared_expert_overlap=False,
        # NeMo-RL trains with these (see _validate_training_config and the DDP
        # config in setup.py): losses arrive already normalized by the global
        # token count, and DDP sums gradients across data-parallel ranks.
        calculate_per_token_loss=True,
    )
    provider.finalize()
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        average_in_collective=False,
        use_distributed_optimizer=True,
        check_for_nan_in_grad=True,
    )
    (model,) = provider.provide_distributed_model(
        ddp_config=ddp_config,
        use_cpu_initialization=False,
        wrap_with_ddp=True,
        mixed_precision_wrapper=None,
    )
    # The case this test exists for: the scheduler's config is a copy.
    assert get_model_config(model) is not provider
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=_LR,
            weight_decay=0.0,
            adam_eps=_ADAM_EPS,
            clip_grad=_NO_CLIP,
            use_distributed_optimizer=True,
            bf16=False,
            fp16=False,
            params_dtype=torch.float32,
        ),
        [model],
    )
    return model, provider, ddp_config, optimizer


def _batch(device: torch.device) -> BatchedDataDict:
    """Two image rows; text tokens and pixels differ between data-parallel ranks."""
    generator = torch.Generator().manual_seed(
        2026 + parallel_state.get_data_parallel_rank()
    )
    input_ids = torch.tensor(
        [
            [7, 21, 18, 18, 22, 9, 10, 0],
            [11, 21, 18, 22, 12, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([7, 5], dtype=torch.long)
    positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
    is_text = (positions < lengths.unsqueeze(1)) & (input_ids < 18)
    random_text = torch.randint(23, 128, input_ids.shape, generator=generator)
    input_ids = torch.where(is_text, random_text, input_ids)
    images = torch.randn(2, 3, 32, 64, generator=generator)
    images[1, :, :, 32:] = 0
    image_sizes = torch.tensor([[32, 64], [32, 32]], dtype=torch.int32)

    data = BatchedDataDict(
        {
            "input_ids": input_ids.to(device),
            "input_lengths": lengths.to(device),
            "token_mask": (positions < lengths.unsqueeze(1)).float().to(device),
            "sample_mask": torch.ones(2, device=device),
            "pixel_values": PackedTensor(
                [images[0:1].to(device), images[1:2].to(device)], dim_to_pack=0
            ),
            "imgs_sizes": PackedTensor(
                [image_sizes[0:1].to(device), image_sizes[1:2].to(device)],
                dim_to_pack=0,
            ),
        }
    )
    data.micro_batch_indices = [[[0, 2]]]
    data.micro_batch_lengths = [[int(lengths.sum().item())]]
    return data


def _forward_backward(model, data: BatchedDataDict, tensor_parallel_size: int):
    """One training forward/backward through NeMo-RL's Megatron train loop."""
    cfg = {
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": True},
        "make_sequence_length_divisible_by": 4 * tensor_parallel_size,
        "megatron_cfg": {
            "tensor_model_parallel_size": tensor_parallel_size,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "sequence_parallel": tensor_parallel_size > 1,
        },
    }
    # NLLLoss normalizes by the global token count, as the policy worker does.
    valid_tokens = (data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)).sum()
    torch.distributed.all_reduce(
        valid_tokens, group=parallel_state.get_data_parallel_group()
    )
    valid_seqs = data["sample_mask"].sum()
    torch.distributed.all_reduce(
        valid_seqs, group=parallel_state.get_data_parallel_group()
    )
    (
        data_iterator,
        num_microbatches,
        micro_batch_size,
        _,
        padded_seq_length,
    ) = get_microbatch_iterator(
        data,
        cfg,
        mbs=2,
        straggler_timer=None,
        model_slices_context_parallel_inputs=True,
    )
    model.zero_grad_buffer()
    megatron_forward_backward(
        model=model,
        data_iterator=data_iterator,
        num_microbatches=num_microbatches,
        seq_length=padded_seq_length,
        mbs=micro_batch_size,
        post_processing_fn=LossPostProcessor(
            loss_fn=NLLLossFn(), cfg=cfg, num_microbatches=num_microbatches
        ),
        forward_only=False,
        global_valid_seqs=valid_seqs,
        global_valid_toks=valid_tokens,
        model_slices_context_parallel_inputs=True,
    )


def _trainable(model):
    return [
        (name, parameter)
        for name, parameter in model.module.named_parameters()
        if parameter.requires_grad
    ]


def _train_step(model, optimizer, data, tensor_parallel_size):
    """One optimizer step; returns this rank's local gradients and the grad norm.

    The local gradients are read from the same backward the optimizer uses, just
    before finalization would reduce them (or after the schedule, when no
    finalizer is installed), so backward nondeterminism can't separate the
    expected update from the applied one.
    """
    config = get_model_config(model)
    finalize_model_grads_func = config.finalize_model_grads_func
    local = {}

    def snapshot_local_gradients():
        local.update(
            (name, parameter.main_grad.detach().clone())
            for name, parameter in _trainable(model)
        )

    def snapshot_then_finalize(*args, **kwargs):
        snapshot_local_gradients()
        return finalize_model_grads_func(*args, **kwargs)

    if finalize_model_grads_func is not None:
        config.finalize_model_grads_func = snapshot_then_finalize
    try:
        optimizer.zero_grad()
        _forward_backward(model, data, tensor_parallel_size)
    finally:
        config.finalize_model_grads_func = finalize_model_grads_func
    if finalize_model_grads_func is None:
        snapshot_local_gradients()
    update_successful, grad_norm, _ = optimizer.step()
    assert update_successful
    return local, float(grad_norm)


def _expected_gradients(model, local, tensor_parallel_size):
    """Local gradients reduced by hand the way finalization should.

    Sums each gradient over the ranks that hold a replica of the parameter: the
    data-parallel group (the expert data-parallel group for experts), and the
    tensor-parallel group for sequence-parallel parameters, whose rank-local
    gradient covers only that rank's slice of the sequence.
    """
    expected, reduced = {}, set()
    for name, parameter in _trainable(model):
        gradient = local[name].clone()
        # Megatron marks expert parameters allreduce=False.
        if getattr(parameter, "allreduce", True):
            groups = [parallel_state.get_data_parallel_group()]
        else:
            groups = [parallel_state.get_expert_data_parallel_group()]
        if tensor_parallel_size > 1 and getattr(parameter, "sequence_parallel", False):
            groups.append(parallel_state.get_tensor_model_parallel_group())
        for group in groups:
            if torch.distributed.get_world_size(group) > 1:
                torch.distributed.all_reduce(gradient, group=group)
                reduced.add(name)
        expected[name] = gradient
    return expected, reduced


def _expected_grad_norm(model, expected) -> float:
    """Global L2 norm over all ranks, counting tensor-parallel replicas once."""
    squared = torch.zeros(1, device=torch.cuda.current_device(), dtype=torch.float64)
    for name, parameter in _trainable(model):
        if param_is_not_tensor_parallel_duplicate(parameter):
            squared += expected[name].double().pow(2).sum()
    torch.distributed.all_reduce(
        squared, group=parallel_state.get_tensor_model_parallel_group()
    )
    return squared.sqrt().item()


def _first_adam_update(gradient: torch.Tensor) -> torch.Tensor:
    """Adam's bias-corrected first step from zero state: -lr * g / (|g| + eps)."""
    return -_LR * gradient / (gradient.abs() + _ADAM_EPS)


def _update_errors(
    updates: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict[str, tuple[float, float]]:
    """Per tensor: relative error of the applied update, and its projection ratio.

    A ratio of 1 is the expected update; 0.5 would mean averaging instead of
    summing, and a ratio near 1/world_size or a large error with ratio near 1
    points at an unreduced, rank-local gradient. Sizes are floored at 1e-6 of the
    largest expected update, so tensors that barely moved don't dominate.
    """
    targets = {name: _first_adam_update(expected[name]).double() for name in updates}
    largest = max(target.norm().item() for target in targets.values())
    errors = {}
    for name, update in updates.items():
        target, update = targets[name], update.double()
        size = max(target.norm().item(), 1e-6 * largest, 1e-30)
        errors[name] = (
            (update - target).norm().item() / size,
            (update * target).sum().item() / size**2,
        )
    return errors


def _run_grad_sync(
    rank: int, world_size: int, *, tensor_parallel_size: int, binding: str
) -> None:
    assert world_size == 2
    model, provider, ddp_config, optimizer = _build_model(tensor_parallel_size)
    model.train()
    if binding == "runtime":
        install_grad_sync_callbacks(
            model,
            SimpleNamespace(
                ddp=ddp_config, dist=SimpleNamespace(align_grad_reduce=True)
            ),
            optimizer,
        )
    else:
        # NeMo-RL's binding before RL #3789: the callbacks land on the provider.
        _update_model_config_funcs(
            [model],
            provider,
            ddp_config,
            optimizer,
            align_grad_reduce=True,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        with pytest.raises(RuntimeError, match="finalize_model_grads_func"):
            check_grad_sync_callbacks(model)

    device = torch.device("cuda", torch.cuda.current_device())
    before = {name: parameter.detach().clone() for name, parameter in _trainable(model)}
    local, grad_norm = _train_step(
        model, optimizer, _batch(device), tensor_parallel_size
    )
    expected, reduced = _expected_gradients(model, local, tensor_parallel_size)
    assert reduced
    expected_norm = _expected_grad_norm(model, expected)

    updates = {
        name: parameter.detach() - before[name] for name, parameter in _trainable(model)
    }
    assert updates
    # Every module type gets a gradient: vision, projection, Mamba, MoE.
    for fragment in (
        "vision_model.",
        "vision_projection.",
        "mixer.",
        "experts.",
    ):
        assert any(fragment in name for name in updates), fragment

    if binding == "runtime":
        errors = _update_errors(updates, expected)
        worst = sorted(errors.items(), key=lambda item: -item[1][0])[:8]
        # The optimizer path rounds at about bf16 precision (~0.3% here); a
        # missing reduction is off by tens of percent or more.
        assert worst[0][1][0] < 2e-2, (
            f"grad_norm {grad_norm:.6g}, expected {expected_norm:.6g}; "
            "update vs expected (relative error, ratio):\n"
            + (
                "\n".join(
                    f"  {name}: {error:.3g}, {ratio:.4f}"
                    for name, (error, ratio) in worst
                )
            )
        )
        assert grad_norm == pytest.approx(expected_norm, rel=1e-3)
    else:
        # Without finalization, a parameter that needed a cross-rank reduction
        # moves by one rank's local gradient instead.
        error = sum(
            (updates[name] - _first_adam_update(expected[name])).double().pow(2).sum()
            for name in reduced
        )
        scale = sum(
            _first_adam_update(expected[name]).double().pow(2).sum() for name in reduced
        )
        assert (error / scale).sqrt().item() > 0.1
        if tensor_parallel_size == 1:
            # Every parameter is data-parallel here, so the norm moves too. With
            # TP, the tensor-parallel shards that need no reduction dominate it.
            assert grad_norm != pytest.approx(expected_norm, rel=1e-2)

    if rank == 0:
        print(
            f"NEMOTRON_OMNI_GRAD_SYNC tp={tensor_parallel_size} binding={binding} "
            f"grad_norm={grad_norm:.6f} expected_grad_norm={expected_norm:.6f} "
            f"tensors={len(updates)} "
            f"dtypes={sorted({str(p.dtype) for _, p in _trainable(model)})}",
            flush=True,
        )

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()


@pytest.mark.parametrize("binding", ["runtime", "provider"])
@pytest.mark.parametrize(
    "tensor_parallel_size", [1, 2], ids=["dp2", "tp2_sequence_parallel"]
)
def test_nemotron_omni_gradients_reduced_across_ranks(
    distributed_test_runner, tensor_parallel_size, binding
):
    """The optimizer applies the cross-rank gradient; the pre-#3789 binding does not."""
    distributed_test_runner(
        functools.partial(
            _run_grad_sync, tensor_parallel_size=tensor_parallel_size, binding=binding
        ),
        world_size=2,
    )

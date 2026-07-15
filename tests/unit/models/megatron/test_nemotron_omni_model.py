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

"""Distributed functional coverage for NeMo-RL's Nemotron Omni contract."""

import copy
import functools
import gc
import os
from dataclasses import dataclass

import pytest
import torch
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT,
    NemotronOmniModelProvider,
)
from nemo_rl.distributed.model_utils import (
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.models.megatron.data import process_microbatch


pytestmark = pytest.mark.mcore

_IMAGE_TOKEN_ID = 18


@dataclass
class _TinyOmniProvider(NemotronOmniModelProvider):
    """Small real RADIO/NemotronH model for a two-rank functional test."""

    has_sound: bool = False
    language_model_type: str = "nemotron6-moe"
    hidden_size: int = 128
    ffn_hidden_size: int = 256
    num_attention_heads: int = 4
    num_query_groups: int = 2
    kv_channels: int = 32
    mamba_num_heads: int = 4
    mamba_head_dim: int = 32
    mamba_num_groups: int = 2
    mamba_state_dim: int = 16
    hybrid_layer_pattern: str = "M"
    vocab_size: int = 128
    seq_length: int = 32
    image_token_index: int = _IMAGE_TOKEN_ID
    img_start_token_id: int = 21
    img_end_token_id: int = 22
    tokenizer_type: str = "nemotron6-moe"
    dynamic_resolution: bool = True
    use_vision_backbone_fp8_arch: bool = False
    vision_proj_ffn_hidden_size: int = 256
    pipeline_model_parallel_size: int = 1
    use_cpu_initialization: bool = True
    gradient_accumulation_fusion: bool = False
    nemotron_omni_contract: str = NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT

    def _build_vision_config(self, language_cfg):
        vision_cfg = copy.deepcopy(language_cfg)
        vision_cfg.sequence_parallel = False
        vision_cfg.context_parallel_size = 1
        vision_cfg.tp_comm_overlap = False
        vision_cfg.recompute_granularity = None
        vision_cfg.recompute_method = None
        vision_cfg.recompute_num_layers = None
        vision_cfg.mtp_num_layers = None
        vision_cfg.num_layers = 1
        vision_cfg.pipeline_model_parallel_size = 1
        vision_cfg.num_attention_heads = 4
        vision_cfg.add_bias_linear = True
        vision_cfg.add_qkv_bias = True
        vision_cfg.hidden_size = 128
        vision_cfg.ffn_hidden_size = 256
        vision_cfg.gated_linear_unit = False
        vision_cfg.kv_channels = 32
        vision_cfg.num_query_groups = 4
        vision_cfg.normalization = "LayerNorm"
        vision_cfg.qk_layernorm = False
        vision_cfg.layernorm_epsilon = 1e-6
        vision_cfg.class_token_len = 10
        return vision_cfg


def _build_distributed_model():
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
    )
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)

    provider = _TinyOmniProvider(
        freeze_language_model=True,
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        sequence_parallel=False,
    )
    provider.finalize()
    models = provider.provide_distributed_model(
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            check_for_nan_in_grad=True,
        ),
        wrap_with_ddp=True,
        mixed_precision_wrapper=None,
    )
    assert len(models) == 1
    return models[0]


def _expanded_fixture(device: torch.device):
    input_ids = torch.tensor(
        [
            [7, 21, 18, 18, 22, 9, 10, 0],
            [11, 21, 18, 22, 12, 0, 0, 0],
        ],
        dtype=torch.long,
        device=device,
    )
    lengths = torch.tensor([7, 5], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(2026)
    images = torch.randn(2, 3, 32, 64, generator=generator, device=device)
    images[1, :, :, 32:] = 0
    image_sizes = torch.tensor(
        [[32, 64], [32, 32]], dtype=torch.int32, device=device
    )
    return input_ids, lengths, images, image_sizes


def _forward(model):
    device = torch.device("cuda", torch.cuda.current_device())
    input_ids, lengths, images, image_sizes = _expanded_fixture(device)
    processed = process_microbatch(
        {"input_ids": input_ids, "input_lengths": lengths},
        seq_length_key="input_lengths",
        pad_individual_seqs_to_multiple_of=4,
        pack_sequences=True,
        delegate_pack_to_model=True,
    )
    output = model(
        input_ids=processed.input_ids_cp_sharded,
        attention_mask=processed.attention_mask,
        packed_seq_params=processed.packed_seq_params,
        pixel_values=images,
        imgs_sizes=image_sizes,
    )
    logprobs = from_parallel_logits_to_logprobs_packed_sequences(
        output,
        target=processed.input_ids,
        cu_seqlens_padded=processed.cu_seqlens_padded,
        unpacked_seqlen=input_ids.shape[1],
        vocab_start_index=0,
        vocab_end_index=output.shape[-1],
        group=parallel_state.get_tensor_model_parallel_group(),
        inference_only=False,
        cp_group=parallel_state.get_context_parallel_group(),
    )
    prediction_mask = torch.arange(
        input_ids.shape[1] - 1, device=device
    ).unsqueeze(0) < (lengths - 1).unsqueeze(1)
    loss = -(logprobs * prediction_mask).sum() / prediction_mask.sum()
    return loss, output


def _run_training_checkpoint_roundtrip(
    rank: int,
    world_size: int,
    *,
    checkpoint_dir: str,
) -> None:
    assert world_size == 2
    model = _build_distributed_model()
    model.train()
    model.zero_grad_buffer()

    loss, output = _forward(model)
    loss.backward()
    model.finish_grad_sync()

    core_model = model.module
    gradients = {}
    before_update = {}
    optimizer_parameters = []
    for name, parameter in core_model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert name.startswith(("vision_model.", "vision_projection."))
        assert hasattr(parameter, "main_grad")
        assert torch.isfinite(parameter.main_grad).all()
        rank_zero_gradient = parameter.main_grad.detach().clone()
        torch.distributed.broadcast(rank_zero_gradient, src=0)
        torch.testing.assert_close(
            parameter.main_grad, rank_zero_gradient, rtol=0, atol=0
        )
        gradients[name] = parameter.main_grad
        before_update[name] = parameter.detach().clone()
        parameter.grad = parameter.main_grad.to(parameter.dtype).clone()
        optimizer_parameters.append(parameter)
    assert gradients

    optimizer = torch.optim.SGD(optimizer_parameters, lr=1.0)
    optimizer.step()
    changed = {
        name
        for name, parameter in core_model.named_parameters()
        if name in before_update and not torch.equal(parameter, before_update[name])
    }
    assert any(name.startswith("vision_model.") for name in changed)
    assert any(name.startswith("vision_projection.") for name in changed)

    model.eval()
    with torch.no_grad():
        _, post_update_output = _forward(model)
    post_update_output = post_update_output.detach().clone()

    metadata = {
        "dp_cp_group": parallel_state.get_data_parallel_group(
            with_context_parallel=True
        )
    }
    sharded_state = core_model.sharded_state_dict(metadata=metadata)
    assert changed <= sharded_state.keys()
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.distributed.barrier()
    dist_checkpointing.save({"model": sharded_state}, checkpoint_dir)

    provider = _TinyOmniProvider(
        freeze_language_model=True,
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        sequence_parallel=False,
    )
    provider.finalize()
    restored_model = provider.provide().cuda().eval()
    restore_template = restored_model.sharded_state_dict(metadata=metadata)
    loaded_state = dist_checkpointing.load(
        {"model": restore_template}, checkpoint_dir
    )
    incompatible = restored_model.load_state_dict(loaded_state["model"])
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

    restored_parameters = dict(restored_model.named_parameters())
    original_parameters = dict(core_model.named_parameters())
    for name in changed:
        torch.testing.assert_close(
            restored_parameters[name], original_parameters[name], rtol=0, atol=0
        )
    with torch.no_grad():
        _, restored_output = _forward(restored_model)
    torch.testing.assert_close(restored_output, post_update_output, rtol=0, atol=0)

    if rank == 0:
        print(
            "NEMOTRON_OMNI_CP2_DCP_ROUNDTRIP "
            f"loss={loss.item():.8f} changed_tensors={len(changed)} "
            "post_restore_max_logit_abs_diff=0.00000000",
            flush=True,
        )

    del output, post_update_output, restored_output, optimizer, model
    del core_model, restored_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()


def test_nemotron_omni_cp2_training_and_checkpoint_roundtrip(
    distributed_test_runner,
    tmp_path,
):
    test_fn = functools.partial(
        _run_training_checkpoint_roundtrip,
        checkpoint_dir=str(tmp_path / "nemotron_omni_cp2_dcp"),
    )
    distributed_test_runner(test_fn, world_size=2)

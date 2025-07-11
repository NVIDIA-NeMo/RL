# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional, Tuple, Union

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from nemo_rl.distributed.model_utils import (
    DistributedLogprob,
    DistributedTokenLevelEntropy,
)


class RotaryEmbedParallel(SequenceParallel):
    """Custom SequenceParallel class for Qwen2 / Gemma3 rotary embeddings because the input is a tuple."""

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        new_inputs = list(inputs)

        if not isinstance(inputs[0], DTensor):
            """Guard the metadata for Sequence Parallel here"""
            try:
                new_inputs[0] = DTensor.from_local(
                    local_tensor=inputs[0],
                    device_mesh=device_mesh,
                    placements=sequence_sharding,
                    run_check=True,
                )
            except ValueError as e:
                raise ValueError(
                    f"Failed to shard tensor for sequence parallelism. Local Shape is ({inputs[0].shape}) "
                    f"at rank {torch.distributed.get_rank()}. Different TP ranks must have the same shape. "
                    f"Original error: {str(e)}"
                ) from e

        if not isinstance(inputs[1], DTensor):
            new_inputs[1] = DTensor.from_local(
                local_tensor=inputs[1],
                device_mesh=device_mesh,
                placements=(Replicate(),),
                run_check=False,
            )

        return type(inputs)(new_inputs)

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return type(outputs)([o.to_local() if use_local_output else o for o in outputs])


def _parallelize_nm5_h(
    model,
    dp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
    cpu_offload: bool = False,
    custom_parallel_plan: Optional[Union[dict, str]] = None,
) -> torch.nn.Module:
    """Parallelize a NemotronHForCausalLM model across data and tensor parallel dimensions."""
    assert not sequence_parallel, (
        "Sequence parallelism is not supported for NemotronHForCausalLM"
    )

    assert custom_parallel_plan is None, (
        "Custom parallel plan is not supported for NemotronHForCausalLM"
    )

    model_tp_plan = {
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    mlp_tp_plan = {
        "mixer.up_proj": ColwiseParallel(),
        "mixer.down_proj": RowwiseParallel(),
    }

    layers: torch.nn.ModuleList = model.backbone.layers

    parallelize_module(model, tp_mesh, model_tp_plan)

    for layer in model.backbone.layers:
        if layer.block_type == "mlp":
            parallelize_module(layer, tp_mesh, mlp_tp_plan)

    if activation_checkpointing:
        for i in range(len(layers)):
            if layers[i].block_type == "mlp":
                layers[i] = checkpoint_wrapper(layers[i])

        if layers[i].block_type == "mamba":
            layers[i] = checkpoint_wrapper(layers[i])

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    offload_policy = (
        CPUOffloadPolicy(pin_memory=False)
        if cpu_offload
        else torch.distributed.fsdp.OffloadPolicy
    )

    for layer in layers:
        fully_shard(
            layer, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
        )

    # do not reshard after forward for root model

    # because its parameters will be used in backward immediately

    return fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=False,
    )


def _parallelize_gemma3(
    model: Union[Gemma3ForCausalLM, Gemma3ForConditionalGeneration],
    dp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: torch.distributed.fsdp.OffloadPolicy,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
):
    """Parallelizes a Gemma3ForCausalLM model across data parallel dimensions.

    Tensor parallelism is not supported for Gemma3 models because of tied word embeddings.
    """
    if isinstance(model, Gemma3ForConditionalGeneration):
        layers = model.language_model.model.layers
        model_prefix = "language_model.model"
        num_attention_heads = model.config.text_config.num_attention_heads
        num_key_value_heads = model.config.text_config.num_key_value_heads
    else:
        layers = model.model.layers
        model_prefix = "model"
        num_attention_heads = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads

    if tp_mesh.size() > 1:
        assert num_key_value_heads % tp_mesh.size() == 0, (
            f"num_key_value_heads ({num_key_value_heads}) must be divisible by TP size ({tp_mesh.size()})"
        )
        assert num_attention_heads % tp_mesh.size() == 0, (
            f"num_attention_heads ({num_attention_heads}) must be divisible by TP size ({tp_mesh.size()})"
        )

        # For gemma3 models, we don't include the model.embed_tokens and lm_head in the
        # parallelization plans because they have tied weights.
        base_model_tp_plan = {
            f"{model_prefix}.layers.*.self_attn.q_proj": ColwiseParallel(),
            f"{model_prefix}.layers.*.self_attn.k_proj": ColwiseParallel(),
            f"{model_prefix}.layers.*.self_attn.v_proj": ColwiseParallel(),
            f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(),
            f"{model_prefix}.layers.*.mlp.up_proj": ColwiseParallel(),
            f"{model_prefix}.layers.*.mlp.gate_proj": ColwiseParallel(),
            f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(),
        }

        base_model_sp_plan = {
            f"{model_prefix}.embed_tokens": PrepareModuleOutput(
                output_layouts=Replicate(),
                desired_output_layouts=Shard(1),
                use_local_output=False,
            ),
            f"{model_prefix}.rotary_emb": RotaryEmbedParallel(use_local_output=True),
            f"{model_prefix}.rotary_emb_local": RotaryEmbedParallel(
                use_local_output=True
            ),
            f"{model_prefix}.layers.*.input_layernorm": SequenceParallel(),
            f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(
                output_layouts=Shard(1)
            ),
            f"{model_prefix}.layers.*.post_attention_layernorm": SequenceParallel(),
            f"{model_prefix}.layers.*.pre_feedforward_layernorm": SequenceParallel(),
            f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(
                output_layouts=Shard(1)
            ),
            f"{model_prefix}.layers.*.post_feedforward_layernorm": SequenceParallel(),
            f"{model_prefix}.norm": SequenceParallel(),
            f"{model_prefix}.lm_head": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
                use_local_output=True,
            ),
        }

        if sequence_parallel:
            # Enable sequence parallelism only if TP size > 1
            base_model_tp_plan.update(base_model_sp_plan)

        parallelize_module(model, tp_mesh, base_model_tp_plan)

    if activation_checkpointing:
        for i in range(len(layers)):
            layers[i].mlp = checkpoint_wrapper(layers[i].mlp)

    for layer in layers:
        fully_shard(
            layer, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
        )

    return fully_shard(
        model, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
    )


def _parallelize_llama(
    model: LlamaForCausalLM,
    dp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: torch.distributed.fsdp.OffloadPolicy,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
):
    """Parallelizes a LlamaForCausalLM model across data and tensor parallel dimensions."""
    if tp_mesh.size() > 1:
        assert not model.config.tie_word_embeddings, (
            "Tie word embeddings not supported when TP is enabled"
        )

        base_model_tp_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
            "lm_head": ColwiseParallel(
                output_layouts=Shard(-1), use_local_output=False
            ),
        }

        base_model_sp_plan = {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),
            "model.norm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False
            ),
        }

        if sequence_parallel:
            # Enable sequence parallelism only if TP size > 1
            base_model_tp_plan.update(base_model_sp_plan)

        parallelize_module(model, tp_mesh, base_model_tp_plan)

    if activation_checkpointing:
        for i in range(len(model.model.layers)):
            model.model.layers[i].mlp = checkpoint_wrapper(model.model.layers[i].mlp)

    for layer in model.model.layers:
        fully_shard(
            layer, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
        )

    return fully_shard(
        model, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
    )


def _parallelize_qwen(
    model: Union[Qwen2ForCausalLM, Qwen3ForCausalLM],
    dp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: torch.distributed.fsdp.OffloadPolicy,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
):
    """Parallelizes a Qwen2ForCausalLM model across data and tensor parallel dimensions."""

    class Qwen3QKNorm(SequenceParallel):
        @staticmethod
        def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
            input_tensor = inputs[0]

            if isinstance(input_tensor, DTensor):
                assert input_tensor.placements == (Shard(dim=2),)
            elif isinstance(input_tensor, torch.Tensor):
                # assume the input passed in already sharded on the sequence dim and create the DTensor
                return DTensor.from_local(
                    input_tensor, device_mesh, sequence_sharding, run_check=False
                )
            else:
                raise ValueError(
                    f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}"
                )

    if tp_mesh.size() > 1:
        assert not model.config.tie_word_embeddings, (
            "Tie word embeddings not supported when TP is enabled"
        )
        if sequence_parallel:
            base_model_tp_plan = {
                "lm_head": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Shard(-1),
                    use_local_output=False,
                ),
                "model.embed_tokens": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "model.rotary_emb": RotaryEmbedParallel(),
                "model.norm": SequenceParallel(),
                "model.layers.*.input_layernorm": SequenceParallel(),
                "model.layers.*.self_attn.q_proj": ColwiseParallel(
                    use_local_output=False
                ),
                "model.layers.*.self_attn.k_proj": ColwiseParallel(
                    use_local_output=False
                ),
                "model.layers.*.self_attn.v_proj": ColwiseParallel(
                    use_local_output=False
                ),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(
                    output_layouts=Shard(1)
                ),
                "model.layers.*.self_attn.q_norm": Qwen3QKNorm(),
                "model.layers.*.self_attn.k_norm": Qwen3QKNorm(),
                "model.layers.*.post_attention_layernorm": SequenceParallel(),
                "model.layers.*.mlp.up_proj": ColwiseParallel(),
                "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(
                    output_layouts=Shard(1)
                ),
            }

        else:
            base_model_tp_plan = {
                "lm_head": ColwiseParallel(
                    output_layouts=Shard(-1), use_local_output=False
                ),
                "model.embed_tokens": RowwiseParallel(
                    input_layouts=Replicate(),
                ),
                "model.layers.*.self_attn.q_proj": ColwiseParallel(),
                "model.layers.*.self_attn.k_proj": ColwiseParallel(),
                "model.layers.*.self_attn.v_proj": ColwiseParallel(),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(),
                "model.layers.*.mlp.up_proj": ColwiseParallel(),
                "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(),
            }

        parallelize_module(model, tp_mesh, base_model_tp_plan)

    if activation_checkpointing:
        for i in range(len(model.model.layers)):
            model.model.layers[i].mlp = checkpoint_wrapper(model.model.layers[i].mlp)

    for layer in model.model.layers:
        fully_shard(
            layer, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
        )

    return fully_shard(
        model, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy
    )


PARALLIZE_FUNCTIONS = {
    Qwen2ForCausalLM: _parallelize_qwen,
    Qwen3ForCausalLM: _parallelize_qwen,
    LlamaForCausalLM: _parallelize_llama,
    # gemma-3-1b-it uses Gemma3ForCausalLM since it is a text-only model
    Gemma3ForCausalLM: _parallelize_gemma3,
    # The larger gemma models use Gemma3ForConditionalGeneration, which are for text-image input
    Gemma3ForConditionalGeneration: _parallelize_gemma3,
}


def _parallelize_model(
    model: Union[Qwen2ForCausalLM, LlamaForCausalLM],
    dp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
    cpu_offload: bool = False,
):
    """Parallelize a model using DTensor.

    Args:
        model (Union[Qwen2ForCausalLM, LlamaForCausalLM]): The model to parallelize.
        dp_mesh (DeviceMesh): Device mesh for data parallelism.
        tp_mesh (DeviceMesh): Device mesh for tensor parallelism.
        param_dtype (torch.dtype): Data type for model parameters.
        sequence_parallel (bool, optional): Whether to use sequence parallelism. Defaults to False.
        activation_checkpointing (bool, optional): Whether to use activation checkpointing. Defaults to False.
        cpu_offload (bool, optional): Whether to enable cpu offloading for FSDP. Defaults to False.

    Returns:
        The parallelized model.

    Raises:
        ValueError: If the model type is not supported for parallelization.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    offload_policy = (
        CPUOffloadPolicy(pin_memory=False)
        if cpu_offload
        else torch.distributed.fsdp.OffloadPolicy
    )

    model_cls = type(model)
    if model_cls.__name__ == "NemotronHForCausalLM":
        # need to do something special for nm5, since it's harder to shard the mamba layers
        # nm5 is not importable, so we check the __name__ attribute
        return _parallelize_nm5_h(
            model,
            dp_mesh,
            tp_mesh,
            param_dtype,
            sequence_parallel,
            activation_checkpointing,
            cpu_offload,
            None,
        )

    if model_cls not in PARALLIZE_FUNCTIONS:
        raise ValueError(f"Model {model_cls} not supported as part of dtensor")

    func = PARALLIZE_FUNCTIONS[type(model)]

    return func(
        model,
        dp_mesh,
        tp_mesh,
        mp_policy,
        offload_policy,
        sequence_parallel,
        activation_checkpointing,
    )


def to_local_if_dtensor(tensor: Union[torch.Tensor, DTensor]) -> torch.Tensor:
    """Returns the local shard of the given tensor if it is a DTensor.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/605f618f237cda8fa80132bc2ccff933512d5a0d/megatron/core/utils.py#L746
    """
    with torch.no_grad():
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def clip_grad_by_total_norm_(
    parameters: Union[List[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    max_grad_norm: Union[int, float],
    total_norm: float,
    dtype: torch.dtype = torch.float32,
):
    """Clips gradient of an iterable of parameters by total norm.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/a695b2bd2a0ca9ca63385a48c41a1c5a033cdd1e/megatron/core/optimizer/clip_grads.py#L138

    Note that the gradients are modified in place.

    Args:
        parameters (Union[List[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]]):
            An iterable of Tensors or DTensors, or a single Tensor or DTensor
            that will have gradients normalized.
        max_grad_norm (Union[float, int]): Maximum norm of the gradients.
        total_norm (float): The pre-computed total norm of the gradients to use for scaling.
    """
    if isinstance(parameters, (torch.Tensor, DTensor)):
        parameters = [parameters]

    # Grads.
    grads = [
        to_local_if_dtensor(p.grad.detach()).to(dtype)
        for p in parameters
        if p.grad is not None
    ]

    # Scale.
    clip_coeff = max_grad_norm / (total_norm + 1.0e-6)

    if clip_coeff < 1.0:
        for g in grads:
            g.mul_(clip_coeff)


def get_grad_sparsity(
    named_parameters: Union[
        List[Tuple[str, Union[torch.Tensor, DTensor]]],
        Tuple[str, Union[torch.Tensor, DTensor]],
    ],
    dp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
) -> float:
    """NOTE: This function is somewhat incorrect if the grads are not all reduced across the data parallel GPUs.
        * this is because the zeros might not be all in the same place across DP ranks

    Args:
        parameters (Union[List[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]]):
            An iterable of Tensors or DTensors, or a single Tensor or DTensor
            that will have gradient norm calculated.
        dp_group (torch.distributed.ProcessGroup): Process group for data parallel communication.
        tp_group (torch.distributed.ProcessGroup): Process group for tensor parallel communication.
        norm_type (Union[int, float]): Type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        float: Total norm of the gradients (viewed as a single vector)
    """
    grad_sparsity_dict = {}
    grads_dict = {
        name: to_local_if_dtensor(p.grad.detach())
        for name, p in sorted(named_parameters)
        if p.grad is not None
    }

    total_num_zeros, total_num_elements = 0, 0
    for name, grad in grads_dict.items():
        num_zeros = grad.numel() - grad.count_nonzero()
        num_elements = grad.numel()

        to_reduce = torch.tensor(
            [num_zeros, num_elements], dtype=torch.float32, device="cuda"
        )
        torch.distributed.all_reduce(to_reduce, group=dp_group)
        torch.distributed.all_reduce(to_reduce, group=tp_group)
        grad_sparsity_dict[name] = (to_reduce[0] / to_reduce[1]).item()
        total_num_zeros += to_reduce[0]
        total_num_elements += to_reduce[1]

    return (total_num_zeros / total_num_elements).item(), grad_sparsity_dict


def get_grad_norm(
    parameters: Union[List[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    dp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    norm_type: Union[int, float] = 2,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Calculate the norm of gradients.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/a695b2bd2a0ca9ca63385a48c41a1c5a033cdd1e/megatron/core/optimizer/clip_grads.py#L51

    Args:
        parameters (Union[List[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]]):
            An iterable of Tensors or DTensors, or a single Tensor or DTensor
            that will have gradient norm calculated.
        dp_group (torch.distributed.ProcessGroup): Process group for data parallel communication.
        tp_group (torch.distributed.ProcessGroup): Process group for tensor parallel communication.
        norm_type (Union[int, float]): Type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        float: Total norm of the gradients (viewed as a single vector)
    """
    if isinstance(parameters, (torch.Tensor, DTensor)):
        parameters = [parameters]

    # Grads.
    grads_for_norm = [
        to_local_if_dtensor(p.grad.detach()).to(dtype)
        for p in parameters
        if p.grad is not None
    ]

    is_finite = [grad.isfinite().all() for grad in grads_for_norm]

    if not all(is_finite):
        print(
            "rank",
            torch.distributed.get_rank(),
            "grads are not finite",
            is_finite,
        )

    # Norm parameters.
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == torch.inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor(
            [float(total_norm)], dtype=torch.float, device="cuda"
        )
        # Take max across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=dp_group
        )
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=tp_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm**norm_type

        print("### rank", torch.distributed.get_rank(), "total_norm", total_norm)

        total_norm = total_norm.cuda()
        # Sum across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=dp_group
        )
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm


def token_level_entropy_from_vocab_parallel_logits(
    vocab_parallel_logits: DTensor,
):
    """Computes token-level entropy from vocabulary-parallel logits.

    Args:
        vocab_parallel_logits (DTensor): Logits distributed across tensor parallel workers,
            with shape [batch_size, seq_len, vocab_size/tp_size].

    Returns:
        torch.Tensor: Token-level entropy tensor with shape [batch_size, seq_len-1].
            The sequence dimension is reduced by 1 due to the target shifting.

    NOTE: this function assumes the vocab dimension includes all valid tokens(no padding)
    """
    tp_mesh = vocab_parallel_logits.device_mesh

    entropy = DistributedTokenLevelEntropy.apply(
        vocab_parallel_logits.to_local(),
        tp_mesh.get_group(),
        not torch.is_grad_enabled(),
    ).contiguous()
    return entropy[:, :-1]


def get_logprobs_from_vocab_parallel_logits(
    vocab_parallel_logits: DTensor, input_ids: torch.Tensor
):
    """Computes log probabilities from vocabulary-parallel logits.

    This function takes logits that are sharded across the vocabulary dimension (tensor parallel)
    and computes the log probabilities for the given input IDs.

    Args:
        vocab_parallel_logits (DTensor): Logits distributed across tensor parallel workers,
            with shape [batch_size, seq_len, vocab_size/tp_size].
        input_ids (torch.Tensor): Input token IDs for which to compute log probabilities,
            with shape [batch_size, seq_len].

    Returns:
        torch.Tensor: Log probabilities for the given input IDs.
    """
    tp_mesh = vocab_parallel_logits.device_mesh
    tp_rank: int = tp_mesh.get_local_rank()

    vocab_interval_per_rank = vocab_parallel_logits.shape[-1] // tp_mesh.size()

    target = input_ids.roll(shifts=-1, dims=-1)
    probs = DistributedLogprob.apply(
        vocab_parallel_logits.to_local(),
        target,
        vocab_interval_per_rank * tp_rank,
        (tp_rank + 1) * vocab_interval_per_rank,
        tp_mesh.get_group(),
        not torch.is_grad_enabled(),
    ).contiguous()
    return probs[:, :-1]

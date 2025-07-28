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

from functools import lru_cache
from types import FunctionType
from typing import Callable, Optional, Union, cast

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from nemo_rl.distributed.model_utils import dtensor_from_parallel_logits_to_logprobs
from nemo_rl.models.policy.utils import import_class_from_path


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


def _parallelize_gemma3(
    model: Union[Gemma3ForCausalLM, Gemma3ForConditionalGeneration],
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a Gemma3ForCausalLM model across data parallel dimensions.

    Tensor parallelism is not supported for Gemma3 models because of tied word embeddings.
    """
    if isinstance(model, Gemma3ForConditionalGeneration):
        model_prefix = "model.language_model"
    else:
        model_prefix = "model"

    # For gemma3 models, we don't include the model.embed_tokens and lm_head in the
    # parallelization plans because they have tied weights.
    base_model_tp_plan: dict[str, ParallelStyle] = {
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
        f"{model_prefix}.rotary_emb_local": RotaryEmbedParallel(use_local_output=True),
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
        "lm_head": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,
        ),
    }

    if sequence_parallel:
        # Enable sequence parallelism only if TP size > 1
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _parallelize_llama(
    model: LlamaForCausalLM,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a LlamaForCausalLM model across data and tensor parallel dimensions."""
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
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
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _parallelize_qwen(
    model: Union[Qwen2ForCausalLM, Qwen3ForCausalLM],
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a Qwen2ForCausalLM model across data and tensor parallel dimensions."""

    class Qwen3QKNorm(SequenceParallel):
        @staticmethod
        def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
            input_tensor = inputs[0]

            if isinstance(input_tensor, DTensor):
                assert input_tensor.placements == (Shard(dim=2),)
                return input_tensor
            elif isinstance(input_tensor, torch.Tensor):
                # assume the input passed in already sharded on the sequence dim and create the DTensor
                return DTensor.from_local(
                    input_tensor, device_mesh, sequence_sharding, run_check=False
                )
            else:
                raise ValueError(
                    f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}"
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
            "model.layers.*.self_attn.q_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "model.layers.*.self_attn.q_norm": Qwen3QKNorm(),
            "model.layers.*.self_attn.k_norm": Qwen3QKNorm(),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
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

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


PARALLIZE_FUNCTIONS: dict[
    type[torch.nn.Module], Callable[..., dict[str, ParallelStyle]]
] = {
    Qwen2ForCausalLM: _parallelize_qwen,
    Qwen3ForCausalLM: _parallelize_qwen,
    LlamaForCausalLM: _parallelize_llama,
    # gemma-3-1b-it uses Gemma3ForCausalLM since it is a text-only model
    Gemma3ForCausalLM: _parallelize_gemma3,
    # The larger gemma models use Gemma3ForConditionalGeneration, which are for text-image input
    Gemma3ForConditionalGeneration: _parallelize_gemma3,
}


@lru_cache
def translate_parallel_style(style: str):
    """Translate parallel style str to parallel type.

    Taken and modified from: https://github.com/NVIDIA/NeMo/blob/6c6169db01bcca73ae8ad3ac35242fadbb9a78ba/nemo/lightning/pytorch/strategies/utils.py#L547
    """
    assert isinstance(style, str), (
        f"parallel style type should be str, but got {type(style)}"
    )

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "sequence_parallel":
        return SequenceParallel()
    else:
        raise ValueError(f"Unknown parallel style: {style}")


def get_hf_tp_plan(model: PreTrainedModel):
    """Get the Hugging Face tensor parallel plan from the model.

    This function:
    - Retrieves TP strategies from model class, instance, and inner model levels.
    - Handles special cases for `embed_tokens` and `lm_head` for speed up.
    - Converts string-based parallel styles to DTensor parallelization strategies.

    Taken and modified from: https://github.com/NVIDIA/NeMo/blob/6c6169db01bcca73ae8ad3ac35242fadbb9a78ba/nemo/lightning/pytorch/strategies/utils.py#L532

    Args:
        model: A Hugging Face model instance

    Returns:
        dict: A dictionary mapping model component paths to their parallelization strategies

    Raises:
        AssertionError: If no TP plan is found
    """
    model_cls = type(model)
    if model_cls == Gemma3ForConditionalGeneration:
        inner_model = model.language_model
        model_prefix = "language_model"
    else:
        inner_model = model.model
        model_prefix = "model"

    hf_tp_plan = {}

    # model_cls._tp_plan will override model_cls after xxxForCausalLM.post_init() (transformers==4.51.3)
    if hasattr(model_cls, "_tp_plan") and model_cls._tp_plan is not None:
        assert isinstance(model_cls._tp_plan, dict), (
            f"model_cls._tp_plan is not a dict: {model_cls._tp_plan}"
        )
        hf_tp_plan.update(model_cls._tp_plan)

    if hasattr(model, "_tp_plan") and model._tp_plan is not None:
        hf_tp_plan.update(model._tp_plan)

    if hasattr(inner_model, "_tp_plan") and inner_model._tp_plan is not None:
        hf_tp_plan.update(
            {f"{model_prefix}.{k}": v for k, v in inner_model._tp_plan.items()}
        )

    assert len(hf_tp_plan) > 0, (
        f"Hugging Face tp plan is not supported for {model_cls}, please set dtensor_cfg.tensor_parallel_size to 1 or provide a custom_parallel_plan. "
        "The usage example of custom_parallel_plan can refer to `docs/design-docs/fsdp2-parallel-plan.md`."
    )

    # hf tp plan not contain embed_tokens, we add it and set to rowwise_rep
    if (
        f"{model_prefix}.embed_tokens" not in hf_tp_plan
        and not model.config.tie_word_embeddings
    ):
        hf_tp_plan[f"{model_prefix}.embed_tokens"] = "rowwise_rep"

    for k, v in hf_tp_plan.items():
        # speed up the tp plan for lm_head
        if (
            k == "lm_head"
            and v == "colwise_rep"
            and not model.config.tie_word_embeddings
        ):
            hf_tp_plan[k] = ColwiseParallel(
                output_layouts=Shard(-1), use_local_output=False
            )
        else:
            hf_tp_plan[k] = translate_parallel_style(v)

    return hf_tp_plan


def _parallelize_model(
    model: Union[
        Qwen2ForCausalLM,
        LlamaForCausalLM,
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
    ],
    dp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
    cpu_offload: bool = False,
    custom_parallel_plan: Optional[Union[dict, str]] = None,
):
    """Parallelize a model using DTensor.

    Args:
        model: The model to parallelize.
        dp_mesh: Device mesh for data parallelism.
        tp_mesh: Device mesh for tensor parallelism.
        param_dtype: Data type for model parameters.
        sequence_parallel: Whether to use sequence parallelism. Defaults to False.
        activation_checkpointing: Whether to use activation checkpointing. Defaults to False.
        cpu_offload: Whether to enable cpu offloading for FSDP. Defaults to False.
        custom_parallel_plan: Custom parallel plan for the model. Defaults to None.
            If it's a dict, it will be used as the parallel plan directly.
            If it's a string, it must be a path that points to a dict or a function that returns a dict.
            The usage example can refer to `docs/design-docs/fsdp2-parallel-plan.md`.

    Returns:
        The parallelized model.

    Raises:
        ValueError: If the model type is not supported for parallelization.
    """
    model_cls = type(model)
    if model_cls == Gemma3ForConditionalGeneration:
        layers: torch.nn.ModuleList = model.language_model.layers  # type: ignore
        num_attention_heads = model.config.text_config.num_attention_heads
        num_key_value_heads = model.config.text_config.num_key_value_heads
    else:
        layers: torch.nn.ModuleList = model.model.layers  # type: ignore
        num_attention_heads = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads

    if tp_mesh.size() > 1:
        assert num_key_value_heads % tp_mesh.size() == 0, (
            f"num_key_value_heads ({num_key_value_heads}) must be divisible by TP size ({tp_mesh.size()})"
        )
        assert num_attention_heads % tp_mesh.size() == 0, (
            f"num_attention_heads ({num_attention_heads}) must be divisible by TP size ({tp_mesh.size()})"
        )

        # first use user's custom parallel plan
        if custom_parallel_plan is not None:
            if isinstance(custom_parallel_plan, dict):
                model_parallel_plan = custom_parallel_plan
            else:
                try:
                    model_parallel_plan = import_class_from_path(custom_parallel_plan)
                    if isinstance(model_parallel_plan, FunctionType):
                        model_parallel_plan = model_parallel_plan()
                    assert isinstance(model_parallel_plan, dict)
                except:
                    raise ValueError(
                        f"Your custom parallel plan is `{custom_parallel_plan}` which is not valid. Please ensure it is one of the following:\n"
                        "1. A dictionary\n"
                        "2. A path to a dictionary\n"
                        "3. A path to a function that returns a dictionary"
                    )
            print("Using custom parallel plan.")

        # second use our optimized parallel plan
        elif model_cls in PARALLIZE_FUNCTIONS:
            # try to use our optimized parallel plan
            try:
                func = PARALLIZE_FUNCTIONS[model_cls]
                model_parallel_plan = func(model, sequence_parallel)
                print("Using optimized parallel plan.")
            # fall back to the HF tp plan
            except Exception as e:
                print(
                    f"Optimized parallel plan is not available: {e}. Falling back to the HF tp plan."
                )
                assert not sequence_parallel, (
                    "sequence_parallel is not support in HF tp plan."
                )
                model_parallel_plan = get_hf_tp_plan(model)

        # final use the default HF tp plan
        else:
            # optimized parallel plan is not support for the model class
            print(
                f"Optimized parallel plan is not support for {model_cls}. Falling back to the HF tp plan."
            )
            assert not sequence_parallel, (
                "sequence_parallel is not support in HF tp plan."
            )
            model_parallel_plan = get_hf_tp_plan(model)

        parallelize_module(model, tp_mesh, model_parallel_plan)

    if activation_checkpointing:
        for i in range(len(layers)):
            layers[i].mlp = checkpoint_wrapper(layers[i].mlp)  # type: ignore

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    offload_policy = (
        CPUOffloadPolicy(pin_memory=False) if cpu_offload else OffloadPolicy()
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


def to_local_if_dtensor(tensor: Union[torch.Tensor, DTensor]) -> torch.Tensor:
    """Returns the local shard of the given tensor if it is a DTensor.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/605f618f237cda8fa80132bc2ccff933512d5a0d/megatron/core/utils.py#L746
    """
    with torch.no_grad():
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def clip_grad_by_total_norm_(
    parameters: Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    max_grad_norm: Union[int, float],
    total_norm: float,
    dtype: torch.dtype = torch.float32,
):
    """Clips gradient of an iterable of parameters by total norm.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/a695b2bd2a0ca9ca63385a48c41a1c5a033cdd1e/megatron/core/optimizer/clip_grads.py#L138

    Note that the gradients are modified in place.

    Args:
        parameters (Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]]):
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


def get_grad_norm(
    parameters: Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    dp_cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    norm_type: Union[int, float] = 2,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Calculate the norm of gradients.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/a695b2bd2a0ca9ca63385a48c41a1c5a033cdd1e/megatron/core/optimizer/clip_grads.py#L51

    Args:
        parameters (Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]]):
            An iterable of Tensors or DTensors, or a single Tensor or DTensor
            that will have gradient norm calculated.
        dp_group (torch.distributed.ProcessGroup): Process group for data parallel communication.
        cp_group (torch.distributed.ProcessGroup): Process group for context parallel communication.
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

    # Norm parameters.
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == torch.inf:
        total_norm = max(grad.abs().max().item() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor(
            [float(total_norm)], dtype=torch.float, device="cuda"
        )
        # Take max across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=dp_cp_group
        )

        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=tp_group
        )
        total_norm = float(total_norm_cuda[0].item())

    else:
        total_norm = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += torch.pow(grad_norm, norm_type)

        # Sum across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=dp_cp_group
        )

        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)  # type: ignore

    return total_norm


def get_logprobs_from_vocab_parallel_logits(
    vocab_parallel_logits: DTensor,
    input_ids: torch.Tensor | DTensor,
    seq_index: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """Computes log probabilities from vocabulary-parallel logits.

    This function takes logits that are sharded across the vocabulary dimension (tensor parallel)
    and computes the log probabilities for the given input IDs.

    Args:
        vocab_parallel_logits (DTensor): Logits distributed across tensor parallel workers,
            with shape [batch_size, seq_len, vocab_size/tp_size].
        input_ids (torch.Tensor | DTensor): Input token IDs for which to compute log probabilities,
            with shape [batch_size, seq_len].
        seq_index (Optional[torch.Tensor]): Sequence index for the input IDs,
            with shape [sequence_length].
        top_k (Optional[int]): Top-k sampling parameter. If provided, only the top-k tokens
            with highest probabilities are considered.
        top_p (Optional[float]): Top-p (nucleus) sampling parameter. If provided, only tokens
            with cumulative probability up to p are considered.

    Returns:
        torch.Tensor: Log probabilities for the given input IDs.
    """
    device_mesh = vocab_parallel_logits.device_mesh
    if seq_index is not None:
        assert (
            device_mesh.mesh_dim_names is not None
            and "cp" in device_mesh.mesh_dim_names
        ), "seq_index must be provided for cp sharded logits"

    tp_size = 1

    tp_group = device_mesh.get_group("tp")
    tp_rank = tp_group.rank()
    tp_size = tp_group.size()

    vocab_interval_per_rank = vocab_parallel_logits.shape[-1] // tp_size

    return dtensor_from_parallel_logits_to_logprobs(
        vocab_parallel_logits.to_local(),
        input_ids,
        vocab_interval_per_rank * tp_rank,
        (tp_rank + 1) * vocab_interval_per_rank,
        tp_group,
        inference_only=not torch.is_grad_enabled(),
        seq_index=seq_index,
        top_k=top_k,
        top_p=top_p,
    )

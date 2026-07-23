# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import lru_cache, reduce
from types import FunctionType
from typing import Any, Dict, Generator, List, Optional, Union

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForConditionalGeneration,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
)
from transformers.models.llava_next_video.modeling_llava_next_video import (
    LlavaNextVideoForConditionalGeneration,
)
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionForConditionalGeneration,
)
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3ForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
)
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration

from nemo_automodel.components.distributed.optimized_tp_plans import PARALLELIZE_FUNCTIONS
from nemo_automodel.components.distributed.parallel_styles import translate_to_lora

# TODO(boxiangw): Change to MegatronFSDP once it got published
HAVE_MEGATRON_FSDP = False
try:
    from megatron_fsdp import fully_shard as megatron_fsdp_fully_shard

    HAVE_MEGATRON_FSDP = True
except:
    pass

# Import as module so tests can patch nemo_automodel.components.distributed.parallelizer_utils.fully_shard_by_dtype
import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils

logger = logging.getLogger(__name__)


class ParallelizationStrategy(ABC):
    """Abstract base class for model parallelization strategies."""

    @abstractmethod
    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
        use_hf_tp_plan: bool = False,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
    ) -> nn.Module:
        """Apply parallelization strategy to the model."""
        pass


class DefaultParallelizationStrategy(ParallelizationStrategy):
    """Default parallelization strategy used by most models."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
        use_hf_tp_plan: bool = False,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
    ) -> nn.Module:
        """Apply the default parallelization flow."""
        tp_mesh = device_mesh[tp_mesh_name]

        # Set FSDP sharding mesh to context parallel mesh if CP > 1, else default to the data parallel mesh.
        # if dp_replicate_size > 1, use HSDP, else use FSDP
        dp_mesh_dim_names = (dp_replicate_mesh_name, dp_shard_cp_mesh_name)
        dp_mesh = device_mesh[dp_mesh_dim_names]

        # Extract layers from the model for parallelization
        layers = _extract_model_layers(model)

        # TP sharding with enhanced plan generation
        if tp_mesh.size() > 1:
            # Validate that attention heads are divisible by TP size
            validate_tp_mesh(model, tp_mesh)

            # Generate or use tensor parallel plan
            model_parallel_plan = {
                k: translate_to_lora(v)
                for k, v in _get_parallel_plan(
                    model,
                    sequence_parallel,
                    tp_shard_plan,
                    use_hf_tp_plan=use_hf_tp_plan,
                ).items()
            }

            # Apply tensor parallelism
            if model_parallel_plan:
                parallelize_module(model, tp_mesh, model_parallel_plan)

        # Apply activation checkpointing to linear layers if requested
        if activation_checkpointing:
            # Disable KV caching during training to ensure deterministic
            # shapes between forward and checkpoint recomputation.
            if hasattr(model, "config") and getattr(model.config, "use_cache", None) is not False:
                try:
                    model.config.use_cache = False
                except Exception:
                    pass

            for i, layer in enumerate(layers):
                if hasattr(layer, "mlp"):
                    layers[i].mlp = checkpoint_wrapper(layer.mlp)
                if hasattr(layer, "self_attn"):
                    layers[i].self_attn = checkpoint_wrapper(layers[i].self_attn)  # type: ignore

                if hasattr(layer, "input_layernorm"):
                    layers[i].input_layernorm = checkpoint_wrapper(
                        layers[i].input_layernorm  # type: ignore
                    )

                if hasattr(layer, "post_attention_layernorm"):
                    layers[i].post_attention_layernorm = checkpoint_wrapper(
                        layers[i].post_attention_layernorm  # type: ignore
                    )

        # Set up mixed precision policy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )

        # Find transformer layers and apply parallelisms
        apply_fsdp2_sharding_recursively(model, dp_mesh, mp_policy, offload_policy)

        # Apply FSDP to the root model
        # Do not reshard after forward for root model because its parameters
        # will be used in backward immediately
        model = fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,
            offload_policy=offload_policy,
            shard_placement_fn=parallelizer_utils.attr_shard_placement_fn,
        )

        return model


class NemotronHParallelizationStrategy(ParallelizationStrategy):
    """Specialized parallelization strategy for NemotronH models."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
    ) -> nn.Module:
        """Apply NemotronH-specific parallelization."""
        assert not sequence_parallel, "Sequence parallelism is not supported for NemotronHForCausalLM"
        logger.info("Custom parallel plan is not supported for NemotronHForCausalLM. Using NemotronH-specific TP plan.")

        # Different NemotronH custom-code variants name the backbone differently:
        #   Nano-30B uses model.backbone; Super-120B uses model.model.
        # Pick whichever attribute holds the NemotronHModel.
        _backbone = getattr(model, "backbone", None)
        if _backbone is None:
            _backbone = getattr(model, "model", None)
        assert _backbone is not None and hasattr(_backbone, "layers"), (
            f"{type(model).__name__}: expected .backbone or .model with .layers"
        )
        layers: torch.nn.ModuleList = _backbone.layers
        tp_mesh = device_mesh[tp_mesh_name]
        if tp_mesh.size() > 1:
            model_tp_plan: dict[str, ParallelStyle] = {
                "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
            }

            mlp_tp_plan: dict[str, ParallelStyle] = {
                "mixer.up_proj": ColwiseParallel(),
                "mixer.down_proj": RowwiseParallel(),
            }


            parallelize_module(model, tp_mesh, model_tp_plan)

            for layer in _backbone.layers:
                if layer.block_type == "mlp":
                    parallelize_module(layer, tp_mesh, mlp_tp_plan)

        if activation_checkpointing:
            for i in range(len(layers)):
                if layers[i].block_type == "mlp":
                    layers[i] = checkpoint_wrapper(layers[i])

                if layers[i].block_type == "mamba":
                    layers[i] = checkpoint_wrapper(layers[i])

        dp_mesh_dim_names = (dp_replicate_mesh_name, dp_shard_cp_mesh_name)
        dp_mesh = device_mesh[dp_mesh_dim_names]

        for layer in layers:
            parallelizer_utils.fully_shard_by_dtype(
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
            shard_placement_fn=parallelizer_utils.attr_shard_placement_fn,
        )


class WanParallelizationStrategy(ParallelizationStrategy):
    """Parallelization strategy for Wan-style transformer modules used in Diffusers.

    Applies TP to condition embedders, FFN projections in each block, and final projection,
    then applies FSDP sharding similarly to other strategies.
    """

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
    ) -> nn.Module:
        # Not using custom tp_shard_plan; apply Wan-specific plan
        tp_mesh = device_mesh[tp_mesh_name]
        dp_mesh_dim_names = (dp_replicate_mesh_name, dp_shard_cp_mesh_name)
        dp_mesh = device_mesh[dp_mesh_dim_names]

        # Apply TP only when TP group size > 1
        if tp_mesh.size() > 1:
            # Condition embedders if present
            try:
                if hasattr(model, "condition_embedder"):
                    cond = model.condition_embedder
                    if hasattr(cond, "text_embedder"):
                        cond.text_embedder = parallelize_module(
                            cond.text_embedder,
                            tp_mesh,
                            {
                                "linear_1": ColwiseParallel(),
                                "linear_2": RowwiseParallel(),
                            },
                        )
                    if hasattr(cond, "time_embedder"):
                        cond.time_embedder = parallelize_module(
                            cond.time_embedder,
                            tp_mesh,
                            {
                                "linear_1": ColwiseParallel(),
                                "linear_2": RowwiseParallel(),
                            },
                        )
                    if hasattr(cond, "time_proj"):
                        cond.time_proj = parallelize_module(
                            cond.time_proj,
                            tp_mesh,
                            {"": ColwiseParallel()},
                        )
            except Exception as e:
                logger.warning(f"Wan strategy: failed to TP condition embedders: {e}")

            # Blocks FFN and final projection
            try:
                if hasattr(model, "blocks"):
                    for block in model.blocks:
                        if hasattr(block, "ffn"):
                            block.ffn = parallelize_module(
                                block.ffn,
                                tp_mesh,
                                {
                                    "net.0.proj": ColwiseParallel(),
                                    "net.2": RowwiseParallel(),
                                },
                            )
                if hasattr(model, "proj_out"):
                    model.proj_out = parallelize_module(model.proj_out, tp_mesh, {"": RowwiseParallel()})
            except Exception as e:
                logger.warning(f"Wan strategy: failed to TP blocks/proj_out: {e}")

        # Mixed precision default like Default strategy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )

        # Apply FSDP sharding recursively and to root
        apply_fsdp2_sharding_recursively(model, dp_mesh, mp_policy, offload_policy)

        return fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )


# Strategy registry mapping model class names to parallelization strategies
PARALLELIZATION_STRATEGIES: Dict[str, ParallelizationStrategy] = {
    "NemotronHForCausalLM": NemotronHParallelizationStrategy(),
    "WanTransformer3DModel": WanParallelizationStrategy(),
}

# Default strategy instance
_DEFAULT_STRATEGY = DefaultParallelizationStrategy()


def get_parallelization_strategy(model: nn.Module) -> ParallelizationStrategy:
    """Get the appropriate parallelization strategy for the given model."""
    model_name = type(model).__name__
    return PARALLELIZATION_STRATEGIES.get(model_name, _DEFAULT_STRATEGY)


def register_parallel_strategy(arg=None, *, name: Optional[str] = None):
    """Decorator to register out-of-tree parallelism strategies.

    Supports:
    - @register_parallel_strategy(name="CustomModelName")
    """

    def _register(cls):
        # The decorator receives a class, not an instance.
        assert isinstance(cls, type) and issubclass(cls, ParallelizationStrategy), (
            f"cls must be a subclass of ParallelizationStrategy, but got {type(cls)} {cls}"
        )
        assert name is not None, "name is required"
        assert name not in PARALLELIZATION_STRATEGIES, f"name {name} already registered"
        PARALLELIZATION_STRATEGIES[name] = cls()
        return cls

    if name is None:
        raise ValueError("name is required")
    # If used with parentheses (possibly with arguments)
    return _register


def apply_fsdp2_sharding_recursively(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy],
    offload_policy: Optional[OffloadPolicy] = None,
) -> None:
    """
    Recursively apply FSDP2 sharding to modules, with optimizations for ModuleList.

    This utility function traverses a model hierarchy and applies FSDP2 sharding
    to each module. For ModuleList instances (commonly used for transformer layers),
    it applies an optimization where the last layer doesn't reshard after forward
    since FSDP2 will prefetch it immediately.

    Handles both single-level and nested ModuleList structures. If a ModuleList
    contains other ModuleLists, it will recurse into them instead of trying to
    wrap them (since ModuleList doesn't have a forward method).

    Args:
        module (nn.Module): The module to apply FSDP sharding to.
        mesh (DeviceMesh): The device mesh for FSDP sharding.
        mp_policy (Optional[MixedPrecisionPolicy]): Mixed precision policy for FSDP.
        offload_policy (Optional[OffloadPolicy]): CPU offload policy for FSDP.
            Defaults to None.

    Note:
        This function modifies the module in-place by replacing modules with their
        FSDP2-subclassed versions.
    """
    if isinstance(module, nn.ModuleList):
        for layer_id, child_module in enumerate(module):
            # If child is also a ModuleList (nested structure), recurse instead of wrapping
            # since ModuleList doesn't have a forward() method required by fully_shard
            if isinstance(child_module, nn.ModuleList):
                apply_fsdp2_sharding_recursively(child_module, mesh, mp_policy, offload_policy)
            else:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = int(layer_id) < len(module) - 1
                fully_shard(
                    child_module,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                    offload_policy=offload_policy,
                    shard_placement_fn=parallelizer_utils.attr_shard_placement_fn,
                )
                module[layer_id] = child_module
    else:
        for name, sub_module in module.named_children():
            apply_fsdp2_sharding_recursively(sub_module, mesh, mp_policy, offload_policy)


def get_hf_tp_shard_plan(model):
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

    # Handle VL models structure
    if model_cls in [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
    ]:
        inner_model = model.model.language_model
        model_prefix = "model.language_model"

    elif model_cls == Gemma3ForConditionalGeneration:
        inner_model = model.language_model
        model_prefix = "language_model"

    elif model_cls == Llama4ForConditionalGeneration:
        inner_model = model.language_model.model
        model_prefix = "language_model.model"

    elif model_cls in [
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextVideoForConditionalGeneration,
        LlavaOnevisionForConditionalGeneration,
    ]:
        inner_model = model.model.language_model
        model_prefix = "model.language_model"

    elif model_cls == Mistral3ForConditionalGeneration:
        inner_model = model.model.language_model
        model_prefix = "model.language_model"

    else:
        inner_model = model.model
        model_prefix = "model"

    hf_tp_plan = {}

    # model_cls._tp_plan will override model_cls after xxxForCausalLM.post_init() (transformers==4.51.3)
    if hasattr(model_cls, "_tp_plan") and model_cls._tp_plan is not None:
        assert isinstance(model_cls._tp_plan, dict), f"model_cls._tp_plan is not a dict: {model_cls._tp_plan}"
        hf_tp_plan.update(model_cls._tp_plan)

    if hasattr(model, "_tp_plan") and model._tp_plan is not None:
        hf_tp_plan.update(model._tp_plan)

    if hasattr(inner_model, "_tp_plan") and inner_model._tp_plan is not None:
        hf_tp_plan.update({f"{model_prefix}.{k}": v for k, v in inner_model._tp_plan.items()})

    assert len(hf_tp_plan) > 0, (
        f"Hugging Face tp plan is not supported for {model_cls}, please set dtensor_cfg.tensor_parallel_size to 1 or provide a custom_parallel_plan. "
        "The usage example of custom_parallel_plan can refer to `docs/design-docs/fsdp2-parallel-plan.md`."
    )

    # hf tp plan not contain embed_tokens, we add it and set to rowwise_rep
    if f"{model_prefix}.embed_tokens" not in hf_tp_plan:
        hf_tp_plan[f"{model_prefix}.embed_tokens"] = "rowwise_rep"

    for k, v in hf_tp_plan.items():
        # speed up the tp plan for lm_head
        if (k == "lm_head" or k == "language_model.lm_head") and v == "colwise_rep":
            hf_tp_plan[k] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
        else:
            hf_tp_plan[k] = translate_to_torch_parallel_style(v)

    logger.info(f"Hugging Face tp plan: {hf_tp_plan}")
    return hf_tp_plan


def import_class_from_path(name: str) -> Any:
    """Import a class from a string path (e.g. 'torch.optim.AdamW').

    Args:
        full_path: Full path to class including module path and class name

    Returns:
        The imported class object
    """
    module_name, cls_name = name.rsplit(".", 1)
    cls_instance = getattr(importlib.import_module(module_name), cls_name)
    return cls_instance


def import_classes_from_paths(class_paths: List[str]):
    """
    Helper function to import classes from string paths.

    Args:
        class_paths (List[str]): The list of string paths to the classes.

    Returns:
        List of imported classes.
    """
    classes = []
    for path in class_paths:
        try:
            cls = import_class_from_path(path)
            classes.append(cls)
        except Exception as e:
            print(f"Warning: Could not import class from path '{path}': {e}")
    return classes


@lru_cache
def translate_to_torch_parallel_style(style: str):
    """
    Translates string descriptions to parallelism plans.

    In model configurations, we use a neutral type (string) to specify parallel
    styles, here we translate them into torch.distributed tensor-parallel
    types.
    """
    assert isinstance(style, str), f"parallel style type should be str, but got {type(style)}"

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


def validate_tp_mesh_for_nemotron_nas(model, tp_size):
    num_attention_heads = model.config.num_attention_heads
    assert num_attention_heads % tp_size == 0, "num_attention_heads in config does not match the TP size"

    assert len(model.config.block_configs) >= model.config.num_hidden_layers, (
        "num_hidden_layers in config does not match the number of block configs"
    )

    for i in range(model.config.num_hidden_layers):
        # Valid layer
        if model.config.block_configs[i].attention.replace_with_linear:
            print(f"By pass checking for linear layer in layer {i}")
            # TODO: Check if the linear layer could support TP.
        else:
            if model.config.block_configs[i].attention.n_heads_in_group is not None:
                num_key_value_heads = num_attention_heads // model.config.block_configs[i].attention.n_heads_in_group
                assert num_key_value_heads % tp_size == 0, (
                    f"layer {i}: num_key_value_heads in config does not match the TP size"
                )
            else:
                assert model.config.block_configs[i].attention.no_op == True


def validate_tp_mesh(model, tp_mesh):
    """
    Validate that attention heads and key value heads are divisible by TP size
    """
    if tp_mesh.size() == 1:
        return  # if tp_mesh.size() == 1, we don't need to validate

    model_cls = type(model)

    # There are cases like DeciLMForCausalLM is defined in transformers_modules
    # which hardly has predefined path to import. Guard access to config/architectures.
    model_arch = None
    if hasattr(model, "config") and hasattr(model.config, "architectures") and model.config.architectures:
        try:
            model_arch = model.config.architectures[0]
        except Exception:
            model_arch = None

    if model_cls in [
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    ]:
        # VL models have the language model at model.language_model
        num_attention_heads = model.language_model.config.num_attention_heads
        num_key_value_heads = model.language_model.config.num_key_value_heads

    elif model_cls == SmolVLMForConditionalGeneration:
        num_attention_heads = model.model.text_model.config.num_attention_heads
        num_key_value_heads = model.model.text_model.config.num_key_value_heads

    elif model_cls in [
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextVideoForConditionalGeneration,
        LlavaOnevisionForConditionalGeneration,
    ]:
        num_attention_heads = model.language_model.config.num_attention_heads
        num_key_value_heads = model.language_model.config.num_key_value_heads

    elif model_cls == Mistral3ForConditionalGeneration:
        num_attention_heads = model.model.language_model.config.num_attention_heads
        num_key_value_heads = model.model.language_model.config.num_key_value_heads

    elif model_cls == Llama4ForConditionalGeneration:
        num_attention_heads = model.language_model.model.config.num_attention_heads
        num_key_value_heads = model.language_model.model.config.num_key_value_heads

    elif model_cls == Gemma3ForConditionalGeneration:
        num_attention_heads = model.config.text_config.num_attention_heads
        num_key_value_heads = model.config.text_config.num_key_value_heads
    elif model_arch == "DeciLMForCausalLM" and getattr(model.config, "model_type", None) == "nemotron-nas":
        validate_tp_mesh_for_nemotron_nas(model, tp_mesh.size())

        # SKip following code and return.
        return
    elif hasattr(model, "config"):
        num_attention_heads = getattr(model.config, "num_attention_heads", 0)
        num_key_value_heads = getattr(model.config, "num_key_value_heads", 0)
    else:
        num_attention_heads = 0
        num_key_value_heads = 0

    # TP sharding with enhanced plan generation
    # Validate that attention heads are divisible by TP size
    assert num_key_value_heads % tp_mesh.size() == 0, (
        f"num_key_value_heads ({num_key_value_heads}) must be divisible by TP size ({tp_mesh.size()})"
    )
    assert num_attention_heads % tp_mesh.size() == 0, (
        f"num_attention_heads ({num_attention_heads}) must be divisible by TP size ({tp_mesh.size()})"
    )


def _find_largest_module_list(model: nn.Module) -> Optional[nn.ModuleList]:
    """
    Heuristic function to find the largest nn.ModuleList in a model.

    This function recursively traverses the model to find all nn.ModuleList instances
    and returns the one with the most modules. This is useful as a fallback when
    the model architecture is unknown, since transformer layers are typically
    organized in ModuleLists.

    Args:
        model (nn.Module): The model to search through.

    Returns:
        Optional[nn.ModuleList]: The largest ModuleList found, or None if no ModuleList exists.
    """
    largest_module_list = None
    largest_size = 0

    def _recursive_search(module: nn.Module, path: str = ""):
        nonlocal largest_module_list, largest_size

        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name

            if isinstance(child, nn.ModuleList):
                current_size = len(child)
                if current_size > largest_size:
                    largest_size = current_size
                    largest_module_list = child
                    logger.debug(f"Found ModuleList at {current_path} with {current_size} modules")

            # Continue recursive search
            _recursive_search(child, current_path)

    _recursive_search(model)

    if largest_module_list is not None:
        logger.info(f"Largest ModuleList found with {largest_size} modules")
    else:
        logger.warning("No ModuleList found in the model")

    return largest_module_list


def _extract_model_layers(model: nn.Module) -> List[nn.Module]:
    """
    Extract layers from different model architectures for parallelization.

    This function handles various model types including vision-language models,
    causal language models, and multimodal models. It collects both language
    model layers and vision model layers where applicable.

    Args:
        model (nn.Module): The model to extract layers from.

    Returns:
        List[nn.Module]: A list of all layers that should be parallelized.
    """

    def _reduce_attrs(model, fqns: List[str]) -> List[nn.Module]:
        if isinstance(fqns, str):
            fqns = [fqns]
        ans = []
        for fqn in fqns:
            parts = fqn.split(".")
            ans.append(reduce(getattr, parts, model))
        return ans

    VLM_MODEL_CLS_TO_LAYERS = {
        Gemma3ForConditionalGeneration: ["language_model.layers", "vision_tower.vision_model.encoder.layers"],
        Qwen2_5_VLForConditionalGeneration: ["language_model.layers", "visual.blocks"],
        Qwen2VLForConditionalGeneration: ["language_model.layers", "visual.blocks"],
        # Note: `model.` is not a mistake here, it's the full fqn
        SmolVLMForConditionalGeneration: ["model.text_model.layers", "model.vision_model.encoder.layers"],
        LlavaForConditionalGeneration: ["model.language_model.layers", "vision_tower.vision_model.encoder.layers"],
        LlavaNextForConditionalGeneration: ["model.language_model.layers", "vision_tower.vision_model.encoder.layers"],
        LlavaNextVideoForConditionalGeneration: [
            "model.language_model.layers",
            "vision_tower.vision_model.encoder.layers",
        ],
        LlavaOnevisionForConditionalGeneration: [
            "model.language_model.layers",
            "vision_tower.vision_model.encoder.layers",
        ],
        Mistral3ForConditionalGeneration: ["model.language_model.layers", "model.vision_tower.transformer.layers"],
        Llama4ForConditionalGeneration: ["language_model.model.layers", "vision_model.model.layers"],
    }
    LLM_MODEL_CLS_TO_LAYERS = {
        "NemotronHForCausalLM": ["backbone.layers"],
        GPT2LMHeadModel: ["transformer.h"],
    }

    MODEL_CLS_TO_LAYERS = VLM_MODEL_CLS_TO_LAYERS | LLM_MODEL_CLS_TO_LAYERS

    model_cls = type(model)
    layers: List[nn.Module] = []
    if model_cls in MODEL_CLS_TO_LAYERS:
        layers.extend(_reduce_attrs(model, MODEL_CLS_TO_LAYERS[model_cls]))
    elif model_cls.__name__ in MODEL_CLS_TO_LAYERS:
        layers.extend(_reduce_attrs(model, MODEL_CLS_TO_LAYERS[model_cls.__name__]))
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Default case for all other models (assumed to be a causal LM)
        if isinstance(model.model.layers, nn.ModuleDict):
            layers.extend(model.model.layers.values())
        else:
            layers.extend(model.model.layers)
    elif hasattr(model, "layers"):
        layers.extend(model.layers)
    else:
        # Use heuristic to find the largest ModuleList in the model
        logger.warning(f"Unknown model type: {model_cls}. Using heuristic to find transformer layers.")
        largest_module_list = _find_largest_module_list(model)
        if largest_module_list is None:
            # If no ModuleList found, still raise an exception
            print(model)
            raise ValueError(f"Unknown model type: {model_cls} and no ModuleList found in model structure")

        layers.extend(largest_module_list)
        logger.info(f"Successfully extracted {len(largest_module_list)} layers using heuristic")

    assert all(isinstance(m, nn.Module) for m in layers), "layers shoudl be nn.Module instances"
    return layers


def _get_parallel_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
    tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
    use_hf_tp_plan: bool = False,
) -> Dict[str, ParallelStyle]:
    """
    Select the tensor-parallel plan for the given model.

    Priority order:
    1) If ``tp_shard_plan`` is provided as a dict or import path (to a dict/function), use it.
    2) If ``use_hf_tp_plan`` is True, use the HF plan directly (asserts when sequence_parallel=True).
    3) If the model type exists in ``PARALLELIZE_FUNCTIONS``, use its optimised plan; on failure, try HF plan
    4) Otherwise, use the default base plan.
    """

    # Generate or use tensor parallel plan
    model_parallel_plan = None
    model_cls = type(model)

    # 1. Use custom parallel plan if provided
    if isinstance(tp_shard_plan, dict):
        model_parallel_plan = tp_shard_plan
        logger.info(f"Using parallel plan (dictionary). {tp_shard_plan}")
    elif tp_shard_plan is not None:
        try:
            plan_obj = import_class_from_path(tp_shard_plan)
            if isinstance(plan_obj, FunctionType):
                model_parallel_plan = plan_obj()
            else:
                model_parallel_plan = plan_obj
            assert isinstance(model_parallel_plan, dict), (
                f"Parallel plan must be a dictionary, got {type(model_parallel_plan)}"
            )
            logger.info(f"Using provided parallel plan (from path). {tp_shard_plan}")
        except Exception as e:
            raise ValueError(
                f"Custom parallel plan '{tp_shard_plan}' is not valid. "
                f"Please ensure it is one of the following:\n"
                "1. A dictionary mapping module names to parallel styles\n"
                "2. A path to a dictionary\n"
                "3. A path to a function that returns a dictionary\n"
                f"Error: {e}"
            )

    # 2. Prefer HF TP plan explicitly if requested
    elif use_hf_tp_plan:
        assert not sequence_parallel, "sequence_parallel is not supported in HF tp plan."
        model_parallel_plan = get_hf_tp_shard_plan(model)

    # 3. Use optimized parallel plan based on model type
    elif model_cls in PARALLELIZE_FUNCTIONS:
        try:
            func = PARALLELIZE_FUNCTIONS[model_cls]
            model_parallel_plan = func(model, sequence_parallel)
            logger.info("Using optimized parallel plan.")
        except Exception as e:
            logger.info(f"Optimized parallel plan is not available: {e}. Falling back to the HF tp plan.")
            assert not sequence_parallel, "sequence_parallel is not supported in HF tp plan."
            model_parallel_plan = get_hf_tp_shard_plan(model)

    # 4. Otherwise, use the default base plan.
    else:
        base_model_tp_plan = {
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # Combined QKV projection
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),  # Fused gate and up projection
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
        }
        if sequence_parallel:
            base_model_sp_plan = {
                "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                "model.norm": SequenceParallel(),
                "model.layers.*.input_layernorm": SequenceParallel(),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                "model.layers.*.post_attention_layernorm": SequenceParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
            }
            base_model_tp_plan.update(base_model_sp_plan)
        model_parallel_plan = base_model_tp_plan
        logger.info("Using default base TP plan. Compatible with huggingface llama3-style models.")

    return model_parallel_plan


# Taken and modified from torchtitan
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
def fsdp2_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    offload_policy: Optional[OffloadPolicy] = None,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
    tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
    dp_replicate_mesh_name: str = "dp_replicate",
    dp_shard_cp_mesh_name: str = "dp_shard_cp",
    tp_mesh_name: str = "tp",
):
    """
    Apply parallelisms and activation checkpointing to the model.

    Enhanced version that uses a strategy pattern for different model parallelization approaches:
    - Automatic strategy selection based on model type
    - Polymorphic parallelization strategies for different model families
    - Custom parallel plan support (dict or string path)
    - Sequence parallel support
    - Activation checkpointing for linear layers
    - Model validation (attention heads divisible by TP size)
    - Better fallback logic

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh for distributed training.
        mp_policy (Optional[MixedPrecisionPolicy]): Mixed precision policy for model parallelism.
        offload_policy (Optional[OffloadPolicy]): The offload policy for FSDP.
        sequence_parallel (bool): Whether to use sequence parallelism. Defaults to False.
        activation_checkpointing (bool): Whether to use activation checkpointing. Defaults to False.
        tp_shard_plan (Optional[Union[Dict[str, ParallelStyle], str]]):
            Custom tensor parallel plan for the model. Can be:
            - A dictionary mapping module names to parallel styles
            - A string path to a dictionary or function that returns a dictionary
            If provided, this takes precedence over automatic plan generation.
        dp_replicate_mesh_name (str): Key name for the data parallel replicate mesh in device_mesh.
            Used when data parallel replicate is enabled. Defaults to "dp_replicate".
        dp_shard_cp_mesh_name (str): Key name for the data parallel shard + context parallel mesh in device_mesh.
            Used when data parallel shard is enabled. Defaults to "dp_shard_cp".
        tp_mesh_name (str): Key name for the tensor parallel mesh in device_mesh.
            Defaults to "tp".

    Returns:
        The parallelized model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # Get the appropriate parallelization strategy for this model
    strategy = get_parallelization_strategy(model)

    # Delegate to the strategy
    return strategy.parallelize(
        model=model,
        device_mesh=device_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        sequence_parallel=sequence_parallel,
        activation_checkpointing=activation_checkpointing,
        tp_shard_plan=tp_shard_plan,
        dp_replicate_mesh_name=dp_replicate_mesh_name,
        dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
        tp_mesh_name=tp_mesh_name,
    )


def megatron_fsdp_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    optimizer=None,
    megatron_fsdp_unit_modules: Optional[List[str]] = None,
    tp_shard_plan: Optional[Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]] = None,
    zero_dp_strategy: int = 3,
    init_fsdp_with_meta_device: bool = False,
    grad_reduce_in_fp32: bool = False,
    preserve_fp32_weights: bool = False,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    check_for_nan_in_grad: bool = True,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
    dp_shard_dim: str = "dp",
    tp_dim: str = "tp",
):
    """
    Apply tensor/data parallelism (MegatronFSDP) and optional activation-checkpointing to the model.

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh describing the physical devices
            used for distributed training.
        megatron_fsdp_unit_modules (Optional[List[str]]): Names of sub-modules that should
            become individual MegatronFSDP units. If None, the full model is wrapped as
            a single unit.
        tp_shard_plan (Optional[Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]]):
            A tensor-parallel sharding plan.
            Keys are module names; values specify the parallel style to apply
            (e.g., RowwiseParallel, ColwiseParallel, SequenceParallel).
        zero_dp_strategy (int): The zero-DP strategy to use.
        init_fsdp_with_meta_device (bool): If True, construct the model on a
            meta device first and materialize weights lazily to reduce memory
            fragmentation.
        grad_reduce_in_fp32 (bool): Reduce gradients in FP32 irrespective of the
            parameter precision to improve numerical stability.
        preserve_fp32_weights (bool): Keep a master FP32 copy of weights when
            training in reduced precision (e.g., FP16/BF16).
        overlap_grad_reduce (bool): If True, overlap gradient reduction with
            backward computation.
        overlap_param_gather (bool): If True, overlap parameter gathering with
            forward computation.
        check_for_nan_in_grad (bool): Whether to check gradients for NaNs/Infs
            before applying the optimizer step.
        average_in_collective (bool): Perform gradient averaging inside the
            collective operation instead of dividing afterward.
        disable_bucketing (bool): Disable gradient bucketing; gradients are
            reduced immediately as they are produced.
        calculate_per_token_loss (bool): Compute loss normalized by the number of
            tokens instead of the number of sequences.
        keep_fp8_transpose_cache (bool): Retain the FP8
            transpose cache when using a custom MegatronFSDP wrapper.
        nccl_ub (bool): Enable NCCL user-buffer API (experimental) for reduced
            latency on some networks.
        fsdp_double_buffer (bool): Enable double buffering of parameters to
            overlap communication and computation in MegatronFSDP.
        dp_shard_dim (str): Key name for the data parallel mesh in device_mesh.
            Defaults to "dp".
        tp_dim (str): Key name for the tensor parallel mesh in device_mesh.
            Defaults to "tp".

    NOTE: The passed-in model should preferably reside on the meta device.
    Otherwise, ensure the model fits into available GPU or CPU memory.

    NOTE: The user must ensure that the provided tp_shard_plan is compatible
    with the model architecture.
    """
    assert HAVE_MEGATRON_FSDP, (
        "MegatronFSDP is not installed, please visit \
        https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src for \
        more information"
    )

    # DP_CP ranks are sharded by FSDP.
    dp_mesh = device_mesh[dp_shard_dim]
    tp_mesh = device_mesh[tp_dim]

    if dp_mesh.size() > 1:
        # TODO(boxiangw): remove this once HSDP is supported.
        assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"

    # TP sharding.
    if tp_mesh.size() > 1:
        parallelize_module(model, tp_mesh, tp_shard_plan)

    # Import MegatronFSDP unit modules specified by the user.
    megatron_fsdp_unit_modules = import_classes_from_paths(megatron_fsdp_unit_modules)

    # Wrap model with MegatronFSDP.
    model, optimizer = megatron_fsdp_fully_shard(
        module=model,
        optimizer=optimizer,
        fsdp_unit_modules=megatron_fsdp_unit_modules,
        device_mesh=device_mesh,
        dp_shard_dim=dp_shard_dim,
        tp_dim=tp_dim,
        zero_dp_strategy=zero_dp_strategy,
        init_model_with_meta_device=init_fsdp_with_meta_device,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        preserve_fp32_weights=preserve_fp32_weights,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        sync_grads_each_step=False,  # For better performance, avoid sync every step
        check_for_nan_in_grad=check_for_nan_in_grad,
        average_in_collective=average_in_collective,
        disable_bucketing=disable_bucketing,
        calculate_per_token_loss=calculate_per_token_loss,
        keep_fp8_transpose_cache=keep_fp8_transpose_cache,
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer,
    )

    return model, optimizer


@contextmanager
def unshard_fsdp2_model(model: nn.Module) -> Generator[None, None, None]:
    """Explicitly unshard and then reshard the FSDP2 modules. Useful for logprob inference."""
    try:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
        yield
    finally:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

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


from typing import Any, Optional

import torch
from torch import nn
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase


class LoRARequestWithCfgAndWeights(LoRARequest):
    lora_cfg: Optional[dict] = None
    lora_weights: Optional[dict[str, Any]] = None


def patched_load_adapter(self, lora_request: LoRARequestWithCfgAndWeights):
    try:
        supported_lora_modules = self._adapter_manager.supported_lora_modules
        packed_modules_mapping = self._adapter_manager.packed_modules_mapping
        expected_lora_lst: list[str] = []
        for module in supported_lora_modules:
            if module in packed_modules_mapping:
                expected_lora_lst.extend(packed_modules_mapping[module])
            else:
                expected_lora_lst.append(module)
            if module == "experts":
                expected_lora_lst.append(module)
        expected_lora_modules = set(expected_lora_lst)
        lora_weights = None

        if isinstance(lora_request, LoRARequestWithCfgAndWeights):
            lora_cfg = lora_request.lora_cfg
            lora_weights = lora_request.lora_weights
            peft_helper = PEFTHelper.from_dict(lora_cfg)
        else:
            lora_path = get_adapter_absolute_path(lora_request.lora_path)

            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                self.max_position_embeddings,
                lora_request.tensorizer_config_dict,
            )

        # Validates the LoRA configuration against requirements before
        # loading weights, throwing an exception if validation fails.
        peft_helper.validate_legal(self.lora_config)

        # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
        # to ensure correct loading of lora weights.
        model = self._adapter_manager.model
        hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
        if isinstance(lora_request, LoRARequestWithCfgAndWeights):
            lora = self._lora_model_cls.from_lora_tensors(
                lora_model_id=lora_request.lora_int_id,
                tensors=lora_weights,
                peft_helper=peft_helper,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                embeddings=None,
                target_embedding_padding=self.vocab_size
                + self.lora_config.lora_extra_vocab_size,
                embedding_modules=self.embedding_modules,
                embedding_padding_modules=self.embedding_padding_modules,
                weights_mapper=hf_to_vllm_mapper,
            )
        else:
            lora = self._lora_model_cls.from_local_checkpoint(
                lora_path,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                target_embedding_padding=self.vocab_size
                + self.lora_config.lora_extra_vocab_size,
                embedding_modules=self.embedding_modules,
                embedding_padding_modules=self.embedding_padding_modules,
                weights_mapper=hf_to_vllm_mapper,
            )

    except FileNotFoundError as e:
        # FileNotFoundError should be raised if both
        # - No adapter found to download from huggingface (or in
        #       offline mode)
        # - No local adapter files found at `lora_request.lora_path`
        # For NotFoundError
        raise ValueError(
            f"Loading lora {lora_request.lora_name} failed: No adapter "
            f"found for {lora_request.lora_path}"
        ) from e
    except Exception as e:
        # For BadRequestError
        raise e

    if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
        raise ValueError(
            f"LoRA added vocab size {lora.extra_vocab_size} is greater than lora_extra_vocab_size "
            f"{self.lora_config.lora_extra_vocab_size}."
        )
    return lora


def patched_get_supported_lora_modules(model: nn.Module) -> list[str]:
    """Skip lm_head modules in the supported_lora_modules.

    In vLLM, all linear layers support LoRA. But in Automodel, lm_head not support LoRA.
    Refer to https://github.com/NVIDIA-NeMo/Automodel/blob/50253d14c2aefa2206036022b4ccce9f3476ba4d/nemo_automodel/components/_peft/module_matcher.py#L99 for more details.
    """
    supported_lora_modules: set[str] = set()
    for name, module in model.named_modules():
        # get the embedding modules if the module's embedding_modules
        # is not empty.
        embedding_modules = getattr(module, "embedding_modules", None)
        if embedding_modules is not None:
            for name in embedding_modules:
                if "lm_head" in name:
                    continue
                supported_lora_modules.add(name)

        # get all the linear subfixes.
        if isinstance(module, (LinearBase,)):
            supported_lora_modules.add(name.split(".")[-1])

        if isinstance(module, (FusedMoE,)):
            supported_lora_modules.add(name.split(".")[-1])

    return list(supported_lora_modules)


def apply_lora_patches():
    # patch the get_supported_lora_modules function
    import vllm.lora.utils as lora_utils

    setattr(
        lora_utils, "get_supported_lora_modules", patched_get_supported_lora_modules
    )

    # patch the get_supported_lora_modules function in lora_models
    import vllm.lora.models as lora_models

    setattr(
        lora_models, "get_supported_lora_modules", patched_get_supported_lora_modules
    )

    # patch the load_adapter function in LRUCacheWorkerLoRAManager
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

    setattr(LRUCacheWorkerLoRAManager, "_load_adapter", patched_load_adapter)


def apply_weight_name_mapping(
    weights: list[tuple[str, torch.Tensor]],
    supported_modules: list[str],
    packed_modules_mapping: dict[str, list[str]],
) -> list[tuple[str, torch.Tensor]]:
    """Apply weight name mapping if LoRA is enabled."""

    def map_param_name(param_name: str) -> str:
        # Vllm add logits_processor to lm_head weight(https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/lora/models.py#L506), we skip mapping for lm_head weight
        if "lm_head" in param_name:
            return param_name
        parts = param_name.split(".")
        if len(parts) < 2:
            return param_name
        base_name = ".".join(parts[:-2])  # prefix
        module_name = parts[-2]  # e.g. q_proj/k_proj/v_proj/gate_proj/up_proj/...
        field_name = parts[-1]  # weight/bias
        resolved_module_name = module_name
        for packed_name, member_names in packed_modules_mapping.items():
            if module_name in member_names:
                resolved_module_name = packed_name
                break
        # use resolved_module_name for checking, but return the original module_name
        if resolved_module_name in supported_modules:
            if base_name != "":
                return f"{base_name}.{module_name}.base_layer.{field_name}"
            else:
                return f"{module_name}.base_layer.{field_name}"
        return param_name

    new_weights = []
    for name, w in weights:
        new_name = map_param_name(name)
        new_weights.append((new_name, w))
    return new_weights

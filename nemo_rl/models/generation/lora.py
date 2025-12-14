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

from vllm.lora.request import LoRARequest


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

        from vllm.lora.peft_helper import PEFTHelper

        if isinstance(lora_request, LoRARequestWithCfgAndWeights):
            lora_cfg = lora_request.lora_cfg
            lora_weights = lora_request.lora_weights
            peft_helper = PEFTHelper.from_dict(lora_cfg)
        else:
            raise ValueError(f"Unsupported LoRA request type: {type(lora_request)}")

        # Validates the LoRA configuration against requirements before
        # loading weights, throwing an exception if validation fails.
        peft_helper.validate_legal(self.lora_config)

        # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
        # to ensure correct loading of lora weights.
        model = self._adapter_manager.model
        hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
        print(f"hf_to_vllm_mapper in lora.patched_load_adapter: {hf_to_vllm_mapper}")
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
            raise ValueError(f"Unsupported LoRA request type: {type(lora_request)}")

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


def apply_lora_patches():
    # func_path = "vllm.lora.worker_manager.LRUCacheWorkerLoRAManager.load_adapter"
    # patcher = patch(func_path, patched_load_adapter)
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

    setattr(LRUCacheWorkerLoRAManager, "_load_adapter", patched_load_adapter)


lora_int_id = 0


# Note: Not sure put it here or in nemo_rl/models/generation/vllm/utils.py
def get_vllm_lora_metadata() -> dict[str, Any]:
    global lora_int_id
    lora_int_id += 1  # Can be any unique id exclude 0
    lora_name = f"{lora_int_id}"
    lora_path = "dummy_lora_path"
    return {
        "lora_name": lora_name,
        "lora_int_id": lora_int_id,
        "lora_path": lora_path,
    }

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

import os
from contextlib import contextmanager
from typing import Any, Generator

import modelopt.torch.opt as mto
import ray
import torch
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.quantization.utils import (
    get_quantizer_state_dict,
    set_quantizer_state_dict,
)
from torch.distributed.checkpoint.state_dict import StateDictOptions

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
    DTensorPolicyWorkerV2Impl,
)
from nemo_rl.models.policy.workers.quantization.utils import (
    get_modelopt_checkpoint_dir,
    get_tokenizer,
    quantize_model,
)

_MODELOPT_STATE_FILE_NAME = "modelopt_state.pth"


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker_v2"))
class DTensorQuantPolicyWorkerV2(DTensorPolicyWorkerV2Impl):
    def __init__(self, *args, **kwargs):
        mto.enable_huggingface_checkpointing()
        # with self._set_model_state_dict_with_modelopt():
        super().__init__(*args, **kwargs)
        assert self.cfg["quant_cfg"] is not None, (
            "quant_cfg must be set to use quantized policy."
        )
        self._quantize()

    def _save_modelopt_state_with_weights(self, weights_path: str):
        modelopt_state = mto.modelopt_state(self.model)
        modelopt_state["modelopt_weights"] = get_quantizer_state_dict(self.model)
        modelopt_state_path = os.path.join(weights_path, _MODELOPT_STATE_FILE_NAME)
        # note we don't have PP support here.
        if torch.distributed.get_rank() == 0:
            torch.save(modelopt_state, modelopt_state_path)
        torch.distributed.barrier()

    def _load_modelopt_state_with_weights(self, modelopt_state_path: str = None):
        if modelopt_state_path is None:
            modelopt_state_path = os.path.join(
                self.cfg["model_name"], _MODELOPT_STATE_FILE_NAME
            )
        if not os.path.exists(modelopt_state_path):
            return
        modelopt_state = torch.load(modelopt_state_path)
        if not mto.ModeloptStateManager.is_converted(self.model):
            mto.restore_from_modelopt_state(self.model, modelopt_state)

        if "modelopt_weights" in modelopt_state:
            model_device = next(self.model.parameters()).device
            for _, module in self.model.named_modules():
                if isinstance(module, TensorQuantizer):
                    module.to(model_device)
            set_quantizer_state_dict(self.model, modelopt_state["modelopt_weights"])

        print("self.model after loading modelopt state: ", self.model)

        return self.model

    def set_model_state_dict(
        self, model_state_dict: dict[str, Any], options: StateDictOptions
    ) -> None:
        model_path = self.cfg["model_name"]
        print("self.model", self.model)
        super().set_model_state_dict(model_state_dict, options)
        self._load_modelopt_state_with_weights()

    def save_checkpoint(self, weights_path: str, *args, **kwargs):
        super().save_checkpoint(weights_path, *args, **kwargs)
        self._save_modelopt_state_with_weights(weights_path)

    @torch.no_grad()
    def _quantize(self):
        """Quantize the model if the model is not quantized yet."""
        if mto.ModeloptStateManager.is_converted(self.model):
            return
        dataset_name = self.cfg.get("quant_dataset_name", "cnn_dailymail")
        quant_cfg = self.cfg["quant_cfg"]
        model_path = self.cfg["model_name"]

        modelopt_checkpoint_dir = get_modelopt_checkpoint_dir()
        if os.path.exists(model_path):
            model_name = os.path.basename(model_path)
        else:
            model_name = model_path
        # The modelopt state is saved at modelopt_checkpoint_dir/model_name_quant_cfg_dataset_name/modelopt_state.pth
        modelopt_state_dir = os.path.join(
            modelopt_checkpoint_dir,
            f"{model_name.replace('/', '_')}_{quant_cfg}_{dataset_name.replace('/', '_')}",
        )
        if not os.path.exists(modelopt_state_dir):
            os.makedirs(modelopt_state_dir)
        modelopt_state_path = os.path.join(
            modelopt_state_dir, _MODELOPT_STATE_FILE_NAME
        )
        if os.path.exists(modelopt_state_path):
            return self._load_modelopt_state_with_weights(modelopt_state_path)

        tokenizer = get_tokenizer(model_path)
        quantize_model(
            self.model,
            quant_cfg,
            tokenizer,
            calib_size=self.cfg.get("quant_calib_size", 512),
            is_megatron=False,
            batch_size=self.cfg.get("quant_batch_size", 32),
            data=dataset_name,
        )

        self._save_modelopt_state_with_weights(modelopt_state_dir)

    @contextmanager
    def disable_quantization(self):
        """Context manager that temporarily disables quantization."""
        quantizers = []

        try:
            for _, module in self.model.named_modules():
                if isinstance(module, TensorQuantizer) and module.enabled:
                    quantizers.append(module)
                    module.disable()

            yield
        finally:
            for module in quantizers:
                module.enable()

    @contextmanager
    def use_reference_model(self) -> Generator[None, None, None]:
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        with self.disable_quantization(), super().use_reference_model():
            yield

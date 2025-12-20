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
import types
from contextlib import contextmanager
from typing import Generator

import ray
import torch.nn as nn
from megatron.bridge.models.gpt_provider import quantization_layer_spec
from megatron.bridge.training.post_training.checkpointing import has_modelopt_state
from megatron.core.utils import unwrap_model
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

import nemo_rl.models.policy.workers.megatron_policy_worker as megatron_policy_worker
from nemo_rl.models.policy.utils import (
    get_megatron_checkpoint_dir,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from nemo_rl.models.policy.workers.quantization.utils import (
    get_tokenizer,
    quantize_model,
)


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_quant_policy_worker")
)  # pragma: no cover
class MegatronQuantPolicyWorker(MegatronPolicyWorkerImpl):
    def __init__(self, config, *args, **kwargs):
        # config["megatron_cfg"]["transformer_layer_spec"] = quantization_layer_spec
        hf_model_name = config["model_name"]
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
        iter_0_path = os.path.join(pretrained_path, "iter_0000000")
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            iter_0_path
        )
        pre_quantized_model_path = os.environ.get(
            "NRL_PRE_QUANTIZED_MEGATRON_MODEL_PATH"
        )
        if pre_quantized_model_path is not None and not pt_checkpoint_exists:
            # create a symlink to the pre-quantized model
            absolute_pre_quantized_model_path = os.path.abspath(
                pre_quantized_model_path
            )
            os.symlink(absolute_pre_quantized_model_path, iter_0_path)

        kwargs["import_model_post_wrap_hook"] = self._quantize
        kwargs["import_model_transformer_layer_spec"] = quantization_layer_spec
        self._patch_setup_megatron_model()
        super().__init__(config, *args, **kwargs)

    def _quantize(self, model):
        """Quantize the model if the model is not quantized yet."""
        quant_cfg = self.cfg["quant_cfg"]
        quant_dataset_name = self.cfg.get("quant_dataset_name", "cnn_dailymail")
        quant_calib_size = self.cfg.get("quant_calib_size", 512)
        quant_batch_size = self.cfg.get("quant_batch_size", 1)
        # if quant_batch_size > 1:
        #     warnings.warn("Quantization batch size > 1 for megatron model is not supported yet. Setting to 1.")
        #     quant_batch_size = 1
        # quant_force_all_expert_routing = self.cfg.get("quant_force_all_expert_routing", True)

        unwrapped_model = unwrap_model(model)[0]

        tokenizer = get_tokenizer(self.cfg["model_name"])
        quantize_model(
            model=unwrapped_model,
            quant_cfg=quant_cfg,
            tokenizer=tokenizer,
            calib_size=quant_calib_size,
            is_megatron=True,
            batch_size=quant_batch_size,
            data=quant_dataset_name,
            force_all_expert_routing=True,
        )
        return model

    @staticmethod
    def _patch_setup_megatron_model():
        if hasattr(megatron_policy_worker, "_original_setup_megatron_model"):
            return
        megatron_policy_worker._original_setup_megatron_model = (
            megatron_policy_worker.setup_megatron_model
        )

        def _setup_megatron_model(policy_cfg, cfg, *args, **kwargs):
            print("Calling patched setup_megatron_model")
            model_path = cfg.checkpoint.pretrained_checkpoint or cfg.checkpoint.load
            print("model_path:", model_path)
            if os.path.exists(os.path.join(model_path, "iter_0000000")):
                model_path = os.path.join(model_path, "iter_0000000")
            if has_modelopt_state(model_path):
                print("setting restore_modelopt_state to True")
                cfg.model.restore_modelopt_state = True

            def _modelopt_pre_wrap_hook(model):
                from megatron.bridge.training.post_training.checkpointing import (
                    has_modelopt_state,
                    load_modelopt_state,
                )

                # Check which checkpoint path has modelopt state
                model_path = cfg.checkpoint.pretrained_checkpoint or cfg.checkpoint.load
                if os.path.exists(os.path.join(model_path, "iter_0000000")):
                    model_path = os.path.join(model_path, "iter_0000000")
                if has_modelopt_state(model_path):
                    checkpoint_path = model_path
                else:
                    raise RuntimeError(f"No modelopt_state found in {model_path}")

                load_modelopt_state(model, checkpoint_path)
                return model

            kwargs["pre_wrap_hook"] = [_modelopt_pre_wrap_hook]
            return megatron_policy_worker._original_setup_megatron_model(
                policy_cfg, cfg, *args, **kwargs
            )

        megatron_policy_worker.setup_megatron_model = _setup_megatron_model

    def _patch_automapping(self):
        # Cache proxy classes by original class name to avoid repeated class creation
        _proxy_class_cache = {}

        def _patched_detect_parallelism_type(self, module: nn.Module) -> str:
            if isinstance(module, DynamicModule):
                # Get or create a proxy class with the correct __name__
                # This avoids modifying the actual class object (thread-safe)
                original_name = module.original_cls.__name__
                # print()
                if original_name not in _proxy_class_cache:
                    _proxy_class_cache[original_name] = type(original_name, (), {})

                # Create proxy instance and copy all attributes
                proxy = _proxy_class_cache[original_name]()
                proxy.__dict__.update(module.__dict__)
                print(
                    "module.original_cls.__name__:",
                    original_name,
                    " proxy name:",
                    type(proxy).__name__,
                )

                return self._original_detect_parallelism_type(proxy)
            print("module in patched_detect_parallelism_type:", module)
            return self._original_detect_parallelism_type(module)

        for task in self.refit_conversion_tasks:
            if hasattr(task.mapping, "_detect_parallelism_type") and not hasattr(
                task.mapping, "_original_detect_parallelism_type"
            ):
                task.mapping._original_detect_parallelism_type = (
                    task.mapping._detect_parallelism_type
                )
                task.mapping._detect_parallelism_type = types.MethodType(
                    _patched_detect_parallelism_type, task.mapping
                )

    def _calculate_refit_param_info(self):
        param_info = super()._calculate_refit_param_info()
        # self._patch_automapping()
        return param_info

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

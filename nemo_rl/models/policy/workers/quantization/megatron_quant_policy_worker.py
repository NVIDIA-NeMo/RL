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
from typing import Generator

import ray
import torch
from megatron.bridge.training.post_training.checkpointing import has_modelopt_state
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.core import parallel_state
from megatron.core.utils import unwrap_model
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

import nemo_rl.models.policy.workers.megatron_policy_worker as megatron_policy_worker
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy.utils import (
    get_megatron_checkpoint_dir,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
    destroy_parallel_state,
)
from nemo_rl.models.policy.workers.quantization.utils import (
    get_tokenizer,
    quantization_layer_spec,
    quantize_model,
)


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_quant_policy_worker")
)  # pragma: no cover
class MegatronQuantPolicyWorker(MegatronPolicyWorkerImpl):
    def __init__(self, config, *args, **kwargs):
        """Initialize the MegatronQuantPolicyWorker."""
        # Turn on bridge quantization mapping
        os.environ["ENABLE_BRIDGE_QUANT_MAPPING"] = "1"
        self._patch_setup_megatron_model()
        super().__init__(config, *args, **kwargs)

        # add quantizer states to reference state dict, so we don't need to modify use_reference_model logic
        if hasattr(self, "reference_state_dict"):
            for name, item in self.model.state_dict().items():
                if "_quantizer." in name:
                    self.reference_state_dict[name] = item.detach().to(
                        device="cpu", non_blocking=True, copy=True
                    )

    def _import_model_from_hf(self, hf_model_name: str):
        """Import a Hugging Face model into Megatron checkpoint format and save the Megatron checkpoint to the output path.

        This will quantize the model before saving. In cases like distillation, if the teacher model is same as the student
        model, we need to save an extra quantized checkpoint.

        Args:
            hf_model_name: Hugging Face model ID or local path (e.g., 'meta-llama/Llama-3.1-8B-Instruct').

        Returns:
            The path to the Megatron checkpoint.
        """
        # Check if the checkpoint already exists
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
        iter0_path = os.path.join(pretrained_path, "iter_0000000")
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            iter0_path
        )
        # In cases like distillation, if the teacher model is same as the student model, we need to save an extra quantized checkpoint
        if pt_checkpoint_exists and not has_modelopt_state(iter0_path):
            pretrained_path += "_quantized"
            iter0_path = os.path.join(pretrained_path, "iter_0000000")
            pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
                iter0_path
            )

        pre_quantized_model_path = os.environ.get(
            "NRL_PRE_QUANTIZED_MEGATRON_MODEL_PATH"
        )
        if pre_quantized_model_path is not None and not pt_checkpoint_exists:
            # create a symlink to the pre-quantized model
            absolute_pre_quantized_model_path = os.path.abspath(
                pre_quantized_model_path
            )
            os.symlink(absolute_pre_quantized_model_path, iter0_path)
            return pretrained_path

        # Ensure clean slate before import
        destroy_parallel_state()

        # Set for rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()
        # Need to initialize the process group before calling into Megatron-Bridge, otherwise Megatron-Bridge will try to set an incorrect device
        torch.distributed.init_process_group("nccl")
        if pt_checkpoint_exists:
            print(f"Checkpoint already exists at {pretrained_path}. Skipping import.")
        else:
            hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}
            import_model_from_hf_name(
                hf_model_name,
                pretrained_path,
                self.cfg["megatron_cfg"],
                model_post_wrap_hook=self._quantize,
                transformer_layer_spec=quantization_layer_spec,
                **hf_config_overrides,
            )

            if parallel_state.model_parallel_is_initialized():
                print("Reinitializing model parallel after loading model state.")
                parallel_state.destroy_model_parallel()
        return pretrained_path

    def _quantize(self, model):
        """Quantize the model if the model is not quantized yet."""
        quant_cfg = self.cfg["quant_cfg"]
        quant_dataset_name = self.cfg.get("quant_dataset_name", "cnn_dailymail")
        quant_calib_size = self.cfg.get("quant_calib_size", 512)
        quant_batch_size = self.cfg.get("quant_batch_size", 1)
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

    def _patch_setup_megatron_model(self):
        """Patch the setup_megatron_model function to restore the modelopt state."""
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

    @contextmanager
    def disable_quantization(self):
        """Context manager that temporarily disables quantization."""
        quantizers = []
        try:
            for _, module in self.model.named_modules():
                if isinstance(module, TensorQuantizer) and module.is_enabled:
                    quantizers.append(module)
                    module.disable()
            yield
        finally:
            for module in quantizers:
                module.enable()

    @contextmanager
    def use_reference_model(self) -> Generator[None, None, None]:
        """Context manager that temporarily swaps the reference model and active model."""
        with (
            self.disable_quantization(),
            self.without_model_config(),
            super().use_reference_model(),
        ):
            yield

    @contextmanager
    def without_model_config(self):
        configs = {}
        try:
            for name, module in self.model.named_modules():
                if isinstance(module, TensorQuantizer):
                    if hasattr(module, "config"):
                        configs[name] = module.config
                        delattr(module, "config")
            yield
        finally:
            for name, config in configs.items():
                setattr(self.model.get_submodule(name), "config", config)

    def save_checkpoint(self, *args, **kwargs):
        """Save the checkpoint."""
        # temp patch, a config is added to Quantizer which will break saving.
        with self.without_model_config():
            return super().save_checkpoint(*args, **kwargs)

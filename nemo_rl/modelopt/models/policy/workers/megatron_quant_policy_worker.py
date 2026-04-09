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
from pathlib import Path
from typing import Generator

import modelopt.torch.quantization as mtq
import ray
import torch
from megatron.bridge.training.post_training.checkpointing import (
    has_modelopt_state,
    load_modelopt_state,
)
from megatron.core import parallel_state
from megatron.core.utils import unwrap_model
from modelopt.torch.quantization.nn.modules.quant_module import QuantModule
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

import nemo_rl.models.megatron.setup as megatron_setup
import nemo_rl.models.policy.workers.megatron_policy_worker as megatron_policy_worker
from nemo_rl.modelopt.models.policy.workers.utils import (
    get_tokenizer,
    quantization_layer_spec,
    quantize_model,
)
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_quant_policy_worker")
)  # pragma: no cover
class MegatronQuantPolicyWorker(MegatronPolicyWorkerImpl):
    def __init__(self, config, *args, **kwargs):
        """Initialize the MegatronQuantPolicyWorker."""
        os.environ["ENABLE_BRIDGE_QUANT_MAPPING"] = "1"
        self._patch_validate_model_paths()
        self._patch_setup_model_and_optimizer()
        with self._patched_model_loading():
            super().__init__(config, *args, **kwargs)

        if hasattr(self, "reference_state_dict"):
            for name, item in self.model.state_dict().items():
                if "_quantizer." in name:
                    self.reference_state_dict[name] = item.detach().to(
                        device="cpu", non_blocking=True, copy=True
                    )
        if self.rank == 0:
            print(f"Quantized model: {self.model}")
            mtq.print_quant_summary(self.model)

    def _quantize(self, model):
        """Quantize the model if the model is not quantized yet."""
        quant_cfg = self.cfg["quant_cfg"]
        quant_calib_data = self.cfg["quant_calib_data"]
        quant_calib_size = self.cfg["quant_calib_size"]
        quant_batch_size = self.cfg["quant_batch_size"]
        quant_sequence_length = self.cfg["quant_sequence_length"]
        unwrapped_model = unwrap_model(model)[0]

        tokenizer = get_tokenizer(self.cfg["model_name"])
        quantize_model(
            model=unwrapped_model,
            quant_cfg=quant_cfg,
            tokenizer=tokenizer,
            calib_size=quant_calib_size,
            is_megatron=True,
            batch_size=quant_batch_size,
            data=quant_calib_data,
            max_sample_length=quant_sequence_length,
        )
        return model

    def _patch_validate_model_paths(self):
        """Patch validate_model_paths to handle quantized checkpoint paths.

        In cases like distillation where the teacher model is the same as the student model,
        we need to save an extra quantized checkpoint. This patch checks for modelopt state
        and redirects to a _quantized suffix path. It also handles pre-quantized model symlinks.
        """
        if hasattr(megatron_policy_worker, "_original_validate_model_paths"):
            return
        megatron_policy_worker._original_validate_model_paths = (
            megatron_policy_worker.validate_model_paths
        )

        def _validate_model_paths(config):
            hf_model_name, pretrained_path, pt_checkpoint_exists = (
                megatron_policy_worker._original_validate_model_paths(config)
            )

            iter0_path = os.path.join(pretrained_path, "iter_0000000")
            if pt_checkpoint_exists and not has_modelopt_state(iter0_path):
                pretrained_path += "_quantized"
                iter0_path = os.path.join(pretrained_path, "iter_0000000")
                pt_checkpoint_exists = os.path.exists(
                    pretrained_path
                ) and os.path.exists(iter0_path)

            pre_quantized_model_path = os.environ.get(
                "NRL_PRE_QUANTIZED_MEGATRON_MODEL_PATH"
            )
            if pre_quantized_model_path is not None and not pt_checkpoint_exists:
                print(f"Using pre-quantized model at: {pre_quantized_model_path}")
                absolute_path = Path(pre_quantized_model_path).resolve()
                os.makedirs(pretrained_path, exist_ok=True)
                os.symlink(
                    absolute_path.as_posix(),
                    iter0_path,
                    target_is_directory=True,
                )
                assert os.path.exists(iter0_path)
                pt_checkpoint_exists = True

            return hf_model_name, pretrained_path, pt_checkpoint_exists

        megatron_policy_worker.validate_model_paths = _validate_model_paths

    def _patch_setup_model_and_optimizer(self):
        """Patch setup_model_and_optimizer to restore modelopt state when loading quantized checkpoints."""
        if hasattr(megatron_policy_worker, "_original_setup_model_and_optimizer"):
            return
        megatron_policy_worker._original_setup_model_and_optimizer = (
            megatron_policy_worker.setup_model_and_optimizer
        )

        def _setup_model_and_optimizer(policy_cfg, megatron_cfg, *args, **kwargs):
            model_path = (
                megatron_cfg.checkpoint.pretrained_checkpoint
                or megatron_cfg.checkpoint.load
            )
            if os.path.exists(os.path.join(model_path, "iter_0000000")):
                model_path = os.path.join(model_path, "iter_0000000")
            if has_modelopt_state(model_path):
                print("setting restore_modelopt_state to True")
                megatron_cfg.model.restore_modelopt_state = True
                megatron_cfg.model.transformer_layer_spec = quantization_layer_spec

            return megatron_policy_worker._original_setup_model_and_optimizer(
                policy_cfg, megatron_cfg, *args, **kwargs
            )

        megatron_policy_worker.setup_model_and_optimizer = _setup_model_and_optimizer

    @contextmanager
    def _patched_model_loading(self):
        """Context manager that patches handle_model_import and load_checkpoint for quantization.

        Patches handle_model_import to pass quantization hooks during HF->Megatron conversion,
        and patches load_checkpoint to restore modelopt state before loading checkpoint weights.
        """
        original_handle_model_import = megatron_policy_worker.handle_model_import
        original_load_checkpoint = megatron_setup.load_checkpoint

        def _handle_model_import(
            config, hf_model_name, pretrained_path, pt_checkpoint_exists
        ):
            if pt_checkpoint_exists:
                print(
                    f"Checkpoint already exists at {pretrained_path}. Skipping import."
                )
            else:
                hf_config_overrides = config.get("hf_config_overrides", {}) or {}
                import_model_from_hf_name(
                    hf_model_name,
                    pretrained_path,
                    config["megatron_cfg"],
                    model_post_wrap_hook=self._quantize,
                    transformer_layer_spec=quantization_layer_spec,
                    **hf_config_overrides,
                )

                if parallel_state.model_parallel_is_initialized():
                    print("Reinitializing model parallel after loading model state.")
                    parallel_state.destroy_model_parallel()

        def _load_checkpoint(state, model, *args, **kwargs):
            cfg = state.cfg
            model_path = cfg.checkpoint.pretrained_checkpoint or cfg.checkpoint.load
            if os.path.exists(os.path.join(model_path, "iter_0000000")):
                model_path = os.path.join(model_path, "iter_0000000")
            if has_modelopt_state(model_path):
                unwrapped_model = unwrap_model(model)
                load_modelopt_state(unwrapped_model, model_path)
            return original_load_checkpoint(state, model, *args, **kwargs)

        try:
            megatron_policy_worker.handle_model_import = _handle_model_import
            megatron_setup.load_checkpoint = _load_checkpoint
            yield
        finally:
            megatron_policy_worker.handle_model_import = original_handle_model_import
            megatron_setup.load_checkpoint = original_load_checkpoint

    @contextmanager
    def hide_tensor_quantizers(self):
        """Context manager that temporarily hides TensorQuantizer modules from module iteration."""
        from megatron.core.distributed import DistributedDataParallel

        if not isinstance(self.model, DistributedDataParallel):
            yield
            return

        inner_module = self.model.module
        original_named_modules = inner_module.named_modules

        def filtered_named_modules(*args, **kwargs):
            for name, module in original_named_modules(*args, **kwargs):
                if not isinstance(module, TensorQuantizer):
                    yield name, module

        try:
            inner_module.named_modules = filtered_named_modules
            yield
        finally:
            inner_module.named_modules = original_named_modules

    def enable_forward_pre_hook(self):
        """Enable forward pre-hook, hiding TensorQuantizer modules."""
        with self.hide_tensor_quantizers():
            super().enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        """Disable forward pre-hook, hiding TensorQuantizer modules."""
        with self.hide_tensor_quantizers():
            super().disable_forward_pre_hook(param_sync=param_sync)

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
    def _hide_extra_state(self):
        """Patch model.state_dict() to exclude _extra_state keys.

        ModelOpt appends quantization calibration data (amax/scale) to TE's
        serialized extra state, making it larger than the non-quantized
        reference model's copy. These are calibration metadata, not learned
        weights, and can also be resized by TE during forward passes.
        Filtering them out lets the base class swap/restore skip them cleanly.
        """
        original_state_dict = self.model.state_dict

        def filtered_state_dict(*args, **kwargs):
            sd = original_state_dict(*args, **kwargs)
            return {k: v for k, v in sd.items() if not k.endswith("._extra_state")}

        try:
            self.model.state_dict = filtered_state_dict
            yield
        finally:
            self.model.state_dict = original_state_dict

    @contextmanager
    def use_reference_model(self) -> Generator[None, None, None]:
        """Context manager that temporarily swaps the reference model and active model."""
        with (
            self.disable_quantization(),
            self.without_model_config(),
            self._hide_extra_state(),
            super().use_reference_model(),
        ):
            yield

    @contextmanager
    def without_model_config(self):
        """Context manager that temporarily removes the model config from modules."""
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

    def get_quantizer_stats(self) -> dict:
        """Return summary statistics for all enabled TensorQuantizers.

        Useful for verifying that calibration ran and amax values are valid.
        """
        total = 0
        enabled = 0
        with_amax = 0
        positive_amax = 0
        for _, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                total += 1
                if module.is_enabled:
                    enabled += 1
                    if hasattr(module, "amax") and module.amax is not None:
                        with_amax += 1
                        if (module.amax > 0).all():
                            positive_amax += 1
        return {
            "total": total,
            "enabled": enabled,
            "with_amax": with_amax,
            "positive_amax": positive_amax,
        }

    def generate(self, **kwargs):
        """Quantized Megatron generation is not supported.

        ModelOpt unconditionally patches flash_decode_and_prefill on quantized
        attention modules, which breaks the Megatron generation path.
        """
        raise NotImplementedError(
            "MegatronQuantPolicyWorker does not support generate(). "
            "Use vLLM or SGLang as the generation backend instead."
        )

    def save_checkpoint(self, *args, **kwargs):
        """Save the checkpoint."""
        with self.without_model_config():
            return super().save_checkpoint(*args, **kwargs)

    @torch.no_grad()
    def prepare_refit_info(self):
        """Prepare refit info, excluding weight_quantizer amax.

        Weight quantizer amax is consumed on the Megatron side via pre-folding,
        so only input_quantizer amax needs to be transferred to vLLM.
        Uses the parent's iterator (without folding) since metadata is shape/dtype only.
        """
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        refit_param_info_hf = {}
        for name, tensor in super()._iter_params_with_optional_kv_scales():
            if "weight_quantizer" in name:
                continue
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)
        return refit_param_info_hf

    @staticmethod
    def _find_weight_quantizer(module, param_weight):
        """Find the enabled weight quantizer that corresponds to ``param_weight``.

        Uses ModelOpt's ``QuantModule.iter_weights_for_calibration`` to discover
        ``(weight, weight_quantizer)`` pairs, then matches by identity.
        This handles standard ``weight`` / ``weight_quantizer`` as well as
        custom names like ``gate_up_proj`` / ``gate_up_proj_weight_quantizer``.

        Returns the matching ``TensorQuantizer`` or ``None``.
        """
        if module is None or param_weight is None:
            return None
        if not isinstance(module, QuantModule):
            return None
        for weight, wq in module.iter_weights_for_calibration():
            if (
                param_weight is weight
                and isinstance(wq, TensorQuantizer)
                and wq.is_enabled
            ):
                return wq
        return None

    def _iter_params_with_optional_kv_scales(self, kv_scales=None):
        """Pre-fold weights on-the-fly via lazy proxy tasks.

        Wraps each conversion task so that reading task.param_weight returns
        weight_quantizer(weight) instead of the raw weight. The folded tensor
        is computed lazily when export_hf_weights accesses it, so only one
        extra weight-sized tensor exists at a time — O(1) extra memory.
        """

        class _FoldedTask:
            """Proxy that applies weight_quantizer(param_weight) on access."""

            def __init__(self, task, wq):
                self._task = task
                self._wq = wq

            @property
            def param_weight(self):
                w = self._task.param_weight
                if w is None:
                    return None
                return self._wq(w.float()).to(w.dtype)

            def __getattr__(self, name):
                return getattr(self._task, name)

        folded_tasks = []
        skipped_fold = []
        for task in self.refit_conversion_tasks:
            matched_wq = self._find_weight_quantizer(
                task.megatron_module, task.param_weight
            )
            if matched_wq is not None:
                folded_tasks.append(_FoldedTask(task, matched_wq))
            else:
                if (
                    task.param_weight is not None
                    and isinstance(task.megatron_module, QuantModule)
                    and any(
                        True
                        for _ in task.megatron_module.iter_weights_for_calibration()
                    )
                ):
                    skipped_fold.append(task.param_name)
                folded_tasks.append(task)

        if skipped_fold and self.rank == 0:
            print(
                f"[QuantFold] Skipped folding {len(skipped_fold)} non-GEMM params "
                f"that share a module with weight_quantizer: {skipped_fold[:5]}"
            )

        original_tasks = self.refit_conversion_tasks
        self.refit_conversion_tasks = folded_tasks
        try:
            for name, tensor in super()._iter_params_with_optional_kv_scales(kv_scales):
                if "weight_quantizer" in name:
                    continue
                yield name, tensor
        finally:
            self.refit_conversion_tasks = original_tasks

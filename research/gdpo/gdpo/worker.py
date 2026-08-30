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
"""Automodel policy-worker extension for GDPO."""

import warnings
from typing import Any

import ray
import torch
from torch.distributed.tensor import DTensor

from gdpo import (
    SdmcElboEstimator,
    accumulate_elbo_logprobs,
    denoise_config_from_generation,
    make_dllm_mask_seeds,
    masked_diffusion_config_from_policy,
    resolve_mask_id,
)
from gdpo.denoise import block_denoise, build_canvas, unpack_generations
from gdpo.train_gdpo import gdpo_forward_backward
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.train import (
    LogprobsPostProcessor,
    forward_with_post_processing_fn,
    prepare_model_forward,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
    DTensorPolicyWorkerV2Impl,
)


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker_v2")
)  # pragma: no cover
class DTensorGDPOPolicyWorker(DTensorPolicyWorkerV2Impl):
    """DTensor policy worker with SDMC scoring and block denoising."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.diffusion_cfg = masked_diffusion_config_from_policy(self.cfg)
        if self.diffusion_cfg is None:
            raise ValueError(
                "DTensorGDPOPolicyWorker requires policy.masked_diffusion.enabled=true"
            )
        generation_cfg = self.cfg["generation"]
        assert generation_cfg is not None
        self.denoise_cfg = denoise_config_from_generation(generation_cfg)
        self.elbo_estimator = SdmcElboEstimator(
            self.diffusion_cfg.likelihood,
            resolve_mask_id(self.diffusion_cfg, self.model_config),
        )

    def _make_loss_post_processor(self, **kwargs: Any):
        kwargs["precomputed_logprobs"] = True
        return super()._make_loss_post_processor(**kwargs)

    def _make_logprobs_post_processor(self, **kwargs: Any):
        kwargs["shift_targets"] = self.diffusion_cfg.shift_targets
        return super()._make_logprobs_post_processor(**kwargs)

    def _forward_backward(self, **kwargs: Any) -> list[tuple[Any, dict[str, Any]]]:
        sequence_dim = kwargs["sequence_dim"]
        return gdpo_forward_backward(
            data_iterator=kwargs["data_iterator"],
            post_processing_fn=kwargs["post_processing_fn"],
            elbo_scorer=self._gdpo_train_elbo_scorer(sequence_dim),
            forward_only=kwargs["forward_only"],
            global_valid_seqs=kwargs["global_valid_seqs"],
            global_valid_toks=kwargs["global_valid_toks"],
            sequence_dim=sequence_dim,
            dp_size=kwargs["dp_size"],
            cp_size=kwargs["cp_size"],
            num_global_batches=kwargs["num_global_batches"],
            train_context_fn=kwargs["train_context_fn"],
            num_valid_microbatches=kwargs["num_valid_microbatches"],
            on_microbatch_start=kwargs["on_microbatch_start"],
        )

    def _gdpo_elbo_logprobs(
        self,
        *,
        processed_mb: Any,
        post_processing_fn: Any,
        sequence_dim: int,
    ) -> torch.Tensor:
        processed_inputs = processed_mb.processed_inputs
        clean_input_ids = processed_inputs.input_ids
        completion_mask = (
            processed_mb.data_dict["token_mask"].to(clean_input_ids.device).bool()
        )
        seed = make_dllm_mask_seeds(clean_input_ids)

        def score_fn(
            masked_input_ids: torch.Tensor, target_ids: torch.Tensor
        ) -> torch.Tensor:
            processed_inputs.input_ids = masked_input_ids
            processed_inputs.target_ids = target_ids
            try:
                # Built inside score_fn, after the masked ids are swapped in:
                # prepare_model_forward snapshots processed_inputs, so hoisting
                # it would score every quadrature point against the clean
                # sequence instead of its own masked view.
                prepared = prepare_model_forward(
                    self.model,
                    processed_inputs,
                    device_mesh=self.device_mesh,
                    cp_size=self.cp_size,
                    padding_token_id=self.tokenizer.pad_token_id or 0,
                    is_reward_model=False,
                    allow_flash_attn_args=self.allow_flash_attn_args,
                )
                with prepared.model_context_factory(), self._autocast_context():
                    logprobs, _metrics, _ = forward_with_post_processing_fn(
                        model=self.model,
                        prepared=prepared,
                        post_processing_fn=post_processing_fn,
                        processed_mb=processed_mb,
                        sampling_params=self.sampling_params,
                        sequence_dim=sequence_dim,
                    )
            finally:
                processed_inputs.input_ids = clean_input_ids
                processed_inputs.target_ids = None
            return logprobs

        return accumulate_elbo_logprobs(
            self.elbo_estimator,
            input_ids=clean_input_ids,
            completion_mask=completion_mask,
            seed=seed,
            score_fn=score_fn,
        )

    def _gdpo_train_elbo_scorer(self, sequence_dim: int):
        post_processor = LogprobsPostProcessor(
            cfg=self.cfg,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            enable_seq_packing=self.enable_seq_packing,
            sampling_params=self.sampling_params,
            shift_targets=self.diffusion_cfg.shift_targets,
        )

        def score(processed_mb: Any) -> torch.Tensor:
            return self._gdpo_elbo_logprobs(
                processed_mb=processed_mb,
                post_processing_fn=post_processor,
                sequence_dim=sequence_dim,
            )

        return score

    def _logprobs_for_microbatch(
        self,
        *,
        processed_mb: Any,
        post_processing_fn: Any,
        sequence_dim: int,
    ) -> torch.Tensor:
        return self._gdpo_elbo_logprobs(
            processed_mb=processed_mb,
            post_processing_fn=post_processing_fn,
            sequence_dim=sequence_dim,
        )

    @torch.no_grad()
    def generate(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
        *,
        seed: int | None = None,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate rollouts by block-wise denoising on live training weights."""
        if seed is None:
            raise ValueError("GDPO generation requires an explicit rollout seed.")

        batch_stop_strings = data.get("stop_strings", [])
        if any(batch_stop_strings):
            raise ValueError(
                "Per-sample stop_strings are not supported by GDPO generation. "
                "Use stop_token_ids instead."
            )

        generation_cfg = self.cfg["generation"]
        assert generation_cfg is not None
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to the Automodel generation worker is not properly "
                f"right-padded: {error_msg}"
            )

        device = torch.cuda.current_device()
        pad_id = generation_cfg.get("_pad_token_id", self.tokenizer.pad_token_id)
        self.model.eval()
        canvas, attention_mask = build_canvas(
            data["input_ids"].to(device),
            data["input_lengths"].to(device),
            gen_length=generation_cfg["max_new_tokens"],
            mask_id=self.elbo_estimator.mask_id,
            pad_id=pad_id,
        )

        generator = torch.Generator(device=canvas.device)
        dp_rank = self.dp_mesh.get_local_rank()
        generator.manual_seed((seed + dp_rank) % (2**31 - 1))

        def logits_fn(input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            logits = self.model(
                input_ids=input_ids, attention_mask=mask.to(input_ids.dtype)
            ).logits
            return logits.full_tensor() if isinstance(logits, DTensor) else logits

        denoised = block_denoise(
            logits_fn,
            canvas,
            attention_mask,
            gen_start=data["input_ids"].shape[1],
            mask_id=self.elbo_estimator.mask_id,
            steps=self.denoise_cfg.diffusion_steps,
            block_length=self.denoise_cfg.block_length,
            temperature=0.0 if greedy else generation_cfg["temperature"],
            top_k=None if greedy else generation_cfg["top_k"],
            top_p=1.0 if greedy else generation_cfg["top_p"],
            cfg_scale=self.denoise_cfg.cfg_scale,
            generator=generator,
        )

        stop_token_ids = generation_cfg["stop_token_ids"] or [
            self.tokenizer.eos_token_id
        ]
        outputs = unpack_generations(
            denoised,
            data["input_lengths"].to(device),
            gen_start=data["input_ids"].shape[1],
            eos_token_ids=stop_token_ids,
            pad_id=pad_id,
        )
        outputs["logprobs"] = torch.zeros_like(
            outputs["output_ids"], dtype=torch.float32
        )
        return BatchedDataDict[GenerationOutputSpec](
            {key: value.cpu() for key, value in outputs.items()}
        )

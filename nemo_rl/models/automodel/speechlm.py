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

"""Native public SpeechLM integration for the synchronous Automodel worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel
from nemo.collections.speechlm2.parts.weight_loading import canonical_parameter_name
from nemo.collections.speechlm2.vllm.salm.config import NeMoSpeechLMConfig
from omegaconf import OmegaConf
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from transformers import AutoConfig

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.models.automodel.speechlm_logprobs import response_logprobs
from nemo_rl.models.policy import SpeechLMNativeConfig


class SpeechLMPolicy(SALMAutomodel):
    """Keep native state names and use Speech's supported input expansion."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        native_model: SpeechLMNativeConfig | dict[str, Any],
    ) -> None:
        self.native_model: dict[str, Any] = SpeechLMNativeConfig.model_validate(
            native_model
        ).model_dump()
        self.config: Any = None
        cfg = dict(cfg)
        cfg.update(self.native_model["model_config_overrides"])
        cfg["use_nemo_automodel"] = True
        cfg["retain_native_mtp"] = True
        cfg["mtp"] = {**(cfg.get("mtp") or {}), "enabled": False}
        cfg["packed_sequences"] = False
        cfg["packed_encoder_sequences"] = False
        cfg["packed_encoder_cp"] = False
        cfg["cross_entropy_backend"] = "eager"
        # Freeze each component, then enable only the configured exact names.
        # Automodel applies this policy before loading and distributed wrapping.
        for prefix, key in (
            ("llm.", "automodel_freeze_config"),
            ("perception.", "perception_freeze_config"),
        ):
            names = [
                name.removeprefix(prefix)
                for name in self.native_model["trainable_names"]
                if name.startswith(prefix)
            ]
            cfg[key] = {
                "freeze_modules": [{"glob": "*"}],
                "unfreeze_parameters": names,
            }
        # pyrefly resolves super() of the Any-typed nemo base to object.
        # pyrefly: ignore[bad-argument-count, unexpected-keyword]
        super().__init__(cfg)

    @classmethod
    def from_native_pretrained(
        cls,
        model_name: str | Path,
        *,
        native_model: SpeechLMNativeConfig | dict[str, Any],
        peft_config: Any = None,
        attn_implementation: Any = None,
        sdpa_method: Any = None,
        **kwargs: Any,
    ) -> SpeechLMPolicy:
        """Load the validated native update surface through Speech's HF loader."""
        if native_model is None:
            raise ValueError("Native SpeechLM requires an explicit native_model config")
        if peft_config is not None:
            raise ValueError(
                "Native SpeechLM GSPO uses an exact parameter surface, not PEFT"
            )
        # Native backbone/backend selection comes from its explicit model config;
        # generic HF-only kwargs must not enter Speech's constructor.
        for name in ("use_liger_kernel", "force_hf"):
            if kwargs.pop(name, False):
                raise ValueError(f"{name} is not supported for native SpeechLM")
        native_config = SpeechLMNativeConfig.model_validate(native_model).model_dump()
        # FP32 loader storage protects trainable checkpoint values. FSDP's BF16
        # compute policy reduces frozen storage before materialization.
        kwargs["torch_dtype"] = torch.float32
        model = super().from_pretrained(
            model_name,
            native_model=native_config,
            strict=True,
            **kwargs,
        )
        model.validate_update_surface()
        model.train(model.training)
        return cast(SpeechLMPolicy, model)

    def configure_model(self, distributed_setup: Any = None, **kwargs) -> None:
        super().configure_model(
            distributed_setup=distributed_setup,
            activation_checkpointing_perception=self.native_model[
                "activation_checkpointing_perception"
            ],
            perception_mixed_precision=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=True,
            ),
            **kwargs,
        )
        # Distributed loading defers backbone construction until this method.
        # Reuse its parsed config rather than serialized JSON float markers.
        config = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(config, dict):
            raise TypeError("Native SpeechLM requires a dictionary model config")
        if not all(isinstance(key, str) for key in config):
            raise TypeError("Native SpeechLM config keys must be strings")
        config = cast(dict[str, Any], config)
        config["llm_config"] = self.llm.config.to_dict()
        self.config = NeMoSpeechLMConfig(**config)

    def refit_parameter_names(self) -> tuple[str, ...]:
        """Serving loads frozen native weights from the seed; refit the update surface."""
        return tuple(self.native_model["trainable_names"])

    def validate_update_surface(self) -> None:
        selected = {
            canonical_parameter_name(name): value
            for name, value in self.named_parameters()
            if value.requires_grad
        }
        expected = set(self.native_model["trainable_names"])
        if set(selected) != expected:
            raise ValueError(
                f"Native trainable surface mismatch: missing={sorted(expected - set(selected))}, extra={sorted(set(selected) - expected)}"
            )
        if any(parameter.dtype != torch.float32 for parameter in selected.values()):
            raise ValueError("Every native trainable tensor must have FP32 storage")
        if (
            len(selected) != self.native_model["expected_trainable_tensors"]
            or sum(parameter.numel() for parameter in selected.values())
            != self.native_model["expected_trainable_scalars"]
        ):
            raise ValueError(
                "Native update tensor/scalar counts do not match the configured contract"
            )
        if any(
            parameter.dtype != torch.bfloat16
            for name, parameter in self.named_parameters()
            if not parameter.requires_grad and not name.startswith("ctc_timestamp.")
        ):
            raise ValueError("Frozen native parameter storage must remain BF16")
        if self.ctc_timestamp is not None:
            artifact_dtypes = self.cfg.ctc_timestamp_artifact.tensor_dtypes
            for name, parameter in self.ctc_timestamp.named_parameters():
                if parameter.requires_grad or parameter.dtype != getattr(
                    torch, artifact_dtypes[name]
                ):
                    raise ValueError(
                        "CTC parameters must remain frozen in their source storage dtype"
                    )
        if self.llm.mtp is not None and any(
            parameter.requires_grad for parameter in self.llm.mtp.parameters()
        ):
            raise ValueError("Native MTP must remain frozen")

    def train(self, mode: bool = True) -> SpeechLMPolicy:
        super().train(mode)
        if (
            self.perception is not None
            and self.native_model["deterministic_perception"]
        ):
            self.perception.eval()
        return self

    def next_token_logprobs(
        self,
        model_batch: dict[str, Any],
        response_mask: torch.Tensor,
        sampling_params: TrainingSamplingParams | None = None,
    ) -> torch.Tensor:
        if need_top_k_or_top_p_filtering(sampling_params):
            raise ValueError(
                "Native SpeechLM response projection requires top_p=1 and disabled top_k"
            )
        inputs = self.prepare_inputs(
            {
                "input_ids": model_batch["input_ids"],
                "audios": model_batch["audio_samples"],
                "audio_lens": model_batch["audio_lengths"].reshape(-1),
                "loss_mask": response_mask.bool(),
            },
            include_mtp_inputs=False,
        )
        if (
            inputs["input_embeds"].shape[1] + 1
            > self.native_model["max_expanded_sequence_length"]
        ):
            raise ValueError(
                "Expanded audio/text sequence exceeds the native context contract"
            )
        llm = self.llm
        if llm is None:
            raise RuntimeError("Native SpeechLM backbone has not been configured")
        outputs = llm(
            inputs_embeds=inputs["input_embeds"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            compute_logits=False,
            compute_mtp=False,
            return_dict=True,
            **inputs["llm_kwargs"],
        )
        hidden = outputs.hidden_states[-1]
        head = self.llm.get_output_embeddings()
        weight = (
            head.weight.full_tensor()
            if isinstance(head.weight, DTensor)
            else head.weight
        )
        # TE bias=False uses an empty ordinary tensor, which is not a bias
        # parameter/buffer and can remain on CPU after model onload. DTensor
        # numel() is global, so a real bias with an empty local shard is kept.
        bias = getattr(head, "bias", None)
        if bias is not None and bias.numel() == 0:
            bias = None
        elif isinstance(bias, DTensor):
            bias = bias.full_tensor()
        target_ids = inputs["target_ids"]
        if not isinstance(target_ids, torch.Tensor):
            raise TypeError("Native target_ids must be a tensor")
        result = response_logprobs(
            hidden,
            target_ids,
            model_batch["input_ids"],
            response_mask,
            weight,
            bias=bias,
            chunk_size=self.native_model["projection_chunk_size"],
            temperature=1.0 if sampling_params is None else sampling_params.temperature,
        )
        dummy_audio_loss = inputs.get("dummy_audio_loss")
        if dummy_audio_loss is not None:
            if not isinstance(dummy_audio_loss, torch.Tensor):
                raise TypeError("Native dummy_audio_loss must be a tensor")
            result = result + dummy_audio_loss * 0
        return result


def register_speechlm_config() -> None:
    """Register only the public HF config; serving plugins remain actor-owned."""
    AutoConfig.register("nemo_speechlm", NeMoSpeechLMConfig, exist_ok=True)

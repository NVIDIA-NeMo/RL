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

from typing import Any, TYPE_CHECKING

import torch
from nemo_automodel.components.utils.parameter_config import ParameterConfig
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import AutoConfig

from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel
from nemo.collections.speechlm2.parts.weight_loading import canonical_parameter_name
from nemo.collections.speechlm2.vllm.salm.config import NeMoSpeechLMConfig
from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.models.automodel.speechlm_logprobs import response_logprobs


if TYPE_CHECKING:
    from nemo_rl.models.policy import SpeechLMNativeConfig


class SpeechLMPolicy(SALMAutomodel):
    """Keep native state names and use Speech's supported input expansion."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        parameter_config: ParameterConfig,
        native_model: SpeechLMNativeConfig,
    ) -> None:
        self.native_model = dict(native_model)
        cfg = dict(cfg)
        cfg.update(self.native_model.get("model_config_overrides", {}))
        cfg["use_nemo_automodel"] = True
        cfg["retain_native_mtp"] = True
        cfg["mtp"] = {**cfg.get("mtp", {}), "enabled": False}
        cfg["packed_sequences"] = False
        cfg["packed_encoder_sequences"] = False
        cfg["packed_encoder_cp"] = False
        cfg["cross_entropy_backend"] = "eager"
        super().__init__(cfg, parameter_config=parameter_config)
        self.config = NeMoSpeechLMConfig(**cfg)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        native_model: SpeechLMNativeConfig,
        peft_config=None,
        attn_implementation=None,
        sdpa_method=None,
        **kwargs,
    ) -> SpeechLMPolicy:
        if peft_config is not None:
            raise ValueError(
                "Native SpeechLM GSPO uses an exact parameter surface, not PEFT"
            )
        # Native backbone/backend selection comes from its explicit model config;
        # generic HF-only kwargs must not enter Speech's constructor.
        for name in ("use_liger_kernel", "force_hf"):
            if kwargs.pop(name, False):
                raise ValueError(f"{name} is not supported for native SpeechLM")
        activation_checkpointing = kwargs.pop("activation_checkpointing", False)
        kwargs["activation_checkpointing_llm"] = activation_checkpointing
        parameter_config = ParameterConfig(
            tuple(native_model["trainable_names"]), torch.float32, torch.bfloat16
        )
        model = super().from_pretrained(
            model_name,
            parameter_config=parameter_config,
            native_model=native_model,
            strict=True,
            **kwargs,
        )
        model.validate_update_surface()
        model.train(model.training)
        return model

    def configure_model(self, distributed_setup: Any = None, **kwargs) -> None:
        super().configure_model(
            distributed_setup=distributed_setup,
            activation_checkpointing_perception=self.native_model.get(
                "activation_checkpointing_perception", True
            ),
            perception_mixed_precision=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=True,
            ),
            **kwargs,
        )

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
            for parameter in self.parameters()
            if not parameter.requires_grad
        ):
            raise ValueError("Frozen native parameter storage must remain BF16")
        if self.llm.mtp is None or any(
            parameter.requires_grad for parameter in self.llm.mtp.parameters()
        ):
            raise ValueError("Native MTP must be retained and frozen")

    def train(self, mode: bool = True) -> SpeechLMPolicy:
        super().train(mode)
        if self.perception is not None and self.native_model.get(
            "deterministic_perception", True
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
        outputs = self.llm(
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
        bias = (
            head.bias.full_tensor()
            if isinstance(getattr(head, "bias", None), DTensor)
            else getattr(head, "bias", None)
        )
        result = response_logprobs(
            hidden,
            inputs["target_ids"],
            model_batch["input_ids"],
            response_mask,
            weight,
            bias=bias,
            chunk_size=self.native_model.get("projection_chunk_size", 256),
            temperature=1.0 if sampling_params is None else sampling_params.temperature,
        )
        if inputs.get("dummy_audio_loss") is not None:
            result = result + inputs["dummy_audio_loss"] * 0
        return result


def register_speechlm_config() -> None:
    """Register only the public HF config; serving plugins remain actor-owned."""
    AutoConfig.register("nemo_speechlm", NeMoSpeechLMConfig, exist_ok=True)

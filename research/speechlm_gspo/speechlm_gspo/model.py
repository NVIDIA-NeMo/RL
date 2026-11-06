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

"""Native SpeechLM policy model served through the generic Automodel worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel
from nemo.collections.speechlm2.parts.weight_loading import canonical_parameter_name
from nemo.collections.speechlm2.vllm.salm.config import NeMoSpeechLMConfig
from omegaconf import OmegaConf
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoProcessor

from nemo_rl.models.automodel.utils import AUTOMODEL_FACTORY
from speechlm_gspo.alignment import align_to_text
from speechlm_gspo.config import SpeechLMNativeConfig
from speechlm_gspo.data import Nemotron3_5TranscribeProcessor

MODEL_TYPE = "nemo_speechlm"


class SpeechLMPolicy(SALMAutomodel):
    """Keep native state names and return logits in the canonical text layout."""

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
    def from_pretrained(
        cls,
        model_name: str | Path,
        *,
        native_model: SpeechLMNativeConfig | dict[str, Any] | None = None,
        peft_config: Any = None,
        attn_implementation: Any = None,
        sdpa_method: Any = None,
        use_liger_kernel: bool = False,
        force_hf: bool = False,
        **kwargs: Any,
    ) -> SpeechLMPolicy:
        """Load the validated native update surface through Speech's HF loader.

        Attention and SDPA selection come from ``model_config_overrides``;
        Automodel's HF-only options are rejected rather than ignored.
        """
        if native_model is None:
            raise ValueError(
                "Set policy.automodel_cfg.automodel_kwargs.native_model for SpeechLM"
            )
        if peft_config is not None:
            raise ValueError(
                "SpeechLM GSPO trains an exact parameter surface, not PEFT"
            )
        if use_liger_kernel or force_hf:
            raise ValueError("use_liger_kernel and force_hf are not supported")
        # FP32 loader storage protects trainable checkpoint values. FSDP's BF16
        # compute policy reduces frozen storage before materialization.
        kwargs["torch_dtype"] = torch.float32
        model = super().from_pretrained(
            model_name,
            native_model=SpeechLMNativeConfig.model_validate(native_model).model_dump(),
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

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_samples: torch.Tensor,
        audio_lengths: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Return next-token logits for ``input_ids``, shaped ``[B, S, V]``.

        NeMo RL passes an all-ones causal ``attention_mask`` and masks padding
        after the forward, so padding is recognized by the text pad ID here.
        """
        if use_cache:
            raise ValueError("SpeechLM training forward does not use a KV cache")
        target_mask = (input_ids != self.text_pad_id) & (
            input_ids != self.audio_locator_tag_id
        )
        # Nothing predicts the first token; left padding would otherwise
        # assign it to a pad position.
        target_mask[:, 0] = False
        inputs = self.prepare_inputs(
            {
                "input_ids": input_ids,
                "audios": audio_samples,
                "audio_lens": audio_lengths.reshape(-1),
                "loss_mask": target_mask,
            },
            include_mtp_inputs=False,
        )
        if (
            inputs["input_embeds"].shape[1] + 1
            > self.native_model["max_expanded_sequence_length"]
        ):
            raise ValueError(
                "Expanded audio/text sequence exceeds max_expanded_sequence_length"
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
        hidden = align_to_text(
            outputs.hidden_states[-1], inputs["target_ids"], input_ids, target_mask
        )
        weight, bias = _full_output_head(llm.get_output_embeddings())
        logits = F.linear(hidden.to(weight.dtype), weight, bias)
        dummy_audio_loss = inputs.get("dummy_audio_loss")
        if dummy_audio_loss is not None:
            # Keeps every perception parameter in the graph for FSDP reduction.
            logits = logits + dummy_audio_loss * 0
        return logits


def register_speechlm() -> None:
    """Make the native config, policy model and processor resolvable by name."""
    AutoConfig.register(MODEL_TYPE, NeMoSpeechLMConfig, exist_ok=True)
    AutoProcessor.register(
        NeMoSpeechLMConfig, Nemotron3_5TranscribeProcessor, exist_ok=True
    )
    AUTOMODEL_FACTORY[MODEL_TYPE] = SpeechLMPolicy


def _full_output_head(
    head: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gather the frozen head once; FSDP may leave it sharded after the LLM call."""
    weight = head.weight
    if isinstance(weight, DTensor):
        weight = weight.full_tensor()
    # TE bias=False uses an empty ordinary tensor, which is not a bias
    # parameter/buffer and can remain on CPU after model onload. DTensor
    # numel() is global, so a real bias with an empty local shard is kept.
    bias = getattr(head, "bias", None)
    if bias is not None and bias.numel() == 0:
        bias = None
    elif isinstance(bias, DTensor):
        bias = bias.full_tensor()
    return weight, bias

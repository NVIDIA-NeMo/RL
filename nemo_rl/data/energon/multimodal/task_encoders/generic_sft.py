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

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Protocol, cast

import torch
from megatron.energon import SampleDecoder, stateless

from nemo_rl.data.energon.multimodal.model_families import (
    ALL_MODEL_FAMILIES,
    supports_model_families,
)
from nemo_rl.data.energon.multimodal.task_encoders.base import (
    BaseSFTTaskEncoder,
    SFTCooker,
)
from nemo_rl.data.energon.multimodal.task_encoders.media import (
    materialize_media_value,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    EncodedSFTSample,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class SFTProcessorAdapter(Protocol):
    """Boundary between canonical and model-specific SFT data."""

    @property
    def fingerprint(self) -> str: ...

    def encode(self, sample: CanonicalSFTSample) -> EncodedSFTSample: ...


def _normalize_messages(
    sample: CanonicalSFTSample, *, materialize: bool = True
) -> list[dict[str, Any]]:
    """Validate the message structure and attach each part's media.

    Args:
        sample: The cooked conversation.
        materialize: Decode each media value and attach the payload. Set False
            to attach the ``MediaRef`` instead.

    The Nemotron renderers replace every media part with text built from
    metadata and then overwrite ``message["content"]`` wholesale, so decoding
    for them is pure waste. It is also waste paid at the wrong time: this runs
    in pre-encode, before ``select_samples_to_pack``, so rows that selection
    discards are decoded too. Measured on video rows at 2771 ms against the
    Megatron reference's 4.7 ms, which defers all frame work to post-encode.

    Only ``GenericSFTTaskEncoder.encode`` consumes the payload, via
    ``get_formatted_message_log``, so it keeps the default.
    """
    messages = deepcopy(sample.messages)
    used_media: list[int] = []
    tool_call_ids: set[str] = set()

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(
                f"Sample {sample.__key__!r} contains a non-object message."
            )
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Sample {sample.__key__!r} has invalid role {role!r}.")

        content = message.get("content")
        if content is None:
            content = []
        elif isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif not isinstance(content, list):
            raise ValueError(
                f"Sample {sample.__key__!r} has unsupported content type "
                f"{type(content).__name__}."
            )

        normalized_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError(
                    f"Sample {sample.__key__!r} contains a non-object content part."
                )
            part = dict(part)
            media_index = part.pop("media_index", None)
            if media_index is not None:
                if not isinstance(media_index, int) or not 0 <= media_index < len(
                    sample.media
                ):
                    raise ValueError(
                        f"Sample {sample.__key__!r} has invalid media index "
                        f"{media_index!r}."
                    )
                media_ref = sample.media[media_index]
                declared_type = part.get("type")
                if declared_type not in (None, media_ref.modality):
                    raise ValueError(
                        f"Sample {sample.__key__!r} maps {declared_type!r} content "
                        f"to {media_ref.modality!r} media."
                    )
                part["type"] = media_ref.modality
                part[media_ref.modality] = (
                    materialize_media_value(
                        media_ref.value,
                        modality=media_ref.modality,
                        sample=sample,
                    )
                    if materialize
                    else media_ref
                )
                used_media.append(media_index)
            normalized_content.append(part)
        message["content"] = normalized_content

        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict) or not isinstance(
                tool_call.get("id"), str
            ):
                raise ValueError(
                    f"Sample {sample.__key__!r} has a tool call without a string id."
                )
            tool_call_ids.add(tool_call["id"])
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or tool_call_id not in tool_call_ids:
                raise ValueError(
                    f"Sample {sample.__key__!r} has a dangling tool result id "
                    f"{tool_call_id!r}."
                )

    if used_media != list(range(len(sample.media))):
        raise ValueError(
            f"Sample {sample.__key__!r} media occurrences must be referenced once "
            f"in order; got {used_media}."
        )
    return messages


class HFMultimodalSFTProcessorAdapter:
    """Hugging Face implementation of the generic processor boundary."""

    def __init__(
        self,
        *,
        processor: Any,
        max_sequence_length: int,
        add_bos: bool,
        add_eos: bool,
        add_generation_prompt: bool,
    ) -> None:
        if not hasattr(processor, "apply_chat_template") or not hasattr(
            processor, "tokenizer"
        ):
            raise TypeError("Energon multimodal SFT requires a Hugging Face processor.")
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_generation_prompt = add_generation_prompt
        tokenizer = processor.tokenizer
        fingerprint_data = {
            "processor_class": type(processor).__name__,
            "processor_name": getattr(processor, "name_or_path", None),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name": getattr(tokenizer, "name_or_path", None),
            "chat_template": getattr(processor, "chat_template", None)
            or getattr(tokenizer, "chat_template", None),
            "max_sequence_length": max_sequence_length,
            "add_bos": add_bos,
            "add_eos": add_eos,
            "add_generation_prompt": add_generation_prompt,
        }
        encoded = json.dumps(fingerprint_data, sort_keys=True, default=str).encode(
            "utf-8"
        )
        self._fingerprint = hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def encode(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        messages = _normalize_messages(sample)
        message_log = get_formatted_message_log(
            messages,
            self.processor,
            TaskDataSpec(),
            add_bos_token=self.add_bos,
            add_eos_token=self.add_eos,
            add_generation_prompt=self.add_generation_prompt,
            tools=sample.tools,
        )
        length = sum(len(message["token_ids"]) for message in message_log)
        loss_multiplier = 1.0
        if length >= self.max_sequence_length:
            # Treat truncated messages as text only. Dropping the placeholder
            # tokens without dropping the media would leave the vision merge
            # with N features and no placeholders to scatter them into, which
            # raises rather than training a zero-weighted row.
            for message in message_log:
                message["token_ids"] = message["token_ids"][
                    : min(4, self.max_sequence_length // len(message_log))
                ]
                for key, value in list(message.items()):
                    if isinstance(value, PackedTensor):
                        message[key] = PackedTensor.empty_like(value)
            loss_multiplier = 0.0

        # group_key is the adapter fingerprint alone. Keying on the tensor names
        # present in message_log would split groups on any model-input difference,
        # but batch() keeps those tensors in the per-message dicts rather than in
        # a stacked batch tensor, so the finer split buys nothing.
        return EncodedSFTSample.derive_from(
            sample,
            message_log=message_log,
            length=length,
            packing_cost=length,
            loss_multiplier=loss_multiplier,
            group_key=(self.fingerprint,),
            sample_key=sample.__key__,
        )


@supports_model_families(ALL_MODEL_FAMILIES)
class GenericSFTTaskEncoder(BaseSFTTaskEncoder):
    """Encode, group, and batch complete multimodal SFT conversations."""

    # Energon reads 0 as "disable tolerance checking" (megatron/energon/errors.py),
    # which would let a systematically broken dataset retry forever. 1 fails on the
    # first bad sample; raise it to tolerate transient decode errors.
    __default_failure_tolerance__ = 1
    sample_schema = "nemo_rl.sft.encoded.v1"
    # Match the existing HF VLM path. Its processor expects PIL RGB images.
    decoder = SampleDecoder(image_decode="pilrgb")

    def __init__(
        self,
        *,
        adapter: SFTProcessorAdapter,
        cooker_functions: Sequence[SFTCooker],
        include_source_ids: bool,
    ) -> None:
        super().__init__(cooker_functions=cooker_functions)
        self.adapter = adapter
        self.include_source_ids = include_source_ids

    @stateless
    def preencode_sample(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        return self.adapter.encode(sample)

    @stateless
    def postencode_sample(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        return sample

    def batch_group_criterion(
        self, sample: EncodedSFTSample
    ) -> tuple[tuple[Any, ...], None]:
        return sample.group_key, None

    @stateless
    def batch(self, samples: list[EncodedSFTSample]) -> BatchedDataDict[Any]:
        if not all(isinstance(sample, EncodedSFTSample) for sample in samples):
            raise TypeError("Energon SFT batches accept only encoded samples.")
        encoded_samples = cast(list[EncodedSFTSample], samples)
        values: dict[str, Any] = {
            "message_log": [sample.message_log for sample in encoded_samples],
            "loss_multiplier": torch.tensor(
                [sample.loss_multiplier for sample in encoded_samples],
                dtype=torch.float32,
            ),
        }
        if self.include_source_ids:
            values["source_ids"] = [sample.sample_key for sample in encoded_samples]
        return BatchedDataDict(values)

    @stateless
    def encode_batch(self, batch: BatchedDataDict[Any]) -> BatchedDataDict[Any]:
        return batch


def build_processor_adapter(
    *,
    processor_adapter: str,
    processor: Any,
    max_sequence_length: int,
    add_bos: bool,
    add_eos: bool,
    add_generation_prompt: bool,
) -> SFTProcessorAdapter:
    """Build the configured model processor adapter."""
    if processor_adapter != "hf_multimodal":
        raise ValueError(f"Unsupported SFT processor adapter {processor_adapter!r}.")
    return HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=max_sequence_length,
        add_bos=add_bos,
        add_eos=add_eos,
        add_generation_prompt=add_generation_prompt,
    )


__all__ = [
    "GenericSFTTaskEncoder",
    "HFMultimodalSFTProcessorAdapter",
    "SFTProcessorAdapter",
    "build_processor_adapter",
]

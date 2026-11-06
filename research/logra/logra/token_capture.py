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

"""Token-exact multimodal continuation and lossless rollout-media storage."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass, field, replace
from itertools import groupby
from typing import Any

import torch
import zstandard
from torch import Tensor

from nemo_rl.utils.routed_experts_codec import decode_routed_experts


def align_image_placeholders(
    engine_prompt: dict[str, Any], token_ids: list[int]
) -> None:
    """Align Omni image embeddings with the exact carried conversation tokens.

    Assistant text can retokenize differently. Native token-in HTTP preserves
    the original tokens, but remaps media only with the optional TQ capture
    service. This recipe uses HTTP capture and locates each image in that exact
    token sequence, including earlier images shifted by previous turns.
    """
    placeholders = engine_prompt.get("mm_placeholders") or {}
    if not placeholders:
        return
    if set(placeholders) != {"image"}:
        raise ValueError("CUA continuation supports image placeholders only")
    spans = placeholders["image"]
    if not spans:
        return
    original = engine_prompt["prompt_token_ids"]
    image_id = None
    blocks = []
    for span in spans:
        local = original[span.offset : span.offset + span.length]
        positions = list(range(span.length))
        if span.is_embed is not None:
            if span.is_embed.dtype != torch.bool or tuple(span.is_embed.shape) != (
                span.length,
            ):
                raise ValueError("Invalid image embedding mask")
            positions = span.is_embed.nonzero().flatten().tolist()
        if not positions or positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("CUA requires contiguous Omni image embeddings")
        token = local[positions[0]]
        if image_id is None:
            image_id = token
        if token != image_id or any(local[index] != image_id for index in positions):
            raise ValueError("Inconsistent Omni image context tokens")
        # The native span includes image-start/end markers; only is_embed
        # positions receive screenshot features. Preserve those markers too.
        blocks.append((local, positions[0], len(positions)))
    image_runs = []
    offset = 0
    for token, group in groupby(token_ids):
        length = sum(1 for _ in group)
        if token == image_id:
            image_runs.append((offset, length))
        offset += length
    if [length for _, length in image_runs] != [count for _, _, count in blocks]:
        raise ValueError("Carried image tokens disagree with rendered screenshots")
    corrected = []
    for span, (local, start, _), (offset, _) in zip(
        spans, blocks, image_runs, strict=True
    ):
        offset -= start
        if offset < 0 or token_ids[offset : offset + span.length] != local:
            raise ValueError("Image boundary tokens changed during continuation")
        corrected.append(replace(span, offset=offset))
    engine_prompt["mm_placeholders"] = {"image": corrected}


@dataclass
class CuaSegment:
    input_ids: Tensor
    token_mask: Tensor
    generation_logprobs: Tensor
    routed_experts: Tensor
    messages: list[dict[str, Any]]
    media: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _compressed_media: bytes | None = field(default=None, init=False, repr=False)

    def compress_media(self) -> None:
        """Keep prepared pixels losslessly compressed while a rollout waits."""
        if self.media is not None:
            self._compressed_media = zstandard.ZstdCompressor(level=1).compress(
                pickle.dumps(self.media, protocol=pickle.HIGHEST_PROTOCOL)
            )
            self.media = None

    def prepare_media(self, processor: Any) -> dict[str, Any]:
        """Materialize exact prepared pixels when a segment enters training."""
        if self._compressed_media is not None:
            self.media = pickle.loads(
                zstandard.ZstdDecompressor().decompress(self._compressed_media)
            )
            self._compressed_media = None
        if self.media is not None:
            return self.media
        # Import the native media helpers only when preparing training inputs.
        from nemo_rl.data.multimodal_utils import (
            PackedTensor,
            _materialize_ragged_pixel_values,
            extract_multimodal_model_inputs,
            resolve_to_image,
        )

        images = [
            resolve_to_image(part["image_url"]["url"])
            for message in self.messages
            if isinstance(message["content"], list)
            for part in message["content"]
            if part["type"] == "image_url"
        ]
        # A desktop action can change resolution midway through a trajectory.
        # Preserve each image's shape for native patchification instead of stacking.
        mixed_shapes = len({image.size for image in images}) > 1
        processed = processor(
            text="<image>" * len(images),
            images=images,
            return_tensors=None if mixed_shapes else "pt",
        )
        if mixed_shapes:
            processed = _materialize_ragged_pixel_values(processed, processor)
        image_id = processor.image_token_id
        if int((processed["input_ids"] == image_id).sum()) != int(
            (self.input_ids == image_id).sum()
        ):
            raise ValueError(
                "Captured tokens and processed screenshots have different image-token counts"
            )
        media = extract_multimodal_model_inputs(processor, processed)
        if mixed_shapes:
            # Keep raw images and their metadata in one conversation row.
            # AutoModel needs NCHW images; eager patchification loses that shape.
            for key in ("imgs_sizes", "num_frames"):
                media[key] = PackedTensor(
                    list(media[key].as_tensor().split(1)), dim_to_pack=0
                )
            for key in ("pixel_values", "imgs_sizes", "num_frames"):
                media[key] = PackedTensor.merge_segments(
                    [media[key].enable_deduplication()]
                )
        if "token_type_ids" in media or "mm_token_type_ids" in media:
            raise ValueError(
                "CUA currently requires the Nemotron image-placeholder processor"
            )
        self.media = media
        return media


@dataclass
class TurnRecorder:
    """Keep engine-returned tokens, probabilities, and routes across agent turns."""

    processor: Any
    segments: list[CuaSegment] = field(default_factory=list, init=False)
    previous_messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    previous_response: dict[str, Any] | None = field(default=None, init=False)

    @staticmethod
    def _text_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        result = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    "<image>" if part["type"] == "image_url" else part["text"]
                    for part in content
                )
            if not isinstance(content, str):
                raise ValueError(
                    "The RFC0015 agent must provide text or image_url content"
                )
            result.append({"role": message["role"], "content": content})
        return result

    def continues(self, payload: dict[str, Any]) -> bool:
        messages = payload["messages"]
        if (
            not self.segments
            or messages[: len(self.previous_messages)] != self.previous_messages
        ):
            return False
        # The original harness also starts a new segment if the chat template
        # rewrites previous assistant text when a new observation arrives.
        chat = self._text_messages(messages)
        kwargs = {"tools": payload["tools"]} if "tools" in payload else {}
        full = self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, **kwargs
        )
        prefix = self.processor.apply_chat_template(
            chat[:-1], tokenize=False, add_generation_prompt=False, **kwargs
        )
        return full.startswith(prefix)

    def request_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = copy.deepcopy(payload)
        # The native string renderer otherwise inserts separators or moves images
        # ahead of text. Explicit placeholders retain the original inline order;
        # image_url parts still carry the corresponding pixels to the engine.
        for message, text in zip(
            body["messages"], self._text_messages(payload["messages"]), strict=True
        ):
            if isinstance(message["content"], list):
                images = [
                    part for part in message["content"] if part["type"] == "image_url"
                ]
                message["content"] = [
                    {"type": "text", "text": text["content"]},
                    *images,
                ]
        body.update(
            logprobs=True,
            top_logprobs=1,
            return_token_ids=True,
            return_tokens_as_token_ids=True,
        )
        if self.continues(payload):
            # The native HTTP renderer splices in the actual previous output;
            # it does not retokenize the assistant's rendered text for training.
            body["required_prefix_token_ids"] = self.segments[-1].input_ids.tolist()
        return body

    def cached_response(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if payload["messages"] == self.previous_messages:
            return copy.deepcopy(self.previous_response)
        return None

    def record(
        self, payload: dict[str, Any], message: dict[str, Any]
    ) -> dict[str, Any]:
        prompt = torch.tensor(message["prompt_token_ids"], dtype=torch.long)
        generated = torch.tensor(message["generation_token_ids"], dtype=torch.long)
        probabilities = torch.tensor(
            message["generation_log_probs"], dtype=torch.float32
        )
        if (
            prompt.ndim != 1
            or generated.ndim != 1
            or probabilities.shape != generated.shape
        ):
            raise ValueError(
                "Invalid token/probability bundle from the generation engine"
            )
        if not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite generation log probabilities")
        tokens = torch.cat((prompt, generated))
        routes = decode_routed_experts(message["routed_experts"], torch.int16)
        if routes.ndim != 3 or routes.shape[0] != tokens.numel():
            raise ValueError(
                "Captured routes must align with every prompt and output token"
            )
        mask = torch.cat(
            (
                torch.zeros_like(prompt, dtype=torch.float32),
                torch.ones_like(generated, dtype=torch.float32),
            )
        )
        logprobs = torch.cat(
            (torch.zeros_like(prompt, dtype=torch.float32), probabilities)
        )
        if self.continues(payload):
            previous = self.segments[-1]
            length = previous.input_ids.numel()
            if not torch.equal(prompt[:length], previous.input_ids):
                raise ValueError(
                    "The engine did not preserve the requested token prefix"
                )
            if routes.shape[1:] != previous.routed_experts.shape[1:]:
                raise ValueError("The routing layout changed within an episode")
            mask[:length] = previous.token_mask
            logprobs[:length] = previous.generation_logprobs
            # First-writer-wins preserves the routes used for earlier actions;
            # the next prefill fills missing routes (including the last token).
            routes[:length] = torch.where(
                previous.routed_experts >= 0, previous.routed_experts, routes[:length]
            )
            self.segments.pop()
        self.segments.append(
            CuaSegment(
                tokens, mask, logprobs, routes, copy.deepcopy(payload["messages"])
            )
        )
        # Keep the agent-visible reply independent of large training metadata.
        response = {
            key: value
            for key, value in message.items()
            if key
            not in {
                "prompt_token_ids",
                "generation_token_ids",
                "generation_log_probs",
                "routed_experts",
            }
        }
        self.previous_messages = copy.deepcopy(payload["messages"])
        self.previous_response = copy.deepcopy(response)
        return response

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

import base64
import inspect
import logging
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedTokenizerBase
from transformers.audio_utils import load_audio
from transformers.video_utils import load_video

# List of allowed placeholder strings for different media types in the dataset string
# e.g. "This is an example of <image>"
MEDIA_TAGS = {
    "image": "<image>",
    "video": "<video>",
    "audio": "<audio>",
    "video-audio": "<video-audio>",
}
MEDIA_TAGS_REVERSED = {v: k for k, v in MEDIA_TAGS.items()}

DEFAULT_MEDIA_EXTENSIONS = {
    "image": ["png", "jpeg", "jpg", "img"],
    "video": ["mp4"],
    "video-audio": ["mp4"],
    "audio": ["wav", "flac", "mp3"],
}

_PLACEHOLDER_STYLE_PROCESSOR_NAMES = frozenset(
    {
        "NemotronNanoVLV2Processor",
        "NemotronH_Nano_Omni_Reasoning_V3Processor",
    }
)


# different media namings maybe used in the raw dataset,
# in which case, they need to be mapped to the allowed ones
# WARNING: values cannot be used as the keys in the same dict to avoid cyclic graph
MEDIA_TAGS_TO_ALLOWED = {
    "speech": "audio",
    "speeches": "audio",
    "sound": "audio",
    "audios": "audio",
    "images": "image",
    "videos": "video",
}


# Build a pattern like: <image>|<video>|<audio>|<video-audio>
MEDIA_TAG_PATTERN = re.compile(
    r"(" + "|".join(re.escape(tag) for tag in MEDIA_TAGS.values()) + ")"
)

logger = logging.getLogger(__name__)


def uses_image_placeholder(processor: Any) -> bool:
    """Return whether a processor requires explicit image placeholders.

    Args:
        processor: Multimodal processor to classify.

    Returns:
        Whether the processor expands image placeholders through ``__call__``
        rather than tokenized ``apply_chat_template``.
    """
    return type(processor).__name__ in _PLACEHOLDER_STYLE_PROCESSOR_NAMES


class PackedTensor:
    """Wrapper around a list of torch tensors and a dimension along which to pack the tensors.

    This class is used to wrap a list of tensors along with a `dim_to_pack` parameter.
    It can be used for data that can be packed along different dimensions (such as multimodal data).

    `dim_to_pack` is used to specify the dimension along which to pack the tensors.

    The list of tensors can be returned as a single packed tensor by calling `as_tensor` which will concatenate the tensors along the `dim_to_pack` dimension.
    """

    def __init__(
        self,
        tensors: Union[torch.Tensor, list[Optional[torch.Tensor]], list[None]],
        dim_to_pack: int,
        *,
        pad_to_max_shape: bool = False,
        dedup_indices: Optional[list[int]] = None,
    ) -> None:
        """Wrap per-item tensors for concatenation along ``dim_to_pack``.

        Args:
            tensors: A tensor or list of per-item tensors. List entries may be
                ``None`` for items without this modality.
            dim_to_pack: Dimension along which ``as_tensor`` concatenates.
            pad_to_max_shape: Pad every non-packing dimension to its batch-wide
                maximum before concatenating. All tensors must have the same rank.
        """
        assert tensors is not None, "Input tensors to PackedTensor cannot be None"

        if isinstance(tensors, torch.Tensor):
            self.tensors: list[Optional[torch.Tensor]] = [tensors]
        elif isinstance(tensors, list):
            assert len(tensors) > 0, (
                "Input tensors to PackedTensor must be a non-empty list"
            )
            self.tensors: list[Optional[torch.Tensor]] = tensors
        else:
            raise ValueError(
                f"Unsupported type for input tensors to PackedTensor: {type(tensors)}"
            )
        self.dim_to_pack = dim_to_pack
        self.pad_to_max_shape = pad_to_max_shape
        if dedup_indices is not None:
            assert dedup_indices, "dedup_indices must be non-empty when provided"
            assert min(dedup_indices) >= 0, (
                "dedup_indices must contain only non-negative values"
            )
            assert max(dedup_indices) < len(self.tensors), (
                "dedup_indices cannot reference out-of-range unique tensor indices"
            )
        self._dedup_indices = dedup_indices

    def as_tensor(
        self, device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        if device is not None:
            # Move only non-None tensors to device, preserve Nones
            for i, item in enumerate(self.tensors):
                if item is not None:
                    self.tensors[i] = item.to(device)
        tensors = self.tensors
        if self._dedup_indices is not None:
            tensors = [self.tensors[index] for index in self._dedup_indices]
        non_none_tensors = [t for t in tensors if t is not None]
        if len(non_none_tensors) == 0:
            return None

        # Some multimodal processors produce a different shape per prompt,
        # such as dynamic-resolution images, variable-frame videos, or audio
        # feature sequences. Concatenation already permits the packing
        # dimension to vary; when explicitly requested, pad every other
        # dimension to the largest size in the batch.
        if self.pad_to_max_shape:
            ranks = {tensor.ndim for tensor in non_none_tensors}
            if len(ranks) != 1:
                raise ValueError(
                    "pad_to_max_shape requires tensors with the same rank, "
                    f"but received ranks {sorted(ranks)}"
                )

            rank = ranks.pop()
            pack_dim = (
                self.dim_to_pack if self.dim_to_pack >= 0 else rank + self.dim_to_pack
            )
            if not 0 <= pack_dim < rank:
                raise IndexError(
                    f"dim_to_pack={self.dim_to_pack} is invalid for tensors with rank {rank}"
                )
            max_shape = [
                max(tensor.shape[dim] for tensor in non_none_tensors)
                for dim in range(rank)
            ]

            def pad_to_batch_shape(tensor: torch.Tensor) -> torch.Tensor:
                padding = []
                for dim in reversed(range(rank)):
                    padding.extend(
                        (
                            0,
                            0
                            if dim == pack_dim
                            else max_shape[dim] - tensor.shape[dim],
                        )
                    )
                return F.pad(tensor, padding)

            non_none_tensors = [
                pad_to_batch_shape(tensor) for tensor in non_none_tensors
            ]

        return torch.cat(non_none_tensors, dim=self.dim_to_pack).to(device)

    def __len__(self) -> int:
        if self._dedup_indices is not None:
            return len(self._dedup_indices)
        return len(self.tensors)

    def to(self, device: str | torch.device | torch.dtype) -> "PackedTensor":
        self.tensors = [
            item.to(device) if item is not None else None for item in self.tensors
        ]
        return self

    def slice(self, indices: Union[list[int], torch.Tensor]) -> "PackedTensor":
        idx = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        if self._dedup_indices is not None:
            selected = [self._dedup_indices[i] for i in idx]
            used_unique_indices = sorted(set(selected))
            remap = {
                old_index: new_index
                for new_index, old_index in enumerate(used_unique_indices)
            }
            return PackedTensor(
                [self.tensors[i] for i in used_unique_indices],
                self.dim_to_pack,
                pad_to_max_shape=self.pad_to_max_shape,
                dedup_indices=[remap[i] for i in selected],
            )
        tensors = [self.tensors[i] for i in idx]
        return PackedTensor(
            tensors,
            self.dim_to_pack,
            pad_to_max_shape=self.pad_to_max_shape,
        )

    @classmethod
    def empty_like(cls, other: "PackedTensor") -> "PackedTensor":
        """Return a new PackedTensor with same length and dim_to_pack as `other`, with all entries None."""
        return cls(
            [None] * len(other.tensors),
            other.dim_to_pack,
            pad_to_max_shape=other.pad_to_max_shape,
            dedup_indices=(
                list(other._dedup_indices) if other._dedup_indices is not None else None
            ),
        )

    @classmethod
    def concat(cls, from_packed_tensors: list["PackedTensor"]) -> "PackedTensor":
        """Concatenate a list of PackedTensor objects into a single PackedTensor.

        The underlying tensors from the PackedTensors are combined into a single list of tensors and used to create a new PackedTensor.

        Each batch must have the same dim_to_pack.

        Example:
        ```{doctest}
        >>> import torch
        >>> from nemo_rl.data.multimodal_utils import PackedTensor
        >>> p1 = PackedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])], dim_to_pack=0)
        >>> p2 = PackedTensor([torch.tensor([7, 8, 9])], dim_to_pack=0)
        >>> p3 = PackedTensor.concat([p1, p2])
        >>> p3.tensors
        [tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9])]
        >>> p3.as_tensor()
        tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>>
        ```
        """
        dim_to_packs = [batch.dim_to_pack for batch in from_packed_tensors]
        assert len(set(dim_to_packs)) == 1, (
            "All packed tensors must have the same dim_to_pack"
        )
        pad_to_max_shapes = [batch.pad_to_max_shape for batch in from_packed_tensors]
        assert len(set(pad_to_max_shapes)) == 1, (
            "All packed tensors must have the same pad_to_max_shape setting"
        )
        if any(
            packed_tensor._dedup_indices is not None
            for packed_tensor in from_packed_tensors
        ):
            tensors: list[Optional[torch.Tensor]] = []
            dedup_indices: list[int] = []
            offset = 0
            for packed_tensor in from_packed_tensors:
                tensors.extend(packed_tensor.tensors)
                if packed_tensor._dedup_indices is None:
                    dedup_indices.extend(
                        range(offset, offset + len(packed_tensor.tensors))
                    )
                else:
                    dedup_indices.extend(
                        offset + index for index in packed_tensor._dedup_indices
                    )
                offset += len(packed_tensor.tensors)
            return cls(
                tensors,
                dim_to_packs[0],
                pad_to_max_shape=pad_to_max_shapes[0],
                dedup_indices=dedup_indices,
            )
        # concatenate the tensors
        tensors = []
        for packed_tensor in from_packed_tensors:
            tensors.extend(packed_tensor.tensors)
        dim_to_pack = dim_to_packs[0]
        return cls(
            tensors,
            dim_to_pack,
            pad_to_max_shape=pad_to_max_shapes[0],
        )

    def deduplicate(self, prompt_indices: torch.Tensor | list[int]) -> "PackedTensor":
        """Share physical tensors for logical positions with the same prompt id."""
        indices = (
            prompt_indices.tolist()
            if isinstance(prompt_indices, torch.Tensor)
            else prompt_indices
        )
        assert len(indices) == len(self), (
            f"PackedTensor has {len(self)} logical entries but received "
            f"{len(indices)} prompt indices"
        )
        logical_tensors = (
            [self.tensors[index] for index in self._dedup_indices]
            if self._dedup_indices is not None
            else self.tensors
        )

        seen: dict[int, int] = {}
        unique_tensors: list[Optional[torch.Tensor]] = []
        dedup_indices: list[int] = []
        for tensor, prompt_index in zip(logical_tensors, indices, strict=True):
            prompt_index = int(prompt_index)
            if prompt_index not in seen:
                seen[prompt_index] = len(unique_tensors)
                unique_tensors.append(tensor)
            dedup_indices.append(seen[prompt_index])
        return PackedTensor(
            unique_tensors,
            self.dim_to_pack,
            pad_to_max_shape=self.pad_to_max_shape,
            dedup_indices=dedup_indices,
        )

    def repeat_interleave(self, num_repeats: int) -> "PackedTensor":
        """Repeat logical rows while sharing their physical tensors."""
        assert num_repeats >= 1, "num_repeats must be positive"
        source_indices = (
            self._dedup_indices
            if self._dedup_indices is not None
            else list(range(len(self.tensors)))
        )
        repeated_indices = [
            index for index in source_indices for _ in range(num_repeats)
        ]
        return PackedTensor(
            list(self.tensors),
            self.dim_to_pack,
            pad_to_max_shape=self.pad_to_max_shape,
            dedup_indices=repeated_indices,
        )

    @classmethod
    def flattened_concat(
        cls, from_packed_tensors: list["PackedTensor"]
    ) -> "PackedTensor":
        """Given a list of PackedTensor objects, flattens each PackedTensor and then concatenates them into a single PackedTensor.

        Each PackedTensor is first flattened by packing along the PackedTensor's `dim_to_pack` dimension. Then, the resulting flattened tensors are used to create a new PackedTensor.

        This is different from `PackedTensor.concat` which simply extends the underlying list of tensors. This is important because the `slice` and `__len__` methods operate on the underlying list of tensors. Note, however, that calling `as_tensor` on the resulting PackedTensor will result in the same tensor as `concat`.

        Each batch must have the same dim_to_pack.

        Example:
        ```{doctest}
        >>> import torch
        >>> from nemo_rl.data.multimodal_utils import PackedTensor
        >>> p1 = PackedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])], dim_to_pack=0)
        >>> p2 = PackedTensor([torch.tensor([7, 8, 9])], dim_to_pack=0)
        >>> p3 = PackedTensor.flattened_concat([p1, p2])
        >>> p3.tensors
        [tensor([1, 2, 3, 4, 5, 6]), tensor([7, 8, 9])]
        >>> p3.as_tensor()
        tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>>
        ```
        """
        dim_to_packs = [batch.dim_to_pack for batch in from_packed_tensors]
        assert len(set(dim_to_packs)) == 1, (
            "All packed tensors must have the same dim_to_pack"
        )
        pad_to_max_shapes = [batch.pad_to_max_shape for batch in from_packed_tensors]
        assert len(set(pad_to_max_shapes)) == 1, (
            "All packed tensors must have the same pad_to_max_shape setting"
        )
        tensors = [p.as_tensor() for p in from_packed_tensors]
        return cls(
            tensors,
            from_packed_tensors[0].dim_to_pack,
            pad_to_max_shape=pad_to_max_shapes[0],
        )


def get_multimodal_keys_from_processor(processor) -> list[str]:
    """Get keys of the multimodal data that can be used as model inputs.

    This will be used in the data_processor function to determine which keys to use as model inputs.
    """
    if isinstance(processor, PreTrainedTokenizerBase):
        return []

    all_keys = set()
    if hasattr(processor, "image_processor"):
        all_keys.update(processor.image_processor.model_input_names)
    if hasattr(processor, "video_processor"):
        all_keys.update(processor.video_processor.model_input_names)
    if hasattr(processor, "feature_extractor"):
        all_keys.update(processor.feature_extractor.model_input_names)
    all_keys.update(processor.model_input_names)
    all_keys.difference_update(set(processor.tokenizer.model_input_names))
    return list(all_keys)


def get_multimodal_default_settings_from_processor(
    processor,
) -> dict[str, dict[str, Any]]:
    if isinstance(processor, PreTrainedTokenizerBase):
        return {}

    default_settings = {}
    if hasattr(processor, "video_processor"):
        video_settings_dict = processor.video_processor.to_dict()
        if (
            "fps" in video_settings_dict
            and video_settings_dict["fps"] is None
            and "num_frames" in video_settings_dict
            and video_settings_dict["num_frames"] is None
            and "max_frames" in video_settings_dict
            and video_settings_dict["max_frames"] is not None
        ):
            video_settings_dict["num_frames"] = video_settings_dict["max_frames"]
        if not hasattr(
            get_multimodal_default_settings_from_processor, "load_video_kwargs"
        ):
            get_multimodal_default_settings_from_processor.load_video_kwargs = [
                param for param in inspect.signature(load_video).parameters
            ]
        default_settings["video"] = {
            arg: video_settings_dict[arg]
            for arg in get_multimodal_default_settings_from_processor.load_video_kwargs
            if arg in video_settings_dict
        }
    if hasattr(processor, "feature_extractor"):
        if not hasattr(
            get_multimodal_default_settings_from_processor, "load_audio_kwargs"
        ):
            get_multimodal_default_settings_from_processor.load_audio_kwargs = [
                param for param in inspect.signature(load_audio).parameters
            ]
        audio_settings_dict = processor.feature_extractor.to_dict()
        default_settings["audio"] = {
            arg: audio_settings_dict[arg]
            for arg in get_multimodal_default_settings_from_processor.load_audio_kwargs
            if arg in audio_settings_dict
        }
    return default_settings


def get_dim_to_pack_along(processor, key: str) -> int:
    """Special considerations for packing certain keys from certain processors.

    In most cases, the packed items are along dim 0
    """
    if processor.__class__.__name__ == "SmolVLMProcessor":
        return 1
    # return zero by default
    return 0


def extract_multimodal_model_inputs(
    processor: Any, processed: dict[str, Any]
) -> dict[str, PackedTensor | torch.Tensor]:
    """Extract packed visual inputs and sequence-aligned auxiliary tensors.

    Multimodal inputs declared by the processor are wrapped in ``PackedTensor``.
    Token-type fields remain ordinary tensors because they align with the full
    language-model token sequence.
    """
    input_ids = processed.get("input_ids")
    if input_ids is None:
        raise ValueError("Processor output is missing input_ids.")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim not in (1, 2):
        raise ValueError(
            "Processor input_ids must be a one- or two-dimensional torch.Tensor."
        )
    if input_ids.ndim == 2 and input_ids.shape[0] != 1:
        raise ValueError(
            "Multimodal chat processing expects a single conversation, got "
            f"input_ids shape {tuple(input_ids.shape)}."
        )
    sequence_length = input_ids.shape[-1]

    extracted: dict[str, PackedTensor | torch.Tensor] = {}
    multimodal_keys = list(get_multimodal_keys_from_processor(processor))
    # Some remote-code processors omit these media inputs from their declared
    # model_input_names even though their model forward requires them.
    for key in (
        "imgs_sizes",
        "num_frames",
        "pixel_values_flat",
        "image_num_patches",
    ):
        if key in processed and key not in multimodal_keys:
            multimodal_keys.append(key)
    for key in multimodal_keys:
        if key not in processed:
            continue
        value = processed[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Processor model input {key!r} must be a torch.Tensor, got "
                f"{type(value).__name__}."
            )
        if key == "imgs_sizes":
            value = value.to(dtype=torch.int32)
        extracted[key] = PackedTensor(
            value, dim_to_pack=get_dim_to_pack_along(processor, key)
        )

    for key in ("token_type_ids", "mm_token_type_ids"):
        if key not in processed:
            continue
        value = processed[key]
        if not isinstance(value, torch.Tensor) or value.ndim not in (1, 2):
            raise ValueError(
                f"Processor sequence input {key!r} must be a one- or "
                "two-dimensional torch.Tensor."
            )
        if value.ndim == 2:
            if value.shape[0] != 1:
                raise ValueError(
                    f"Processor sequence input {key!r} must contain one "
                    f"conversation, got shape {tuple(value.shape)}."
                )
            value = value[0]
        if len(value) != sequence_length:
            raise ValueError(
                f"Processor sequence input {key!r} has length {len(value)}, "
                f"but input_ids has length {sequence_length}."
            )
        extracted[key] = value
    return extracted


def resolve_to_image(image_path_or_image: str | Image.Image) -> Image.Image:
    """Resolve the image path to a PIL.Image object.

    image_path can be either:
    - path to local file
    - url to image
    - base64 encoded image
    """
    if isinstance(image_path_or_image, Image.Image):
        return image_path_or_image

    if image_path_or_image.startswith(("http://", "https://")):
        # Handle URL
        response = requests.get(image_path_or_image)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    elif image_path_or_image.startswith("data:"):
        # Handle base64 encoded image
        # Format: data:image/jpeg;base64,/9j/4AAQSkZJRg...
        header, encoded = image_path_or_image.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(BytesIO(image_data)).convert("RGB")
    elif image_path_or_image.startswith("file://"):
        return Image.open(image_path_or_image.removeprefix("file://")).convert("RGB")
    else:
        # Handle local file path
        return Image.open(image_path_or_image).convert("RGB")


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 ``data:`` URL.

    Args:
        image: PIL image to encode.
        fmt: PIL image format used for serialization (e.g. ``"PNG"``, ``"JPEG"``).
            The value is also lowercased and embedded in the MIME type of the
            returned URL.

    Returns:
        A ``data:image/<fmt>;base64,<payload>`` URL suitable for embedding in
        an OpenAI Responses ``input_image`` content part.
    """
    buf = BytesIO()
    image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def encode_images_in_examples(nemo_gym_examples: list[dict]) -> list[dict]:
    """Normalize local image and video references for vLLM HTTP requests.

    Walks each example's ``responses_create_params.input[].content[]`` items
    and rewrites local ``input_image`` references as base64 ``data:`` URLs and
    bare local video paths as ``file://`` URLs. Already-qualified HTTP(S), data,
    and file URLs are preserved. Malformed items are skipped without raising.

    The examples are mutated in place; the same list is also returned for
    convenience so callers can chain the call.

    Args:
        nemo_gym_examples: List of NeMo Gym example dicts. Each example is
            expected to contain a ``responses_create_params`` mapping with an
            ``input`` list of Responses API messages.

    Returns:
        The same list with local media references normalized in place.
    """
    for example in nemo_gym_examples:
        input_items = example.get("responses_create_params", {}).get("input", [])
        if not isinstance(input_items, list):
            continue
        for item in input_items:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "input_image":
                    media_key = "image_url"
                elif part_type in ("input_video", "video", "video_url"):
                    media_key = "video_url" if "video_url" in part else "video"
                else:
                    continue

                url = part.get(media_key, "")
                if isinstance(url, dict):
                    url = url.get("url", "")
                if not isinstance(url, str) or not url:
                    continue
                if url.startswith(("http://", "https://", "data:", "file://")):
                    continue
                if part_type == "input_image":
                    part[media_key] = image_to_data_url(resolve_to_image(url))
                else:
                    normalized_url = Path(url).expanduser().resolve().as_uri()
                    original_value = part.get(media_key)
                    if isinstance(original_value, dict):
                        original_value["url"] = normalized_url
                    else:
                        part[media_key] = normalized_url
    return nemo_gym_examples


def get_media_from_message(message: dict[str, Any]) -> dict[str, list[Any]]:
    """Get all media from a message log item."""
    # Handle None or missing content (e.g., assistant messages with only tool_calls)
    if message.get("content") is None:
        return {}
    # Handle string content (no images)
    if isinstance(message["content"], str):
        return {}
    # iterate over the content list
    media = defaultdict(list)
    for item in message["content"]:
        tag = item["type"]
        if tag in MEDIA_TAGS:
            media[tag].extend(list(item[tag])) if isinstance(
                item[tag], (list, tuple)
            ) else media[tag].append(item[tag])
    return media


def load_media_from_message(
    message: dict[str, Any],
    processor=None,
    multimodal_load_kwargs: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, list[Any]]:
    loaded_media = defaultdict(list)
    media_in_message = get_media_from_message(message)

    if multimodal_load_kwargs is None:
        multimodal_load_kwargs = {}

    if not multimodal_load_kwargs and processor is not None:
        multimodal_load_kwargs = get_multimodal_default_settings_from_processor(
            processor
        )

    if "image" in media_in_message:
        loaded_media["image"] += [
            resolve_to_image(img) for img in media_in_message["image"]
        ]
    if "audio" in media_in_message:
        for aud in media_in_message["audio"]:
            if isinstance(aud, str):
                if (
                    "audio" not in multimodal_load_kwargs
                    or "sampling_rate" not in multimodal_load_kwargs.get("audio", {})
                ):
                    raise ValueError(
                        "multimodal_load_kwargs must include 'audio' with a 'sampling_rate' "
                        "key to load audio from file path."
                    )
                try:
                    loaded_media["audio"].append(
                        load_audio(aud, **multimodal_load_kwargs["audio"])
                    )
                except (RuntimeError, FileNotFoundError, OSError) as e:
                    logger.warning("Audio loading failed. Falling back to torchaudio.")
                    import torchaudio

                    waveform, sr = torchaudio.load(aud)
                    target_sr = multimodal_load_kwargs["audio"]["sampling_rate"]
                    if sr != target_sr:
                        waveform = torchaudio.functional.resample(
                            waveform, sr, target_sr
                        )
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(0, keepdim=True)
                    loaded_media["audio"].append(
                        waveform.numpy()[get_dim_to_pack_along(processor, "audio")]
                    )
            else:
                loaded_media["audio"].append(aud)
    if "video" in media_in_message:
        for vid in media_in_message["video"]:
            if isinstance(vid, str):
                load_video_kwargs = (
                    multimodal_load_kwargs["video"]
                    if "video" in multimodal_load_kwargs
                    else {}
                )
                loaded_media["video"].append(
                    load_video(vid, backend="torchcodec", **load_video_kwargs)[0]
                )
            else:
                loaded_media["video"].append(vid)

    return loaded_media

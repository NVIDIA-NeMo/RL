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
from typing import Any, Optional, Protocol, Union

import decord
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


class MultimodalProcessorAdapter(Protocol):
    """Adapter for processor-specific multimodal chat preprocessing."""

    def process(
        self,
        processor: Any,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Render and tokenize a complete multimodal conversation."""


_MULTIMODAL_PROCESSOR_ADAPTERS: dict[str, MultimodalProcessorAdapter] = {}


def register_multimodal_processor_adapter(
    processor_class_name: str, adapter: MultimodalProcessorAdapter
) -> None:
    """Register preprocessing behavior for a nonstandard processor class."""
    if not processor_class_name:
        raise ValueError("processor_class_name must be non-empty")
    if not callable(getattr(adapter, "process", None)):
        raise TypeError("adapter must implement a callable process method")
    _MULTIMODAL_PROCESSOR_ADAPTERS[processor_class_name] = adapter


def _images_from_messages(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                images.append(resolve_to_image(part["image"]))
    return images


class _HuggingFaceMultimodalProcessorAdapter:
    """Adapter for processors supporting multimodal ``apply_chat_template``."""

    def process(
        self,
        processor: Any,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> tuple[str, dict[str, Any]]:
        formatted_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        processed = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        return formatted_text, dict(processed)


class _PlaceholderMultimodalProcessorAdapter:
    """Adapter for processors that expand image placeholder text in ``__call__``."""

    def process(
        self,
        processor: Any,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> tuple[str, dict[str, Any]]:
        images = _images_from_messages(messages)
        image_token = getattr(processor, "image_token", "<image>")
        placeholder_messages = []
        for message in messages:
            converted_message = dict(message)
            content = message.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.get("type") == "image":
                        text_parts.append(image_token)
                    elif part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    else:
                        raise ValueError(
                            "Placeholder-style image processors only support text "
                            f"and image content, got {part.get('type')!r}."
                        )
                # Empty separator matches vLLM's
                # `chat_template_content_format="string"` handler, which
                # concatenates multimodal parts (image_token + text) with no
                # inter-part separator. Using "\n" here inserted an extra
                # newline token per <image>/text boundary and made the
                # reconstructed trajectory diverge from what vLLM tokenized
                # during rollout (see examples/nemo_gym/debug_nano_omni_template.py).
                converted_message["content"] = "".join(text_parts)
            placeholder_messages.append(converted_message)

        template_messages = placeholder_messages
        if hasattr(processor, "conversation_preprocessor"):
            template_messages = [
                processor.conversation_preprocessor(message)
                for message in placeholder_messages
            ]
        formatted_text = processor.apply_chat_template(
            template_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        processed = processor(
            text=formatted_text,
            images=images,
            return_tensors="pt",
        )
        return formatted_text, dict(processed)


def process_multimodal_chat(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> tuple[str, dict[str, Any]]:
    """Render and process multimodal chat through a registered or HF adapter.

    Processors with a nonstandard multimodal calling convention must register an
    adapter. All other processors are expected to support Hugging Face's
    multimodal ``apply_chat_template`` interface.
    """
    adapter = _MULTIMODAL_PROCESSOR_ADAPTERS.get(
        type(processor).__name__, _HuggingFaceMultimodalProcessorAdapter()
    )
    formatted_text, processed = adapter.process(
        processor,
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    if "input_ids" not in processed:
        raise ValueError(
            f"{type(processor).__name__} did not return required input_ids."
        )

    images = _images_from_messages(messages)
    model_inputs = extract_multimodal_model_inputs(processor, processed)
    visual_keys = set(
        getattr(getattr(processor, "image_processor", None), "model_input_names", [])
    )
    visual_keys.update(
        key
        for key in get_multimodal_keys_from_processor(processor)
        if any(marker in key for marker in ("image", "img", "pixel", "aspect_ratio"))
    )
    visual_keys.add("imgs_sizes")
    if images and not any(key in model_inputs for key in visual_keys):
        raise ValueError(
            f"{type(processor).__name__} processed {len(images)} image(s) but "
            "returned no visual model inputs. Register a custom multimodal "
            "processor adapter if this processor does not support the standard "
            "Hugging Face multimodal chat-template interface."
        )
    return formatted_text, processed


for _processor_class_name in (
    "NemotronNanoVLV2Processor",
    "NemotronH_Nano_Omni_Reasoning_V3Processor",
):
    register_multimodal_processor_adapter(
        _processor_class_name, _PlaceholderMultimodalProcessorAdapter()
    )


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
    ) -> None:
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

    def as_tensor(
        self, device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        if device is not None:
            # Move only non-None tensors to device, preserve Nones
            for i, item in enumerate(self.tensors):
                if item is not None:
                    self.tensors[i] = item.to(device)
        non_none_tensors = [t for t in self.tensors if t is not None]
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
        # this is the number of tensors in this data wrapper
        return len(self.tensors)

    def to(self, device: str | torch.device) -> "PackedTensor":
        self.tensors = [
            item.to(device) if item is not None else None for item in self.tensors
        ]
        return self

    def slice(self, indices: Union[list[int], torch.Tensor]) -> "PackedTensor":
        idx = indices.tolist() if isinstance(indices, torch.Tensor) else indices
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
    # Some remote-code processors omit this per-image input from their declared
    # model_input_names even though their model forward requires it.
    if "imgs_sizes" in processed and "imgs_sizes" not in multimodal_keys:
        multimodal_keys.append("imgs_sizes")
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
    """Encode a PIL Image as a base64 data URL."""
    buf = BytesIO()
    image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def encode_images_in_examples(nemo_gym_examples: list[dict]) -> list[dict]:
    """Walk examples and replace local image paths with base64 data URLs.

    Operates in-place on each example's responses_create_params.input[].content[]
    items of type 'input_image'. HTTP(S) and data URLs are preserved; local
    paths, including file:// URLs, are encoded as data URLs.
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
                if not isinstance(part, dict) or part.get("type") != "input_image":
                    continue
                url = part.get("image_url", "")
                if isinstance(url, dict):
                    url = url.get("url", "")
                if not isinstance(url, str) or not url:
                    continue
                if url.startswith(("http://", "https://", "data:")):
                    continue
                part["image_url"] = image_to_data_url(resolve_to_image(url))
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
                    logger.warning("Audio loading failed. Fall back to decord.")
                    # use decord
                    loaded_audio = decord.AudioReader(
                        aud,
                        sample_rate=multimodal_load_kwargs["audio"]["sampling_rate"],
                        mono=True,
                    )
                    loaded_media["audio"].append(
                        loaded_audio[:].asnumpy()[
                            get_dim_to_pack_along(processor, "audio")
                        ]
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
                # seems decord backend loads video faster with multithread ffmpeg and it is easier to install
                loaded_media["video"].append(
                    load_video(vid, backend="decord", **load_video_kwargs)[0]
                )
            else:
                loaded_media["video"].append(vid)

    return loaded_media

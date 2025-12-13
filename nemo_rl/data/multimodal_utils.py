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

import re
import inspect
import decord
from PIL import Image
from collections import defaultdict
from typing import Any, Optional, Union

import torch
from transformers.audio_utils import load_audio
from transformers.video_utils import load_video
from transformers import PreTrainedTokenizerBase

load_audio_kwargs = [param for param in inspect.signature(load_audio).parameters]
load_video_kwargs = [param for param in inspect.signature(load_video).parameters]


# List of allowed placeholder strings for different media types in the dataset string
# e.g. "This is an example of <image>"
media_tags = {
    'image': '<image>',
    'video': '<video>',
    'audio': '<audio>',
    'video-audio': '<video-audio>',
}
media_tags_reversed = {v: k for k, v in media_tags.items()}

default_media_extensions = {
    'image': ['png','jpeg','jpg', 'img'],
    'video': ['mp4'],
    'video-audio': ['mp4'],
    'audio': ['wav', 'flac', "mp3"],
}


# different media namings maybe used in the raw dataset,
# in which case, they need to be mapped to the allowed ones
# WARNING: values cannot be used as the keys in the same dict to avoid cyclic graph
media_tags_to_allowed = {
    'speech': 'audio',
    'speeches': 'audio',
    'sound': 'audio',
    'audios': 'audio',
    'images': 'image',
    'videos': 'video',
}


# Build a pattern like: <image>|<video>|<audio>|<video-audio>
media_tag_pattern = re.compile(
    r"(" + "|".join(re.escape(tag) for tag in media_tags.values()) + ")"
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
        else:
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
        return PackedTensor(tensors, self.dim_to_pack)

    @classmethod
    def empty_like(cls, other: "PackedTensor") -> "PackedTensor":
        """Return a new PackedTensor with same length and dim_to_pack as `other`, with all entries None."""
        return cls([None] * len(other.tensors), other.dim_to_pack)

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
        # concatenate the tensors
        tensors = []
        for packed_tensor in from_packed_tensors:
            tensors.extend(packed_tensor.tensors)
        dim_to_pack = dim_to_packs[0]
        return cls(tensors, dim_to_pack)

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
        tensors = [p.as_tensor() for p in from_packed_tensors]
        return cls(tensors, from_packed_tensors[0].dim_to_pack)


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
    # all_keys.update(processor.model_input_names)
    all_keys.difference_update(set(processor.tokenizer.model_input_names))
    return list(all_keys)


def get_multimodal_default_settings_from_processor(processor) -> dict[str, dict[str, Any]]:
    if isinstance(processor, PreTrainedTokenizerBase):
        return []

    default_settings = {}
    if hasattr(processor, "video_processor"):
        video_settings_dict = processor.video_processor.to_dict()
        if "fps" in video_settings_dict and video_settings_dict["fps"] is None and \
            "num_frames" in video_settings_dict and video_settings_dict["num_frames"] is None and \
            "max_frames" in video_settings_dict and video_settings_dict["max_frames"] is not None:
            video_settings_dict["num_frames"] = video_settings_dict["max_frames"]
        default_settings["video"] = {arg: video_settings_dict[arg] for arg in load_video_kwargs if arg in video_settings_dict}
    if hasattr(processor, "feature_extractor"):
        audio_settings_dict = processor.feature_extractor.to_dict()
        default_settings["audio"] = {arg: audio_settings_dict[arg] for arg in load_audio_kwargs if arg in audio_settings_dict}
    return default_settings


def get_dim_to_pack_along(processor, key: str) -> int:
    """Special considerations for packing certain keys from certain processors.

    In most cases, the packed items are along dim 0
    """
    if processor.__class__.__name__ == "SmolVLMProcessor":
        return 1
    # return zero by default
    return 0


def get_media_from_message(message: dict[str, Any]) -> dict[str, list[Any]]:
    """Get all media from a message log item."""
    # Handle None or missing content (e.g., assistant messages with only tool_calls)
    if message.get("content") is None:
        return []
    # Handle string content (no images)
    if isinstance(message["content"], str):
        return []
    # iterate over the content list
    media = defaultdict(list)
    for item in message["content"]:
        tag = item["type"]
        if tag in media_tags:
            media[tag].extend(list(item[tag])) if isinstance(
                item[tag], (list, tuple)
            ) else media[tag].append(item[tag])
    return media


def load_media_from_message(
    message: dict[str, Any], 
    processor = None, 
    multimodal_load_kwargs: dict[str, dict[str, Any]] = {},
) -> dict[list[Any]]:
    loaded_media = defaultdict(list)
    media_in_message = get_media_from_message(message)

    if not multimodal_load_kwargs and processor is not None:
        multimodal_load_kwargs = get_multimodal_default_settings_from_processor(processor)

    if "image" in media_in_message:
        loaded_media["image"] += [Image.open(img) if isinstance(img, str) else img for img in media_in_message["image"]]
    if "audio" in media_in_message:
        for aud in media_in_message["audio"]:
            if isinstance(aud, str):
                print(multimodal_load_kwargs)
                assert "audio" in multimodal_load_kwargs and "sampling_rate"  in multimodal_load_kwargs["audio"]
                try:
                    loaded_media["audio"].append(load_audio(aud, **multimodal_load_kwargs["audio"]))
                except:
                    # use decord
                    loaded_audio = decord.AudioReader(aud, sample_rate=multimodal_load_kwargs["audio"]["sampling_rate"], mono=True)
                    loaded_media["audio"].append(loaded_audio[:].asnumpy()[get_dim_to_pack_along(tokenizer, key)])
            else:
                loaded_media["audio"].append(aud)
    if "video" in media_in_message:
        for vid in media_in_message["video"]:
            if isinstance(vid, str):
                load_video_kwargs = multimodal_load_kwargs["video"] if "video" in multimodal_load_kwargs else {}
                # seems decord backend loads video faster with multithread ffmpeg and it is easier to install
                loaded_media["video"].append(load_video(vid, backend="decord", **load_video_kwargs)[0])
            else:
                loaded_media["video"].append(vid)

    return loaded_media

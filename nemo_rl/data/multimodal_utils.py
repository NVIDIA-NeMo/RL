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

import base64
import inspect
import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from io import BytesIO
from typing import Any, Optional, Union

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedTokenizerBase
from transformers.audio_utils import load_audio
from transformers.video_utils import load_video

VLLM_MULTIMODAL_DATA_KEYS = frozenset({"vllm_images", "vllm_videos", "vllm_audios"})
NATIVE_MULTIMODAL_KEYS = frozenset({"vllm_content", *VLLM_MULTIMODAL_DATA_KEYS})
MULTIMODAL_CONTENT_TYPES = frozenset(
    {"input_image", "image", "image_url", "video", "audio"}
)

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
    """A logical batch of rows backed by packable tensor segments.

    The default representation is intentionally the legacy one: every entry in
    ``tensors`` is one logical row and no deduplication metadata is allocated.
    ``enable_deduplication`` adds stable provenance to the physical segments.
    Operations that combine or slice dedup-enabled values then use a CSR-like
    logical-row mapping:

    - ``_row_offsets`` partitions the flattened logical segment references.
    - ``_segment_indices`` maps each logical segment reference to ``tensors``.
    - ``_segment_provenance`` is stable across deepcopy/pickle and is the only
      evidence used to re-intern physical segments.

    Prompt identity is deliberately absent: belonging to the same prompt group
    makes media a candidate for sharing, but never proves media equality.

    Worked example. Prompt A has one image and prompt B has two, each already
    merged by :meth:`merge_segments` into a single logical row::

        A: tensors = [tA]        _row_offsets = [0, 1]  _segment_indices = [0]
        B: tensors = [tB0, tB1]  _row_offsets = [0, 2]  _segment_indices = [0, 1]

    GRPO then expands each prompt into G generations. The rows are deep-copied
    while ``_prepare_multimodal_sharing`` aliases the media leaves, so
    :meth:`concat` re-interns them by provenance. For G=3::

        tensors          = [tA, tB0, tB1]            # 3 physical segments
        _row_offsets     = [0, 1, 2, 3, 5, 7, 9]     # 6 logical rows
        _segment_indices = [0, 0, 0, 1, 2, 1, 2, 1, 2]
        len()            = 6

    Six logical rows over nine segment references and three physical tensors,
    so physical memory is flat in G. Row 4 reads as
    ``_segment_indices[_row_offsets[4]:_row_offsets[5]] == [1, 2]``, i.e. both
    of prompt B's images, without copying either.
    """

    def __init__(
        self,
        tensors: Union[torch.Tensor, list[Optional[torch.Tensor]], list[None]],
        dim_to_pack: int,
        *,
        pad_to_max_shape: bool = False,
        _row_offsets: Optional[list[int]] = None,
        _segment_indices: Optional[list[int]] = None,
        _segment_provenance: Optional[list[bytes]] = None,
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
            if not tensors and _row_offsets is None:
                raise AssertionError(
                    "Input tensors to PackedTensor must be a non-empty list"
                )
            self.tensors: list[Optional[torch.Tensor]] = tensors
        else:
            raise ValueError(
                f"Unsupported type for input tensors to PackedTensor: {type(tensors)}"
            )
        self.dim_to_pack = dim_to_pack
        self.pad_to_max_shape = pad_to_max_shape
        if (_row_offsets is None) != (_segment_indices is None):
            raise ValueError(
                "_row_offsets and _segment_indices must either both be set or both be None"
            )
        if _row_offsets is not None:
            if not _row_offsets or _row_offsets[0] != 0:
                raise ValueError("_row_offsets must start with 0")
            if any(
                current > following
                for current, following in zip(_row_offsets, _row_offsets[1:])
            ):
                raise ValueError("_row_offsets must be non-decreasing")
            assert _segment_indices is not None
            if _row_offsets[-1] != len(_segment_indices):
                raise ValueError(
                    "_row_offsets must end at the number of logical segment references"
                )
            if _segment_indices and (
                min(_segment_indices) < 0 or max(_segment_indices) >= len(self.tensors)
            ):
                raise ValueError(
                    "_segment_indices cannot reference an out-of-range physical segment"
                )
        if _segment_provenance is not None and len(_segment_provenance) != len(
            self.tensors
        ):
            raise ValueError(
                "_segment_provenance must have one entry per physical segment"
            )
        self._row_offsets = _row_offsets
        self._segment_indices = _segment_indices
        self._segment_provenance = _segment_provenance

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore both current and pre-deduplication pickled instances."""
        self.__dict__.update(state)
        self.__dict__.setdefault("_row_offsets", None)
        self.__dict__.setdefault("_segment_indices", None)
        self.__dict__.setdefault("_segment_provenance", None)

    @property
    def deduplication_enabled(self) -> bool:
        """Whether this value carries stable physical-segment provenance."""
        return self._segment_provenance is not None

    def logical_segment_counts_by_row(self) -> list[int]:
        """Return the number of non-empty media segments in each logical row."""
        if self._row_offsets is None:
            return [int(tensor is not None) for tensor in self.tensors]
        assert self._segment_indices is not None
        return [
            sum(
                self.tensors[physical_index] is not None
                for physical_index in self._segment_indices[
                    self._row_offsets[row] : self._row_offsets[row + 1]
                ]
            )
            for row in range(len(self))
        ]

    def iter_logical_segments(self):
        """Yield physical tensor segments in logical row/segment order."""
        if self._segment_indices is None:
            yield from self.tensors
            return
        for physical_index in self._segment_indices:
            yield self.tensors[physical_index]

    def enable_deduplication(self) -> "PackedTensor":
        """Assign stable provenance lazily without changing logical contents."""
        if self._segment_provenance is None:
            self._segment_provenance = [
                uuid.uuid4().bytes for _ in range(len(self.tensors))
            ]
        return self

    def _row_segment_indices(self, row: int) -> list[int]:
        if self._row_offsets is None:
            return [row]
        assert self._segment_indices is not None
        return self._segment_indices[
            self._row_offsets[row] : self._row_offsets[row + 1]
        ]

    def __deepcopy__(self, memo: dict[int, Any]) -> "PackedTensor":
        """Share immutable media segments only for an explicitly enabled value."""
        if self._row_offsets is None and not self.deduplication_enabled:
            copied = PackedTensor(
                [deepcopy(item, memo) for item in self.tensors],
                self.dim_to_pack,
                pad_to_max_shape=self.pad_to_max_shape,
            )
        else:
            copied = PackedTensor(
                (
                    list(self.tensors)
                    if self.deduplication_enabled
                    else [deepcopy(item, memo) for item in self.tensors]
                ),
                self.dim_to_pack,
                pad_to_max_shape=self.pad_to_max_shape,
                _row_offsets=(
                    list(self._row_offsets) if self._row_offsets is not None else None
                ),
                _segment_indices=(
                    list(self._segment_indices)
                    if self._segment_indices is not None
                    else None
                ),
                _segment_provenance=(
                    list(self._segment_provenance)
                    if self._segment_provenance is not None
                    else None
                ),
            )
        memo[id(self)] = copied
        return copied

    def as_tensor(
        self, device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        if device is not None:
            # Move only non-None tensors to device, preserve Nones
            for i, item in enumerate(self.tensors):
                if item is not None:
                    self.tensors[i] = item.to(device)
        tensors = self.tensors
        if self._segment_indices is not None:
            tensors = [self.tensors[index] for index in self._segment_indices]
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
        if self._row_offsets is not None:
            return len(self._row_offsets) - 1
        return len(self.tensors)

    def to(self, device: str | torch.device) -> "PackedTensor":
        """Move physical segments in place, retaining provenance.

        A device move is value-preserving, so provenance stays valid and two
        copies of the same segment still re-intern. This is the opposite of
        :meth:`to_dtype`, which changes values and therefore mints fresh
        provenance. The asymmetry is deliberate: :meth:`concat` re-interns on
        provenance alone, so a value-changing operation must not keep it.

        The corollary is that two values sharing provenance on different
        devices would merge to whichever appears first in ``concat``. That is
        currently unreachable -- ``BatchedDataDict.to`` and
        ``get_multimodal_dict`` only touch top-level values, while message-level
        segments are nested inside ``message_log`` and worker-side device moves
        happen post-serialization on independent copies -- and it would surface
        as a loud mixed-device ``torch.cat`` error rather than silent
        corruption. Keep it that way: do not move a subset of segments.
        """
        self.tensors = [
            item.to(device) if item is not None else None for item in self.tensors
        ]
        return self

    def to_dtype(self, dtype: torch.dtype) -> "PackedTensor":
        """Return an independent wrapper without expanding logical segments.

        Dtype conversion creates new physical tensor values, so deduplicated
        inputs receive new provenance. When the dtype already matches, immutable
        tensor segments and their provenance remain shared, but mutable wrapper
        state is copied. The logical row-to-segment mapping is preserved exactly.

        Non-floating-point segments are returned unchanged. Integer media
        metadata (grid sizes, frame counts) is index data, so casting it to a
        float dtype would silently corrupt it.
        """

        def converted(item: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if item is None or not item.is_floating_point():
                return item
            return item.to(dtype=dtype)

        requires_conversion = any(
            item is not None and item.is_floating_point() and item.dtype != dtype
            for item in self.tensors
        )

        return PackedTensor(
            (
                [converted(item) for item in self.tensors]
                if requires_conversion
                else list(self.tensors)
            ),
            self.dim_to_pack,
            pad_to_max_shape=self.pad_to_max_shape,
            _row_offsets=(
                list(self._row_offsets) if self._row_offsets is not None else None
            ),
            _segment_indices=(
                list(self._segment_indices)
                if self._segment_indices is not None
                else None
            ),
            _segment_provenance=(
                [uuid.uuid4().bytes for _ in self.tensors]
                if requires_conversion and self._segment_provenance is not None
                else (
                    list(self._segment_provenance)
                    if self._segment_provenance is not None
                    else None
                )
            ),
        )

    def slice(self, indices: Union[list[int], torch.Tensor]) -> "PackedTensor":
        idx = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        if not self.deduplication_enabled and self._row_offsets is None:
            tensors = [self.tensors[i] for i in idx]
            return PackedTensor(
                tensors,
                self.dim_to_pack,
                pad_to_max_shape=self.pad_to_max_shape,
            )

        physical_remap: dict[int, int] = {}
        tensors: list[Optional[torch.Tensor]] = []
        provenances: list[bytes] = []
        segment_indices: list[int] = []
        row_offsets = [0]
        for row in idx:
            if row < 0:
                row += len(self)
            if not 0 <= row < len(self):
                raise IndexError(f"PackedTensor row index {row} is out of range")
            for physical_index in self._row_segment_indices(row):
                if physical_index not in physical_remap:
                    physical_remap[physical_index] = len(tensors)
                    tensors.append(self.tensors[physical_index])
                    if self._segment_provenance is not None:
                        provenances.append(self._segment_provenance[physical_index])
                segment_indices.append(physical_remap[physical_index])
            row_offsets.append(len(segment_indices))
        return PackedTensor(
            tensors,
            self.dim_to_pack,
            pad_to_max_shape=self.pad_to_max_shape,
            _row_offsets=row_offsets,
            _segment_indices=segment_indices,
            _segment_provenance=(
                provenances if self._segment_provenance is not None else None
            ),
        )

    @classmethod
    def empty_like(cls, other: "PackedTensor") -> "PackedTensor":
        """Return empty logical rows matching ``other``."""
        return cls.empty_rows_like(other, len(other))

    @classmethod
    def empty_rows_like(cls, other: "PackedTensor", num_rows: int) -> "PackedTensor":
        """Return ``num_rows`` logical rows containing no media segments."""
        if num_rows < 0:
            raise ValueError("num_rows must be non-negative")
        if other.deduplication_enabled or other._row_offsets is not None:
            return cls(
                [],
                other.dim_to_pack,
                pad_to_max_shape=other.pad_to_max_shape,
                _row_offsets=[0] * (num_rows + 1),
                _segment_indices=[],
                _segment_provenance=[],
            )
        if num_rows == 0:
            return cls(
                [],
                other.dim_to_pack,
                pad_to_max_shape=other.pad_to_max_shape,
                _row_offsets=[0],
                _segment_indices=[],
                _segment_provenance=None,
            )
        return cls(
            [None] * num_rows,
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
        if any(
            packed_tensor.deduplication_enabled
            or packed_tensor._row_offsets is not None
            for packed_tensor in from_packed_tensors
        ):
            tensors: list[Optional[torch.Tensor]] = []
            provenances: list[bytes] = []
            provenance_to_physical: dict[bytes, int] = {}
            segment_indices: list[int] = []
            row_offsets = [0]

            for packed_tensor in from_packed_tensors:
                physical_remap: dict[int, int] = {}
                for physical_index, tensor in enumerate(packed_tensor.tensors):
                    provenance = (
                        packed_tensor._segment_provenance[physical_index]
                        if packed_tensor._segment_provenance is not None
                        else None
                    )
                    if provenance is not None and provenance in provenance_to_physical:
                        new_index = provenance_to_physical[provenance]
                    else:
                        new_index = len(tensors)
                        tensors.append(tensor)
                        if provenance is None:
                            provenance = uuid.uuid4().bytes
                        provenances.append(provenance)
                        provenance_to_physical[provenance] = new_index
                    physical_remap[physical_index] = new_index

                for row in range(len(packed_tensor)):
                    segment_indices.extend(
                        physical_remap[index]
                        for index in packed_tensor._row_segment_indices(row)
                    )
                    row_offsets.append(len(segment_indices))

            return cls(
                tensors,
                dim_to_packs[0],
                pad_to_max_shape=pad_to_max_shapes[0],
                _row_offsets=row_offsets,
                _segment_indices=segment_indices,
                _segment_provenance=provenances,
            )

        # Legacy flag-off behavior: concatenate the tensors without metadata.
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
    def merge_segments(
        cls, from_packed_tensors: list["PackedTensor"]
    ) -> "PackedTensor":
        """Merge message-turn values, collapsing compact inputs to one row.

        The legacy path retains one logical row per physical segment; its caller
        materializes those segments into one conversation row later.
        """
        if not any(
            packed_tensor.deduplication_enabled
            or packed_tensor._row_offsets is not None
            for packed_tensor in from_packed_tensors
        ):
            return cls.concat(from_packed_tensors)

        concatenated = cls.concat(from_packed_tensors)
        assert concatenated._segment_indices is not None
        return cls(
            concatenated.tensors,
            concatenated.dim_to_pack,
            pad_to_max_shape=concatenated.pad_to_max_shape,
            _row_offsets=[0, len(concatenated._segment_indices)],
            _segment_indices=concatenated._segment_indices,
            _segment_provenance=concatenated._segment_provenance,
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
        if any(
            packed_tensor.deduplication_enabled
            or packed_tensor._row_offsets is not None
            for packed_tensor in from_packed_tensors
        ):
            assert all(len(p) == 1 for p in from_packed_tensors), (
                "flattened_concat requires one logical row per input; "
                "merge_segments only collapses dedup-enabled values"
            )
            return cls.concat(from_packed_tensors)
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


def get_pad_to_max_shape(processor: Any, key: str) -> bool:
    """Return whether a processor input must pad non-packing dimensions."""
    return uses_image_placeholder(processor) and key == "pixel_values"


def extract_multimodal_model_inputs(
    processor: Any, processed: dict[str, Any]
) -> dict[str, PackedTensor | torch.Tensor]:
    """Extract packed media inputs and sequence-aligned auxiliary tensors."""
    processed = dict(processed)
    if (
        uses_image_placeholder(processor)
        and "pixel_values" in processed
        and "imgs_sizes" not in processed
        and processed["pixel_values"].ndim == 4
    ):
        pixel_values = processed["pixel_values"]
        num_tiles, _, height, width = pixel_values.shape
        processed["imgs_sizes"] = torch.tensor(
            [[height, width]] * num_tiles,
            dtype=torch.long,
        )
    if "imgs_sizes" in processed and "num_frames" not in processed:
        processed["num_frames"] = torch.ones(
            len(processed["imgs_sizes"]),
            dtype=torch.long,
        )

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
    for key in ("imgs_sizes", "num_frames"):
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
            value,
            dim_to_pack=get_dim_to_pack_along(processor, key),
            pad_to_max_shape=get_pad_to_max_shape(processor, key),
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


def extract_input_image_sources_from_responses_messages(
    messages: Any,
) -> list[str | Image.Image]:
    """Extract image sources from Responses-API messages in encounter order."""
    if not isinstance(messages, list):
        return []

    sources: list[str | Image.Image] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content") or []
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in ("input_image", "image", "image_url"):
                continue
            source = part.get("image") or part.get("image_url") or part.get("url")
            if isinstance(source, dict):
                source = source.get("url")
            if isinstance(source, (str, Image.Image)):
                sources.append(source)
    return sources


def extract_input_images_from_responses_messages(
    messages: Any,
) -> list[Image.Image]:
    """Load images from Responses-API input messages in encounter order."""
    return [
        resolve_to_image(source)
        for source in extract_input_image_sources_from_responses_messages(messages)
    ]


def _materialize_ragged_pixel_values(
    processed: dict[str, Any], processor: Any
) -> dict[str, Any]:
    """Fold a ragged per-image ``pixel_values`` list into one padded tensor.

    Processors with dynamic per-image resolution return a list of CHW tensors
    rather than a stacked batch. ``imgs_sizes`` is derived from the *unpadded*
    shapes first, since those exact sizes are what the projector slices with;
    padding happens afterwards so downstream sees the single tensor its
    torch.Tensor contract expects.
    """
    processed = dict(processed)
    pixel_values = processed.get("pixel_values")

    # pixel_values is only ragged when the images genuinely differ in size; two
    # images of equal resolution come back already stacked. Either way the rest
    # of the batch still needs restoring to tensors, so that runs below rather
    # than behind this branch.
    if isinstance(pixel_values, list):
        tiles = [torch.as_tensor(item) for item in pixel_values]
        if not tiles or any(item.ndim != 3 for item in tiles):
            raise ValueError(
                "Ragged pixel_values must contain one CHW tensor per image."
            )
        if len({item.shape[0] for item in tiles}) != 1:
            raise ValueError("Ragged pixel_values must use the same channel count.")
        _stack_ragged_pixel_values(processed, tiles, processor)

    _restore_tensors(processed)
    return processed


def _stack_ragged_pixel_values(
    processed: dict[str, Any], tiles: list[torch.Tensor], processor: Any
) -> None:
    """Derive imgs_sizes from unpadded shapes, then pad into one tensor."""
    if uses_image_placeholder(processor) and "imgs_sizes" not in processed:
        processed["imgs_sizes"] = torch.tensor(
            [[int(item.shape[-2]), int(item.shape[-1])] for item in tiles],
            dtype=torch.long,
        )
    stacked = PackedTensor(
        [item.unsqueeze(0) for item in tiles],
        dim_to_pack=0,
        pad_to_max_shape=True,
    ).as_tensor()
    assert stacked is not None
    processed["pixel_values"] = stacked


def _restore_tensors(processed: dict[str, Any]) -> None:
    """Convert a processor's list outputs back to tensors, in place.

    ``return_tensors=None`` makes the processor hand back *every* output as
    plain Python lists, not only the ragged ``pixel_values`` that mode was
    requested for. Downstream expects tensors -- ``input_ids`` in particular is
    rank-checked -- so restore the rest of the batch to what
    ``return_tensors="pt"`` would have produced. Values that resist conversion
    (genuinely ragged per-image metadata) are left for their own handling.
    """
    for key, value in processed.items():
        if key == "pixel_values" or isinstance(value, torch.Tensor):
            continue
        if isinstance(value, (list, tuple)):
            try:
                processed[key] = torch.as_tensor(value)
            except (TypeError, ValueError, RuntimeError):
                continue


def attach_image_model_inputs_to_message(
    message: dict[str, Any],
    *,
    images: list[Image.Image],
    processor: Any,
    pad_dynamic_image_shapes: bool = False,
) -> None:
    """Attach processor-owned image tensors without replacing rollout tokens."""
    if not images or processor is None:
        return

    image_token = getattr(processor, "image_token", "<image>")
    # Processors that emit dynamic per-image resolutions return a ragged CHW list
    # for heterogeneous multi-image turns. Asking BatchFeature for PT tensors
    # would make it stack those and fail before the exact imgs_sizes are read off
    # them. Off by default, so every other caller keeps the stacked path.
    allow_ragged_output = pad_dynamic_image_shapes and len(images) > 1
    processed = processor(
        text=image_token * len(images),
        images=images,
        return_tensors=None if allow_ragged_output else "pt",
    )
    processed = dict(processed)
    if allow_ragged_output:
        processed = _materialize_ragged_pixel_values(processed, processor)
    model_inputs = extract_multimodal_model_inputs(processor, processed)
    message.update(
        {
            key: value
            for key, value in model_inputs.items()
            if isinstance(value, PackedTensor)
        }
    )


def encode_images_in_examples(nemo_gym_examples: list[dict]) -> list[dict]:
    """Replace local image paths in NeMo Gym examples with base64 data URLs.

    Walks each example's ``responses_create_params.input[].content[]`` items
    and rewrites any ``input_image`` part whose ``image_url`` is a local path
    (or ``file://`` URL) into a base64 ``data:`` URL via
    :func:`image_to_data_url`. Parts whose URL already starts with ``http://``,
    ``https://``, or ``data:`` are left untouched. Malformed items (non-dict
    entries, missing/empty URLs, non-list ``input``/``content``) are skipped
    without raising.

    The examples are mutated in place; the same list is also returned for
    convenience so callers can chain the call.

    Args:
        nemo_gym_examples: List of NeMo Gym example dicts. Each example is
            expected to contain a ``responses_create_params`` mapping with an
            ``input`` list of Responses API messages.

    Returns:
        The same ``nemo_gym_examples`` list, with local image references
        rewritten to base64 data URLs in place.
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


def build_media_token_validity_mask(
    input_ids: torch.Tensor,
    media_token_id: int,
    media_counts_by_row: Sequence[int],
    base_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Mark media-token positions in rows that carry no media of that modality.

    A media token is an ordinary vocabulary entry with its own embedding row.
    It only means "a projected feature belongs here" when media is attached;
    in a text-only row the same id is whatever the author wrote, and counting
    it as a placeholder makes the model demand a feature that does not exist.

    Rows that do carry media keep every position valid, so a real
    placeholder/feature disagreement there is still reported rather than
    silently masked away.

    Args:
        input_ids: ``[B, S]`` token ids, one row per sample.
        media_token_id: Vocabulary id the model treats as a media placeholder.
        media_counts_by_row: Attached media items per row, e.g. from
            :meth:`PackedTensor.logical_segment_counts_by_row`.
        base_mask: Optional ``[B, S]`` validity mask to refine, so masks for
            several modalities can be combined.

    Returns:
        A ``[B, S]`` bool mask, or ``None`` when nothing needs masking and the
        caller should leave the model's own derivation untouched.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B, S], got {tuple(input_ids.shape)}")
    if len(media_counts_by_row) != input_ids.shape[0]:
        raise ValueError(
            "media_counts_by_row must have one entry per row: got "
            f"{len(media_counts_by_row)} for {input_ids.shape[0]} rows"
        )

    empty_rows = [row for row, count in enumerate(media_counts_by_row) if count == 0]
    if not empty_rows:
        return base_mask

    is_media_token = input_ids == media_token_id
    if not bool(is_media_token[empty_rows].any()):
        # Text-only rows exist but none of them spell the token, so the
        # model's own derivation is already correct.
        return base_mask

    mask = (
        torch.ones_like(input_ids, dtype=torch.bool)
        if base_mask is None
        else base_mask.clone()
    )
    for row in empty_rows:
        mask[row] &= ~is_media_token[row]
    return mask


def media_placeholder_token_id_from_chunks(chunks: Sequence[Any]) -> Optional[int]:
    """The vocabulary id these model chunks treat as a media placeholder, if any."""
    for chunk in chunks:
        token_id = getattr(chunk, "image_token_index", None)
        if token_id is not None:
            return int(token_id)
    return None


def chunks_accept_media_token_validity_mask(chunks: Sequence[Any]) -> bool:
    """Whether a model chunk's forward takes an explicit media-token validity mask.

    Checked against the signature rather than a class flag so a model that does
    not know about the mask never receives it: such a forward would absorb it
    into ``**kwargs`` and silently ignore it, which looks identical to the mask
    having been applied. Failing to send it is visible; sending it into a void
    is not.
    """
    for chunk in chunks:
        try:
            parameters = inspect.signature(chunk.forward).parameters
        except (TypeError, ValueError):
            continue
        if "media_token_validity_mask" in parameters:
            return True
    return False


def image_counts_by_row(batch: Any, num_rows: int) -> Optional[list[int]]:
    """How many images each row of the batch actually carries.

    Returns None when the batch describes its images in a way this cannot read,
    so the caller leaves the model's own derivation alone rather than guessing a
    count and masking against it.
    """
    pixel_values = batch.get("pixel_values", None)
    if pixel_values is None:
        # A text-only batch genuinely has no images anywhere, which is exactly
        # the case the mask exists for -- not a missing-data case.
        return [0] * num_rows
    if not isinstance(pixel_values, PackedTensor):
        return None
    counts = pixel_values.logical_segment_counts_by_row()
    return counts if len(counts) == num_rows else None


def attach_media_token_validity_mask(batch: Any, media_token_id: Optional[int]) -> None:
    """Mark media tokens that anchor nothing, so the model keeps their embedding.

    Builds the mask while rows still are samples. Sequence packing later
    concatenates those rows into one THD sequence, after which no per-row
    question can be asked, so the packing step carries this through the same
    transform as ``input_ids`` rather than deriving it downstream.

    The batch is duck-typed rather than annotated as ``BatchedDataDict``:
    that module imports this one, so naming it here would be circular.
    """
    if media_token_id is None:
        return
    input_ids = batch.get("input_ids", None)
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return
    counts = image_counts_by_row(batch, input_ids.shape[0])
    if counts is None:
        return
    mask = build_media_token_validity_mask(input_ids, media_token_id, counts)
    if mask is not None:
        batch["media_token_validity_mask"] = mask

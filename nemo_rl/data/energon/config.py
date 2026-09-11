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

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from nemo_rl.data.energon.multimodal.model_families import ModelFamily


class EnergonSourceConfig(BaseModel, extra="allow"):
    """One prepared Energon dataset split."""

    path: str
    split: str
    virtual_epoch_length: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1)] | None = None


class EnergonPackingOptions(BaseModel, extra="forbid"):
    """Options for task-encoder-owned sequence packing."""

    max_sequence_length: Annotated[int, Field(ge=1)]
    sequence_length_pad_multiple: Annotated[int, Field(ge=1)]
    balanced_knapsack_delta: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def _validate_alignment(self) -> "EnergonPackingOptions":
        if self.max_sequence_length % self.sequence_length_pad_multiple:
            raise ValueError(
                "Energon pack capacity must be divisible by its padding multiple."
            )
        return self


class EnergonPackingConfig(BaseModel, extra="allow"):
    """One task-encoder-owned packing implementation."""

    name: str
    buffer_size: Annotated[int, Field(ge=1)]
    options: EnergonPackingOptions


class EnergonTaskEncoderOptions(BaseModel, extra="forbid"):
    """Typed settings used by the Nemotron multimodal task encoder."""

    patch_dim: Annotated[int, Field(ge=1)] = 16
    temporal_patch_size: Annotated[int, Field(ge=1)] = 2
    prompt_format: Literal["nemotron-h-5p5-reasoning", "nemotron6-moe"] = (
        "nemotron-h-5p5-reasoning"
    )
    thinking_trace_format: Literal["default", "normalized", "ultra"] = "normalized"
    relax_thinking_trace_check: bool = False
    packing_sequence_length: Annotated[int, Field(ge=1)] | None = None
    video_min_num_frames: Annotated[int, Field(ge=1)] = 8
    video_max_num_frames: Annotated[int, Field(ge=1)] = 32
    video_default_fps: Annotated[int, Field(ge=1)] = 2
    video_frame_temporal_jitter: bool = False
    video_aug_scale_frames_up: Annotated[int, Field(ge=1)] | None = None
    video_aug_scale_resolution_up: Annotated[int, Field(ge=1)] | None = None
    video_aug_scale_resolution_only: bool = False
    tiling_augment_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.4
    allow_large_videos: bool = False
    video_decode_thread_count: Annotated[int, Field(ge=0)] = 8
    audio_subsampling_factor: Annotated[int, Field(ge=1)] | None = None
    audio_num_mel_bins: Annotated[int, Field(ge=1)] = 128
    audio_clip_duration_seconds: Annotated[float, Field(gt=0)] = 30.0
    min_audio_duration_seconds: Annotated[float, Field(gt=0)] = 0.1
    max_audio_duration_seconds: Annotated[float, Field(gt=0)] = 1800.0

    @model_validator(mode="after")
    def _validate_audio_settings(self) -> "EnergonTaskEncoderOptions":
        if (
            self.audio_subsampling_factor is not None
            and self.audio_subsampling_factor & (self.audio_subsampling_factor - 1)
        ):
            raise ValueError("audio_subsampling_factor must be a power of two.")
        if self.min_audio_duration_seconds > self.audio_clip_duration_seconds:
            raise ValueError(
                "min_audio_duration_seconds must not exceed "
                "audio_clip_duration_seconds."
            )
        if self.max_audio_duration_seconds < self.audio_clip_duration_seconds:
            raise ValueError(
                "max_audio_duration_seconds must not be smaller than "
                "audio_clip_duration_seconds."
            )
        return self


class EnergonTaskEncoderConfig(BaseModel, extra="allow"):
    """One task encoder and its optional packing implementation."""

    name: str = "generic_sft"
    options: EnergonTaskEncoderOptions = Field(
        default_factory=EnergonTaskEncoderOptions
    )
    packing: EnergonPackingConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_registry_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value


class EnergonCookerConfig(BaseModel, extra="allow"):
    """One source cooker selected by registry key."""

    name: str = "generic_conversation"
    options: dict[str, Any] = Field(default_factory=dict)
    has_subflavors: dict[str, str | int | float | bool | None] | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_registry_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value


class EnergonLoaderConfig(BaseModel, extra="allow"):
    """Shared Energon settings for driver- and worker-owned SFT loaders."""

    model_family: ModelFamily = Field(
        description="Model family used to validate cooker and task-encoder support."
    )
    num_workers: Annotated[int, Field(ge=0)] = 8
    shuffle_buffer_size: Annotated[int, Field(ge=0)] = 1000
    max_samples_per_sequence: (
        Annotated[
            int,
            Field(
                ge=1,
                description="Maximum sequential sample run used when sharding a dataset.",
            ),
        ]
        | None
    ) = None
    # Packing is configured by task_encoder.packing. Keep the old field in the
    # resolved config so older recipes that set it to null remain loadable.
    packing_buffer_size: None = None
    batch_grouping: Literal["auto"] = "auto"
    processor_adapter: Literal["hf_multimodal"] = "hf_multimodal"
    topology_mapper: Literal["default"] = "default"
    task_encoder: EnergonTaskEncoderConfig = Field(
        default_factory=EnergonTaskEncoderConfig
    )
    cookers: list[EnergonCookerConfig] = Field(
        default_factory=lambda: [EnergonCookerConfig()]
    )
    seed_offset: int = 0
    prefetch_factor: Annotated[int, Field(ge=1)] = 2
    checkpoint_every_sec: Annotated[float, Field(gt=0)] = 60.0
    watchdog_timeout_seconds: Annotated[float, Field(gt=0)] | None = 60.0
    nvdataset_cache_dir: str | None = Field(
        default=None,
        description=(
            "Root used to resolve dss:// dataset paths. When set, this is "
            "exported as NVDATASET_CACHE_DIR for Energon loader workers."
        ),
    )
    cache_pool_max_gbytes: Annotated[float, Field(gt=0)] | None = Field(
        default=None,
        description="Maximum file-store cache size in GiB; None uses Energon's limit.",
    )
    cache_pool_num_workers: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Number of FileStoreCachePool worker processes.",
    )
    gc_collect_every_n_steps: Annotated[int, Field(ge=1)] = Field(
        default=100000,
        description="Loader steps between forced Python garbage collections.",
    )

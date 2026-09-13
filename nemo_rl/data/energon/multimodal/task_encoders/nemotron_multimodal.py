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
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from megatron.energon import edataclass, stateless

from nemo_rl.data.energon.multimodal.model_families import supports_model_families
from nemo_rl.data.energon.multimodal.task_encoders.base import SFTCooker
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    GenericSFTTaskEncoder,
    HFMultimodalSFTProcessorAdapter,
    SFTProcessorAdapter,
    _normalize_messages,
)
from nemo_rl.data.energon.multimodal.task_encoders.media import (
    decode_selected_av_bytes,
)
from nemo_rl.data.energon.multimodal.task_encoders.nemotron_tokenization import (
    validate_text_content,
)
from nemo_rl.data.energon.multimodal.task_encoders.nemotron_visual import (
    COMPACT_IMAGE_PLACEHOLDER,
    NemotronEncodedSFTSample,
    _NemotronVisualProcessorAdapter,
    _expand_visual_placeholders,
    _normalize_assistant_thinking,
    _token_id,
    _tokenize_nemotron_sample,
    _truncate_message_log,
    _video_prompt,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    EncodedSFTSample,
    MediaRef,
)
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data.packing import SequencePacker

SOUND_TOKEN = SOUND_PLACEHOLDER = "<so_embedding>"
SOUND_START = "<so_start>"
SOUND_END = "<so_end>"

# Audio frame and subsampling math follows Megatron-Bridge revision
# 8c46dc4259080c510b7455f43e836fdff222c5d3,
# models/nemotron_omni/nemotron_omni_utils.py. Per-clip sizing follows the
# Energon reference at revision 6822175d92a40e0528be905aee50f5930cfa0c98,
# examples/multimodal/data_loading/audio_processing.py.


@dataclass(frozen=True)
class _AudioPlan:
    media_index: int
    num_embeddings: int
    num_clips: int
    audio_length: int
    timestamps: tuple[float, float]
    samples_per_clip: tuple[int, ...]
    valid_frame_counts: tuple[int, ...]
    embedding_counts: tuple[int, ...]
    source_sampling_rate: int

    @property
    def total_embeddings(self) -> int:
        return self.num_embeddings


@edataclass
class NemotronOmniEncodedSFTSample(NemotronEncodedSFTSample):
    """Encoded Omni sample with the exact audio plan from pre-encoding."""

    audio_plans: tuple[_AudioPlan, ...]


def _required_metadata_int(ref: MediaRef, key: str) -> int:
    value = dict(ref.metadata).get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"Nemotron audio media requires positive integer metadata {key!r}; "
            f"got {value!r}. Exact lazy audio sizing requires immutable source "
            "length and sample-rate metadata."
        )
    return value


def _source_sampling_rate(ref: MediaRef) -> int:
    return _required_metadata_int(ref, "audio_sample_rate")


def _audio_duration(ref: MediaRef) -> float:
    metadata = dict(ref.metadata)
    duration = metadata.get("audio_duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError(
            "Nemotron audio media requires numeric metadata 'audio_duration'; "
            f"got {duration!r}."
        )
    return float(duration)


def _decoded_audio_clips(ref: MediaRef) -> tuple[torch.Tensor, ...]:
    """Load audio in the same channels-first form used by the reference."""
    payload = ref.value.get() if callable(getattr(ref.value, "get", None)) else ref.value
    from_soundfile = False
    if isinstance(payload, str):
        import soundfile as sf

        payload, sampling_rate = sf.read(
            payload,
            dtype="float32",
            always_2d=True,
        )
        expected_sampling_rate = _source_sampling_rate(ref)
        if sampling_rate != expected_sampling_rate:
            raise ValueError(
                f"Nemotron audio file sampling rate {sampling_rate} does not match "
                f"audio_sample_rate={expected_sampling_rate}."
            )
        from_soundfile = True
    if isinstance(payload, (bytes, bytearray, memoryview)):
        payload = decode_selected_av_bytes(payload, modality="audio")
    if callable(getattr(payload, "get_audio", None)):
        payload = payload.get_audio().audio_clips

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        values = list(payload)
    else:
        values = [payload]
    if not values:
        raise ValueError("Nemotron audio payload has no clips.")

    expected_channels = dict(ref.metadata).get("audio_channels")
    clips: list[torch.Tensor] = []
    for value in values:
        clip = torch.as_tensor(value)
        if clip.ndim == 1:
            clip = clip.unsqueeze(0)
        elif clip.ndim != 2:
            raise ValueError(
                "Nemotron audio clips must have shape [channels, samples]; "
                f"got {tuple(clip.shape)}."
            )
        if from_soundfile:
            clip = clip.transpose(0, 1)
        elif (
            type(expected_channels) is int
            and clip.shape[0] != expected_channels
            and clip.shape[1] == expected_channels
        ):
            clip = clip.transpose(0, 1)
        clips.append(clip)
    return tuple(clips)


def _normalized_mono_audio(ref: MediaRef) -> torch.Tensor:
    clips = _decoded_audio_clips(ref)
    try:
        audio = torch.stack(clips, dim=0)
    except RuntimeError as error:
        raise ValueError(
            "Nemotron audio decoder returned clips with different sample lengths."
        ) from error

    if audio.dtype == torch.int16:
        audio = audio.to(torch.float32) / 32768.0
    elif audio.dtype == torch.int32:
        audio = audio.to(torch.float32) / 2147483648.0
    else:
        audio = audio.to(torch.float32)
    max_value = audio.abs().max()
    if max_value > 1.0:
        audio = audio / max_value
    return audio.mean(dim=1, keepdim=True)


def _subsampled_length(frame_count: int, subsampling_factor: int) -> int:
    length = frame_count
    for _ in range(int(math.log2(subsampling_factor))):
        # Parakeet uses kernel_size=3, stride=2, and padding=1.
        length = (length + 1) // 2
    return max(1, length)

def _audio_plan(
    ref: MediaRef,
    *,
    media_index: int,
    target_sampling_rate: int,
    hop_length: int,
    subsampling_factor: int,
    clip_duration_seconds: float,
    min_duration_seconds: float,
    max_duration_seconds: float,
) -> _AudioPlan:
    metadata_duration = _audio_duration(ref)
    if metadata_duration < min_duration_seconds:
        raise ValueError(
            f"Nemotron audio duration {metadata_duration:.3f}s is below "
            f"min_audio_duration_seconds={min_duration_seconds}."
        )
    if metadata_duration > max_duration_seconds:
        raise ValueError(
            f"Nemotron audio duration {metadata_duration:.3f}s exceeds "
            f"max_audio_duration_seconds={max_duration_seconds}."
        )

    source_sampling_rate = _source_sampling_rate(ref)
    source_audio = _decoded_audio_clips(ref)
    source_samples = source_audio[0].shape[-1]
    source_duration = max(min_duration_seconds, source_samples / source_sampling_rate)
    target_samples = round(source_duration * target_sampling_rate)
    min_samples = round(min_duration_seconds * target_sampling_rate)
    clip_samples = round(clip_duration_seconds * target_sampling_rate)
    num_clips = math.ceil(target_samples / clip_samples)
    remainder = target_samples % clip_samples
    last_clip_samples = clip_samples if remainder == 0 else max(remainder, min_samples)
    samples_per_clip = (clip_samples,) * (num_clips - 1) + (last_clip_samples,)
    audio_length = sum(samples_per_clip)
    if audio_length > target_samples:
        source_duration = audio_length / target_sampling_rate
    valid_frame_counts = tuple(count // hop_length for count in samples_per_clip)
    if any(count <= 0 for count in valid_frame_counts):
        raise ValueError(
            "Nemotron audio clip produces no mel frames. Increase "
            "min_audio_duration_seconds or reduce the feature-extractor hop length."
        )
    embedding_counts = tuple(
        _subsampled_length(count, subsampling_factor) for count in valid_frame_counts
    )
    return _AudioPlan(
        media_index=media_index,
        num_embeddings=sum(embedding_counts),
        num_clips=num_clips,
        audio_length=audio_length,
        timestamps=(0.0, source_duration),
        samples_per_clip=samples_per_clip,
        valid_frame_counts=valid_frame_counts,
        embedding_counts=embedding_counts,
        source_sampling_rate=source_sampling_rate,
    )


def _render_omni_messages(
    sample: CanonicalSFTSample,
    *,
    temporal_patch_size: int,
    prompt_format: str,
    thinking_trace_format: str,
    relax_thinking_trace_check: bool,
    video_timestamps: dict[int, tuple[float, ...]] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, MediaRef]]]:
    # materialize=False: every media part below is replaced by text built from
    # metadata, and message["content"] is overwritten wholesale, so decoding
    # the payload here would be discarded immediately.
    messages = _normalize_messages(sample, materialize=False)
    occurrences: list[tuple[int, MediaRef]] = []
    media_index = 0
    for message_index, message in enumerate(messages):
        _normalize_assistant_thinking(
            message,
            sample=sample,
            message_index=message_index,
            prompt_format=prompt_format,
            thinking_trace_format=thinking_trace_format,
            relax_thinking_trace_check=relax_thinking_trace_check,
        )
        rendered_content: list[dict[str, str]] = []
        for part in message["content"]:
            part_type = part["type"]
            if part_type == "text":
                text = str(part.get("text", ""))
                validate_text_content(text, sample_key=sample.__key__)
                rendered_content.append({"type": "text", "text": text})
                continue
            if part_type not in {"audio", "image", "video", "video_frame"}:
                raise ValueError(f"Nemotron Omni does not support {part_type!r} media.")
            if media_index >= len(sample.media):
                raise ValueError(
                    "Nemotron Omni message media exceeds the canonical media list."
                )
            ref = sample.media[media_index]
            if ref.modality != part_type:
                raise ValueError(
                    f"Nemotron Omni media order mismatch: expected {part_type!r}, "
                    f"got {ref.modality!r}."
                )
            if part_type in {"image", "video_frame"}:
                text = COMPACT_IMAGE_PLACEHOLDER
            elif part_type == "video":
                text, _ = _video_prompt(
                    ref,
                    temporal_patch_size=temporal_patch_size,
                    frame_timestamps=(video_timestamps or {}).get(media_index),
                )
            else:
                text = f"{SOUND_START}{SOUND_PLACEHOLDER}{SOUND_END}"
            rendered_content.append({"type": "text", "text": text})
            occurrences.append((message_index, ref))
            media_index += 1
        message["content"] = rendered_content
    if media_index != len(sample.media):
        raise ValueError(
            f"Nemotron Omni messages reference {media_index}/{len(sample.media)} "
            "media items."
        )
    return messages, occurrences


def _expand_audio_placeholders(
    message_log: list[dict[str, Any]],
    occurrences: Sequence[tuple[int, MediaRef, _AudioPlan]],
    *,
    sound_token_id: int,
) -> None:
    plans_by_message: defaultdict[int, list[_AudioPlan]] = defaultdict(list)
    for message_index, _, plan in occurrences:
        plans_by_message[message_index].append(plan)

    for message_index, plans in plans_by_message.items():
        token_ids = message_log[message_index].get("token_ids")
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise ValueError(
                "Nemotron Omni tokenized messages require one-dimensional token_ids."
            )
        token_loss_mask = message_log[message_index].get("token_loss_mask")
        if (
            not isinstance(token_loss_mask, torch.Tensor)
            or token_loss_mask.ndim != 1
            or token_loss_mask.shape != token_ids.shape
        ):
            raise ValueError(
                "Nemotron Omni tokenized messages require token_loss_mask aligned "
                "with token_ids."
            )
        placeholder_positions = torch.where(token_ids == sound_token_id)[0].tolist()
        if len(placeholder_positions) != len(plans):
            raise ValueError(
                f"Nemotron Omni message {message_index} has "
                f"{len(placeholder_positions)} compact sound placeholders for "
                f"{len(plans)} audio items."
            )
        token_pieces: list[torch.Tensor] = []
        mask_pieces: list[torch.Tensor] = []
        start = 0
        for position, plan in zip(placeholder_positions, plans, strict=True):
            token_pieces.append(token_ids[start:position])
            token_pieces.append(
                torch.full(
                    (plan.total_embeddings,),
                    sound_token_id,
                    dtype=token_ids.dtype,
                    device=token_ids.device,
                )
            )
            mask_pieces.append(token_loss_mask[start:position])
            mask_pieces.append(
                token_loss_mask[position].expand(plan.total_embeddings)
            )
            start = position + 1
        token_pieces.append(token_ids[start:])
        mask_pieces.append(token_loss_mask[start:])
        message_log[message_index]["token_ids"] = torch.cat(token_pieces)
        message_log[message_index]["token_loss_mask"] = torch.cat(mask_pieces)


def _feature_output(features: Any) -> Mapping[str, Any]:
    if isinstance(features, Mapping):
        return features
    data = getattr(features, "data", None)
    if isinstance(data, Mapping):
        return data
    raise TypeError("Nemotron audio feature extractor must return a mapping.")


class NemotronMultiModalProcessorAdapter(_NemotronVisualProcessorAdapter):
    """Predict Nemotron image, video, and optional sound widths lazily."""

    def __init__(
        self,
        *,
        processor: Any,
        max_sequence_length: int,
        packing_sequence_length: int | None = None,
        patch_dim: int = 16,
        temporal_patch_size: int = 2,
        prompt_format: str = "nemotron-h-5p5-reasoning",
        thinking_trace_format: str = "normalized",
        relax_thinking_trace_check: bool = False,
        video_min_num_frames: int = 8,
        video_max_num_frames: int = 32,
        video_default_fps: int = 2,
        video_frame_temporal_jitter: bool = False,
        video_aug_scale_frames_up: int | None = None,
        video_aug_scale_resolution_up: int | None = None,
        video_aug_scale_resolution_only: bool = False,
        allow_large_videos: bool = False,
        tiling_augment_prob: float = 0.4,
        video_decode_thread_count: int = 8,
        audio_subsampling_factor: int | None = None,
        audio_num_mel_bins: int = 128,
        audio_clip_duration_seconds: float = 30.0,
        min_audio_duration_seconds: float = 0.1,
        max_audio_duration_seconds: float = 1800.0,
        add_bos: bool = False,
        add_eos: bool = False,
        add_generation_prompt: bool = False,
    ) -> None:
        super().__init__(
            processor=processor,
            max_sequence_length=max_sequence_length,
            packing_sequence_length=packing_sequence_length,
            patch_dim=patch_dim,
            temporal_patch_size=temporal_patch_size,
            prompt_format=prompt_format,
            thinking_trace_format=thinking_trace_format,
            relax_thinking_trace_check=relax_thinking_trace_check,
            video_min_num_frames=video_min_num_frames,
            video_max_num_frames=video_max_num_frames,
            video_default_fps=video_default_fps,
            video_frame_temporal_jitter=video_frame_temporal_jitter,
            video_aug_scale_frames_up=video_aug_scale_frames_up,
            video_aug_scale_resolution_up=video_aug_scale_resolution_up,
            video_aug_scale_resolution_only=video_aug_scale_resolution_only,
            allow_large_videos=allow_large_videos,
            tiling_augment_prob=tiling_augment_prob,
            video_decode_thread_count=video_decode_thread_count,
            add_bos=add_bos,
            add_eos=add_eos,
            add_generation_prompt=add_generation_prompt,
        )
        feature_extractor = getattr(processor, "feature_extractor", None)
        target_sampling_rate = getattr(feature_extractor, "sampling_rate", None)
        if target_sampling_rate is None:
            target_sampling_rate = getattr(processor, "audio_sampling_rate", None)
        hop_length = getattr(feature_extractor, "hop_length", None)
        if hop_length is None:
            hop_length = getattr(processor, "audio_hop_length", None)
        has_audio_frontend = (
            type(target_sampling_rate) is int
            and target_sampling_rate > 0
            and type(hop_length) is int
            and hop_length > 0
        )
        if type(audio_num_mel_bins) is not int or audio_num_mel_bins <= 0:
            raise ValueError("audio_num_mel_bins must be a positive integer.")
        if audio_subsampling_factor is None:
            audio_subsampling_factor = getattr(
                processor,
                "audio_subsampling_factor",
                None,
            )
        has_audio_frontend = has_audio_frontend and (
            type(audio_subsampling_factor) is int
        )
        if has_audio_frontend and (
            audio_subsampling_factor <= 0
            or audio_subsampling_factor & (audio_subsampling_factor - 1)
        ):
            raise ValueError("audio_subsampling_factor must be a power of two.")
        if (
            audio_subsampling_factor is not None
            and type(audio_subsampling_factor) is not int
        ):
            raise ValueError(
                "Nemotron audio_subsampling_factor must be an integer when set."
            )
        if not 0 < min_audio_duration_seconds <= audio_clip_duration_seconds:
            raise ValueError(
                "min_audio_duration_seconds must be positive and no larger than "
                "audio_clip_duration_seconds."
            )
        if max_audio_duration_seconds < audio_clip_duration_seconds:
            raise ValueError(
                "max_audio_duration_seconds must be no smaller than "
                "audio_clip_duration_seconds."
            )
        self.feature_extractor = feature_extractor
        self.target_sampling_rate = target_sampling_rate
        self.hop_length = hop_length
        self.audio_subsampling_factor = audio_subsampling_factor
        self.audio_num_mel_bins = audio_num_mel_bins
        self.audio_clip_duration_seconds = audio_clip_duration_seconds
        self.min_audio_duration_seconds = min_audio_duration_seconds
        self.max_audio_duration_seconds = max_audio_duration_seconds
        fingerprint_data = {
            "visual_fingerprint": self.fingerprint,
            "target_sampling_rate": target_sampling_rate,
            "hop_length": hop_length,
            "audio_subsampling_factor": audio_subsampling_factor,
            "audio_num_mel_bins": audio_num_mel_bins,
            "audio_clip_duration_seconds": audio_clip_duration_seconds,
            "min_audio_duration_seconds": min_audio_duration_seconds,
            "max_audio_duration_seconds": max_audio_duration_seconds,
        }
        encoded = json.dumps(fingerprint_data, sort_keys=True).encode()
        self._fingerprint = hashlib.sha256(encoded).hexdigest()

    def _plan_audio(self, ref: MediaRef, *, media_index: int) -> _AudioPlan:
        if (
            type(self.target_sampling_rate) is not int
            or type(self.hop_length) is not int
            or type(self.audio_subsampling_factor) is not int
        ):
            raise ValueError(
                "This Nemotron processor has no audio frontend; audio samples "
                "require sampling_rate, hop_length, and audio_subsampling_factor."
            )
        return _audio_plan(
            ref,
            media_index=media_index,
            target_sampling_rate=self.target_sampling_rate,
            hop_length=self.hop_length,
            subsampling_factor=self.audio_subsampling_factor,
            clip_duration_seconds=self.audio_clip_duration_seconds,
            min_duration_seconds=self.min_audio_duration_seconds,
            max_duration_seconds=self.max_audio_duration_seconds,
        )

    def preencode(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        """Expand sound placeholders and retain exact media transform plans."""
        video_selections = self._video_selections(sample)
        messages, occurrences = _render_omni_messages(
            sample,
            temporal_patch_size=self.temporal_patch_size,
            prompt_format=self.prompt_format,
            thinking_trace_format=self.thinking_trace_format,
            relax_thinking_trace_check=self.relax_thinking_trace_check,
            video_timestamps={
                index: selection.timestamps
                for index, selection in video_selections.items()
            },
        )
        message_log = _tokenize_nemotron_sample(
            sample,
            messages,
            processor=self.processor,
            prompt_format=self.prompt_format,
        )
        audio_occurrences = [
            (0, ref, self._plan_audio(ref, media_index=media_index))
            for media_index, (_, ref) in enumerate(occurrences)
            if ref.modality == "audio"
        ]
        sound_token_id = _token_id(self.processor.tokenizer, SOUND_PLACEHOLDER)
        _expand_audio_placeholders(
            message_log,
            audio_occurrences,
            sound_token_id=sound_token_id,
        )

        length = sum(len(message["token_ids"]) for message in message_log)
        visual_media_indexes = [
            index
            for index, (_, ref) in enumerate(occurrences)
            if ref.modality != "audio"
        ]
        visual_occurrences = [
            (0, occurrences[index][1]) for index in visual_media_indexes
        ]
        visual_plans = self._plan_visual_occurrences(
            visual_occurrences,
            video_selections=video_selections,
            num_tokens_available=self.max_sequence_length - length - 4,
            data_augment=bool(sample.__subflavors__.get("data_augment", False)),
            media_indexes=visual_media_indexes,
        )
        visual_embeddings = sum(plan.num_embeddings for plan in visual_plans)
        visual_placeholders = sum(
            len(plan.embedding_widths) for plan in visual_plans
        )
        audio_embeddings = sum(
            plan.total_embeddings for _, _, plan in audio_occurrences
        )
        max_text_tokens = (
            self.packing_sequence_length - visual_embeddings + visual_placeholders
        )
        image_tokens_before_truncation = sum(
            len(message["visual_placeholder_positions"])
            for message in message_log
        )
        # Separate the two ways the placeholder count can disagree with the plan.
        # Conflating them blames truncation for samples that were never truncated.
        #
        # See the sibling check in nemotron_visual: a literal "<image>" in prose
        # is rejected at render time and cannot reach this comparison.
        if image_tokens_before_truncation != visual_placeholders:
            raise ValueError(
                f"Nemotron Omni sample {sample.__key__!r} rendered "
                f"{image_tokens_before_truncation} visual placeholders but "
                f"pre-encoding planned {visual_placeholders} from "
                f"{len(visual_plans)} visual media items."
            )
        original_length, length = _truncate_message_log(
            message_log,
            max_text_tokens=max_text_tokens,
            sample=sample,
        )
        remaining_visual_placeholders = sum(
            len(message["visual_placeholder_positions"])
            for message in message_log
        )
        if remaining_visual_placeholders != image_tokens_before_truncation:
            raise ValueError(
                f"Nemotron Omni truncation removed visual placeholders from sample "
                f"{sample.__key__!r}: expected {image_tokens_before_truncation}, "
                f"found {remaining_visual_placeholders}; original text/audio "
                f"length: {original_length}; max text/audio tokens: "
                f"{max_text_tokens}."
            )
        remaining_audio_embeddings = sum(
            int((message["token_ids"] == sound_token_id).sum().item())
            for message in message_log
        )
        if remaining_audio_embeddings != audio_embeddings:
            raise ValueError(
                f"Nemotron Omni truncation removed sound tokens from sample "
                f"{sample.__key__!r}: expected {audio_embeddings}, found "
                f"{remaining_audio_embeddings}; original text/audio length: "
                f"{original_length}; max text/audio tokens: {max_text_tokens}."
            )
        packing_cost = length + visual_embeddings - visual_placeholders
        # The fingerprint alone; see nemotron_visual.py for the measurement.
        # Here the model-input names made up to four partitions -- visual,
        # sound, both, neither -- so the fragmentation was worse than on the
        # visual-only path.
        return NemotronOmniEncodedSFTSample.derive_from(
            sample,
            message_log=message_log,
            length=length,
            packing_cost=packing_cost,
            loss_multiplier=1.0,
            group_key=(self.fingerprint,),
            sample_key=sample.__key__,
            pending_sample=sample,
            visual_plans=visual_plans,
            audio_plans=tuple(plan for _, _, plan in audio_occurrences),
        )

    def _process_audio(
        self,
        ref: MediaRef,
        plan: _AudioPlan,
    ) -> dict[str, PackedTensor]:
        feature_extractor = self.feature_extractor
        if feature_extractor is None:
            from transformers import ParakeetFeatureExtractor

            feature_extractor = ParakeetFeatureExtractor(
                feature_size=self.audio_num_mel_bins,
                sampling_rate=self.target_sampling_rate,
                hop_length=self.hop_length,
            )
        audio = _normalized_mono_audio(ref)
        if plan.source_sampling_rate != self.target_sampling_rate:
            # Librosa is an optional audio dependency, so keep it off non-audio
            # import paths. ParakeetFeatureExtractor also requires this package.
            import librosa

            audio = torch.from_numpy(
                librosa.resample(
                    audio.numpy(),
                    orig_sr=plan.source_sampling_rate,
                    target_sr=self.target_sampling_rate,
                )
            )
        min_samples = round(
            self.min_audio_duration_seconds * self.target_sampling_rate
        )
        if audio.shape[2] < min_samples:
            audio = F.pad(audio, (0, min_samples - audio.shape[2]))
        audio = audio.squeeze(1)
        if audio.shape[1] < plan.audio_length:
            audio = F.pad(audio, (0, plan.audio_length - audio.shape[1]))
        elif audio.shape[1] > plan.audio_length:
            audio = audio[:, : plan.audio_length]

        audio_lengths = torch.tensor([plan.audio_length], dtype=torch.long)
        clip_width = round(self.audio_clip_duration_seconds * self.target_sampling_rate)
        if audio.shape[1] > clip_width:
            clips = list(torch.split(audio, clip_width, dim=1))
            if len(clips) != plan.num_clips:
                raise ValueError(
                    f"Nemotron audio plan expects {plan.num_clips} clips, got "
                    f"{len(clips)} after resampling."
                )
            audio_lengths = torch.tensor(plan.samples_per_clip, dtype=torch.long)
            clips[-1] = F.pad(clips[-1], (0, clip_width - clips[-1].shape[1]))
            audio = torch.stack(clips).squeeze(1)

        if audio.ndim != 2:
            raise ValueError(
                "Nemotron audio transform must produce [clips, samples], got "
                f"{tuple(audio.shape)}."
            )
        if audio_lengths.sum().item() != plan.audio_length:
            raise ValueError(
                f"Nemotron audio plan expects {plan.audio_length} samples, got "
                f"{audio_lengths.sum().item()}."
            )
        if audio.shape[0] != plan.num_clips:
            raise ValueError(
                f"Nemotron audio plan expects {plan.num_clips} clips, got "
                f"{audio.shape[0]}."
            )
        if tuple(audio_lengths.tolist()) != plan.samples_per_clip:
            raise ValueError(
                "Nemotron audio samples per clip changed after resampling: "
                f"expected {plan.samples_per_clip}, got "
                f"{tuple(audio_lengths.tolist())}."
            )

        mel_features: list[torch.Tensor] = []
        valid_lengths: list[int] = []
        actual_embedding_counts: list[int] = []
        for clip, sample_count in zip(audio, plan.samples_per_clip, strict=True):
            # Bridge consumes mel tensors. Extract only the planned valid prefix;
            # batch collation supplies the physical mel padding.
            clip = clip[:sample_count]
            features = _feature_output(
                feature_extractor(
                    clip,
                    sampling_rate=self.target_sampling_rate,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
            )
            mel = torch.as_tensor(features.get("input_features"))
            attention_mask = torch.as_tensor(features.get("attention_mask"))
            if mel.ndim != 3 or mel.shape[0] != 1:
                raise ValueError(
                    "Nemotron audio feature extractor must return input_features "
                    f"with shape [1, frames, mel_bins], got {tuple(mel.shape)}."
                )
            if attention_mask.shape != mel.shape[:2]:
                raise ValueError(
                    "Nemotron audio attention_mask must match the feature frame "
                    f"shape, got {tuple(attention_mask.shape)} for "
                    f"{tuple(mel.shape)}."
                )
            mask = attention_mask.to(dtype=torch.bool)
            valid_length = int(mask.sum().item())
            expected_mask = torch.arange(mel.shape[1])[None, :] < valid_length
            if valid_length <= 0 or not torch.equal(mask.cpu(), expected_mask):
                raise ValueError(
                    "Nemotron audio attention_mask must contain one non-empty "
                    "contiguous prefix."
                )
            mel_features.append(mel[0])
            valid_lengths.append(valid_length)
            actual_embedding_counts.append(
                _subsampled_length(valid_length, self.audio_subsampling_factor)
            )

        if tuple(valid_lengths) != plan.valid_frame_counts:
            raise ValueError(
                "Nemotron audio frame count changed after feature extraction: "
                f"predicted {plan.valid_frame_counts}, got {tuple(valid_lengths)}."
            )
        if tuple(actual_embedding_counts) != plan.embedding_counts:
            raise ValueError(
                "Nemotron audio embedding count changed after feature extraction: "
                f"predicted {plan.embedding_counts}, got "
                f"{tuple(actual_embedding_counts)}."
            )
        max_frames = max(mel.shape[0] for mel in mel_features)
        num_mel_bins = {int(mel.shape[1]) for mel in mel_features}
        if len(num_mel_bins) != 1:
            raise ValueError("Nemotron audio clips must use one mel feature width.")
        sound_clips = mel_features[0].new_zeros(
            (len(mel_features), max_frames, num_mel_bins.pop())
        )
        for clip_index, mel in enumerate(mel_features):
            sound_clips[clip_index, : mel.shape[0]] = mel
        return {
            "sound_clips": PackedTensor(
                sound_clips,
                dim_to_pack=0,
                pad_to_max_shape=True,
            ),
            "sound_length": PackedTensor(
                torch.tensor(valid_lengths, dtype=torch.long),
                dim_to_pack=0,
            ),
        }

    def postencode(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        """Load the selected Omni media and apply the plans from pre-encoding."""
        pending_sample = sample.pending_sample
        if pending_sample is None:
            raise ValueError(
                f"Nemotron Omni sample {sample.sample_key!r} has no pending media."
            )
        if not isinstance(sample, NemotronOmniEncodedSFTSample):
            raise TypeError(
                "Nemotron Omni post-encoding requires the audio plan saved during "
                "pre-encoding."
            )
        message_log = deepcopy(sample.message_log)
        inputs_by_message: defaultdict[int, defaultdict[str, list[PackedTensor]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        visual_occurrences: list[tuple[int, tuple[int, ...]]] = []

        # Apply the plans saved during pre-encoding, mirroring the Megatron
        # reference's postencode_sample: load the selected media and run
        # apply_params against it. The saved plan alone decides the widths, so
        # nothing here re-renders the conversation or re-derives a width.
        for plan in sample.visual_plans:
            ref = pending_sample.media[plan.media_index]
            media_inputs, _ = self._process_media(ref, plan, pending_sample)
            for key, value in media_inputs.items():
                inputs_by_message[plan.message_index][key].append(value)
            visual_occurrences.append((plan.message_index, plan.embedding_widths))

        for plan in sample.audio_plans:
            ref = pending_sample.media[plan.media_index]
            for key, value in self._process_audio(ref, plan).items():
                inputs_by_message[0][key].append(value)

        image_token_id = _token_id(self.processor.tokenizer, "<image>")
        _expand_visual_placeholders(
            message_log,
            visual_occurrences,
            image_token_id=image_token_id,
        )
        expanded_length = sum(len(message["token_ids"]) for message in message_log)
        for message_index, keyed_inputs in inputs_by_message.items():
            message_log[message_index].update(
                {
                    key: PackedTensor.merge_segments(values)
                    for key, values in keyed_inputs.items()
                }
            )
        return EncodedSFTSample.derive_from(
            sample,
            message_log=message_log,
            length=expanded_length,
            packing_cost=sample.packing_cost,
            loss_multiplier=sample.loss_multiplier,
            group_key=sample.group_key,
            sample_key=sample.sample_key,
            pending_sample=None,
        )


@supports_model_families("nemotron")
class NemotronMultiModalTaskEncoder(GenericSFTTaskEncoder):
    """Run the reference Nemotron image, video, and optional audio lifecycle."""

    # Keep WebDataset payloads lazy until postencode selects the sample.
    decoder = None

    def __init__(
        self,
        *,
        adapter: SFTProcessorAdapter,
        cooker_functions: Sequence[SFTCooker],
        include_source_ids: bool,
        packer: SequencePacker | None = None,
        tokenizer: Any | None = None,
        sequence_length_pad_multiple: int = 1,
        only_unmask_final: bool = False,
        packing_sequence_length: int | None = None,
        patch_dim: int = 16,
        temporal_patch_size: int = 2,
        prompt_format: str = "nemotron-h-5p5-reasoning",
        thinking_trace_format: str = "normalized",
        relax_thinking_trace_check: bool = False,
        video_min_num_frames: int = 8,
        video_max_num_frames: int = 32,
        video_default_fps: int = 2,
        video_frame_temporal_jitter: bool = False,
        video_aug_scale_frames_up: int | None = None,
        video_aug_scale_resolution_up: int | None = None,
        video_aug_scale_resolution_only: bool = False,
        allow_large_videos: bool = False,
        tiling_augment_prob: float = 0.4,
        video_decode_thread_count: int = 8,
        audio_subsampling_factor: int | None = None,
        audio_num_mel_bins: int = 128,
        audio_clip_duration_seconds: float = 30.0,
        min_audio_duration_seconds: float = 0.1,
        max_audio_duration_seconds: float = 1800.0,
    ) -> None:
        if isinstance(adapter, NemotronMultiModalProcessorAdapter):
            omni_adapter = adapter
        elif isinstance(adapter, HFMultimodalSFTProcessorAdapter):
            omni_adapter = NemotronMultiModalProcessorAdapter(
                processor=adapter.processor,
                max_sequence_length=adapter.max_sequence_length,
                packing_sequence_length=packing_sequence_length,
                patch_dim=patch_dim,
                temporal_patch_size=temporal_patch_size,
                prompt_format=prompt_format,
                thinking_trace_format=thinking_trace_format,
                relax_thinking_trace_check=relax_thinking_trace_check,
                video_min_num_frames=video_min_num_frames,
                video_max_num_frames=video_max_num_frames,
                video_default_fps=video_default_fps,
                video_frame_temporal_jitter=video_frame_temporal_jitter,
                video_aug_scale_frames_up=video_aug_scale_frames_up,
                video_aug_scale_resolution_up=video_aug_scale_resolution_up,
                video_aug_scale_resolution_only=video_aug_scale_resolution_only,
                allow_large_videos=allow_large_videos,
                tiling_augment_prob=tiling_augment_prob,
                video_decode_thread_count=video_decode_thread_count,
                audio_subsampling_factor=audio_subsampling_factor,
                audio_num_mel_bins=audio_num_mel_bins,
                audio_clip_duration_seconds=audio_clip_duration_seconds,
                min_audio_duration_seconds=min_audio_duration_seconds,
                max_audio_duration_seconds=max_audio_duration_seconds,
                add_bos=adapter.add_bos,
                add_eos=adapter.add_eos,
                add_generation_prompt=adapter.add_generation_prompt,
            )
        else:
            raise TypeError(
                "Nemotron multimodal SFT requires the Hugging Face or Nemotron "
                "processor adapter."
            )
        super().__init__(
            adapter=omni_adapter,
            cooker_functions=cooker_functions,
            include_source_ids=include_source_ids,
            packer=packer,
            tokenizer=tokenizer,
            sequence_length_pad_multiple=sequence_length_pad_multiple,
            only_unmask_final=only_unmask_final,
            loss_mask_mode="precomputed",
        )
        self._multimodal_adapter = omni_adapter

    @stateless(restore_seeds=True)
    def preencode_sample(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        return self._multimodal_adapter.preencode(sample)

    @stateless(restore_seeds=True)
    def postencode_sample(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        return self._multimodal_adapter.postencode(sample)


__all__ = [
    "NemotronMultiModalProcessorAdapter",
    "NemotronMultiModalTaskEncoder",
    "SOUND_END",
    "SOUND_PLACEHOLDER",
    "SOUND_START",
]

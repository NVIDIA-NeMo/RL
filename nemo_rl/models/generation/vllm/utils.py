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
import json
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from PIL import Image

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    ROUTED_EXPERTS_FALLBACK_DTYPE,
    ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
    GenerationDatumSpec,
)
from nemo_rl.utils.routed_experts_codec import encode_routed_experts

R3_MISSING_ROUTE_SENTINEL = ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL
VLLM_LOGPROB_FLOOR = -9999.0

_VIDEO_SAMPLING_STYLE_ENV = "NRL_VIDEO_SAMPLING_STYLE"
_VIDEO_SAMPLING_STYLE_CURRENT = "current_fixed"
_VIDEO_SAMPLING_STYLE_SFT_V2_DURATION = "sft_v2_duration"
_VIDEO_SAMPLING_STYLE_NEMOTRON_VL = "nemotron_vl"
_VIDEO_SAMPLING_STYLE_DEFAULT = _VIDEO_SAMPLING_STYLE_SFT_V2_DURATION
_VIDEO_TEMPORAL_PATCH_SIZE_ENV = "NRL_VIDEO_TEMPORAL_PATCH_SIZE"
_SUPPORTED_VIDEO_SAMPLING_STYLES = {
    _VIDEO_SAMPLING_STYLE_CURRENT,
    _VIDEO_SAMPLING_STYLE_NEMOTRON_VL,
    _VIDEO_SAMPLING_STYLE_SFT_V2_DURATION,
}

_TORCHCODEC_END_OF_STREAM_ERROR = (
    "Requested next frame while there are no more frames left to decode."
)
_CACHED_VIDEO_FRAME_MANIFEST_MAGIC = b"NEMO_RL_CACHED_VIDEO_FRAMES_V1\n"
_CACHED_VIDEO_FRAME_MANIFEST_MIME = "video/x-nemo-rl-cached-frames"


def _get_video_sampling_style() -> str:
    style = os.environ.get(_VIDEO_SAMPLING_STYLE_ENV, _VIDEO_SAMPLING_STYLE_DEFAULT)
    style = style.strip().lower()
    if style not in _SUPPORTED_VIDEO_SAMPLING_STYLES:
        supported = ", ".join(sorted(_SUPPORTED_VIDEO_SAMPLING_STYLES))
        raise ValueError(
            f"Unsupported {_VIDEO_SAMPLING_STYLE_ENV}={style!r}; supported: {supported}"
        )
    return style


def _get_positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _round_video_frame_count(
    num_frames: int,
    *,
    total_frames_in_file: int,
    max_frames: int,
    temporal_patch_size: int,
) -> int:
    num_frames = min(num_frames, total_frames_in_file)
    if temporal_patch_size > 1 and num_frames % temporal_patch_size != 0:
        rounded_down = (num_frames // temporal_patch_size) * temporal_patch_size
        rounded_up = rounded_down + temporal_patch_size
        if rounded_up <= total_frames_in_file and rounded_up <= max_frames:
            num_frames = rounded_up
        else:
            num_frames = max(temporal_patch_size, rounded_down)
    return num_frames


def _timestamp_to_video_frame_index(
    timestamp_s: float, fps: float, total_frames: int
) -> int:
    """Convert a timestamp according to the active video sampling contract."""
    if _get_video_sampling_style() == _VIDEO_SAMPLING_STYLE_NEMOTRON_VL:
        frame_idx = round(timestamp_s * fps)
    else:
        frame_idx = int(timestamp_s * fps)
    return max(0, min(int(frame_idx), total_frames - 1))


def _select_video_frame_count(
    *,
    total_duration: float,
    requested_num_frames: int,
    total_frames_in_file: int,
    temporal_patch_size: int,
) -> int:
    requested_num_frames = max(1, int(requested_num_frames))
    if _get_video_sampling_style() == _VIDEO_SAMPLING_STYLE_SFT_V2_DURATION:
        min_frames = _get_positive_int_env("NRL_VIDEO_SFT_MIN_FRAMES", 8)
        max_frames = _get_positive_int_env("NRL_VIDEO_SFT_MAX_FRAMES", 256)
        default_fps = _get_positive_int_env("NRL_VIDEO_SFT_DEFAULT_FPS", 2)
        if total_frames_in_file < min_frames:
            num_frames = total_frames_in_file
        else:
            duration_frames = int(default_fps * total_duration)
            num_frames = min(max(duration_frames, min_frames), max_frames)
        num_frames = min(num_frames, requested_num_frames)
    else:
        num_frames = requested_num_frames

    return _round_video_frame_count(
        num_frames,
        total_frames_in_file=total_frames_in_file,
        max_frames=requested_num_frames,
        temporal_patch_size=temporal_patch_size,
    )


def _compute_video_timestamps(
    total_duration: float,
    num_frames: int,
    total_frames_in_file: int,
    original_num_frames: int,
    temporal_patch_size: int,
) -> tuple[int, list[float]]:
    num_frames = _select_video_frame_count(
        total_duration=total_duration,
        requested_num_frames=original_num_frames,
        total_frames_in_file=total_frames_in_file,
        temporal_patch_size=temporal_patch_size,
    )

    if _get_video_sampling_style() == _VIDEO_SAMPLING_STYLE_NEMOTRON_VL:
        if num_frames <= 1 or total_duration <= 0 or total_frames_in_file <= 1:
            return max(1, num_frames), [0.0]
        fps = total_frames_in_file / total_duration
        last_timestamp_s = (total_frames_in_file - 1) / fps
        timestamps_s = np.linspace(0.0, last_timestamp_s, num_frames, dtype=float)
        frame_indices = [
            max(0, min(round(float(ts) * fps), total_frames_in_file - 1))
            for ts in timestamps_s
        ]
        timestamps_s = [idx / fps for idx in frame_indices]
        return num_frames, timestamps_s

    if num_frames <= 1:
        return 1, [total_duration / 2.0]

    effective_span = max(total_duration - 1, 0)
    segment_size = effective_span / num_frames
    return num_frames, [
        segment_size * (frame_idx + 0.5) for frame_idx in range(num_frames)
    ]


def _build_video_metadata(
    *,
    fps: float,
    total_frames: int,
    sampled_indices: list[int],
    backend: str,
) -> dict[str, Any]:
    return {
        "fps": fps,
        "duration": total_frames / fps,
        "total_num_frames": total_frames,
        "frames_indices": sampled_indices,
        "video_backend": backend,
        "video_sampling_style": _get_video_sampling_style(),
        "do_sample_frames": False,
    }


def _resolve_cached_video_media_path(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise ValueError(
            "Cached Gym video frames require local paths or file:// URLs, "
            f"got scheme {parsed.scheme!r}."
        )
    else:
        path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Cached Gym video paths must be absolute, got {value!r}.")

    resolved = path.resolve()
    media_root_value = os.environ.get("NEMO_RL_VIDEO_MEDIA_ROOT")
    if not media_root_value:
        raise ValueError(
            "NEMO_RL_VIDEO_MEDIA_ROOT must be set when using cached Gym video frames."
        )
    media_root = Path(media_root_value).expanduser().resolve()
    if resolved != media_root and media_root not in resolved.parents:
        raise ValueError(
            f"Cached Gym video path {resolved} must be under "
            f"NEMO_RL_VIDEO_MEDIA_ROOT={media_root}."
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Cached Gym video file does not exist: {resolved}")
    return resolved


def build_cached_video_frame_data_url(
    frame_paths: list[str],
) -> str:
    """Build a compact native-video URL backed by lossless cached PNG frames."""
    if not frame_paths:
        raise ValueError("Cached Gym video requires at least one frame.")

    resolved_frames = [
        str(_resolve_cached_video_media_path(frame_path)) for frame_path in frame_paths
    ]
    manifest = {
        "frame_paths": resolved_frames,
        # The cache contains lossless sampled frames but not source timing
        # metadata, and original videos are not guaranteed to remain mounted.
        # Match vLLM's built-in image-sequence contract: one synthetic second
        # per cached frame with stable sequential indices.
        "metadata": {
            "fps": 1.0,
            "duration": float(len(resolved_frames)),
            "total_num_frames": len(resolved_frames),
            "frames_indices": list(range(len(resolved_frames))),
            "video_backend": "cached_png_sequence",
            "do_sample_frames": False,
        },
    }
    payload = _CACHED_VIDEO_FRAME_MANIFEST_MAGIC + json.dumps(
        manifest, separators=(",", ":")
    ).encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{_CACHED_VIDEO_FRAME_MANIFEST_MIME};base64,{encoded}"


def _load_cached_video_frame_manifest(
    data: bytes,
    *,
    num_frames: int,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Load an internal cached-frame manifest passed through vLLM VideoMediaIO."""
    if not data.startswith(_CACHED_VIDEO_FRAME_MANIFEST_MAGIC):
        return None

    try:
        manifest = json.loads(data[len(_CACHED_VIDEO_FRAME_MANIFEST_MAGIC) :])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid cached Gym video frame manifest.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Cached Gym video frame manifest must be a JSON object.")

    frame_paths = manifest.get("frame_paths")
    metadata = manifest.get("metadata")
    if (
        not isinstance(frame_paths, list)
        or not frame_paths
        or not all(isinstance(path, str) and path for path in frame_paths)
    ):
        raise ValueError(
            "Cached Gym video frame manifest requires non-empty frame_paths."
        )
    if num_frames >= 0 and len(frame_paths) != num_frames:
        raise ValueError(
            "Cached Gym video frame count does not match vLLM's requested "
            f"num_frames: cached={len(frame_paths)}, requested={num_frames}."
        )
    if not isinstance(metadata, dict):
        raise ValueError("Cached Gym video frame manifest requires metadata.")

    fps = metadata.get("fps")
    frame_indices = metadata.get("frames_indices")
    total_num_frames = metadata.get("total_num_frames")
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("Cached Gym video metadata requires a positive fps.")
    if not isinstance(frame_indices, list) or not all(
        isinstance(index, int) and index >= 0 for index in frame_indices
    ):
        raise ValueError(
            "Cached Gym video metadata requires non-negative frames_indices."
        )
    if len(frame_indices) != len(frame_paths):
        raise ValueError(
            "Cached Gym video metadata/frame mismatch: "
            f"indices={len(frame_indices)}, frames={len(frame_paths)}."
        )
    if not isinstance(total_num_frames, int) or total_num_frames <= 0:
        raise ValueError(
            "Cached Gym video metadata requires a positive total_num_frames."
        )

    frames = []
    expected_size = None
    for frame_path in frame_paths:
        resolved_path = _resolve_cached_video_media_path(frame_path)
        with Image.open(resolved_path) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if expected_size is None:
            expected_size = frame.shape
        elif frame.shape != expected_size:
            raise ValueError(
                "Cached Gym video frames must have one shape, got "
                f"{expected_size} and {frame.shape}."
            )
        frames.append(frame)

    loaded_metadata = dict(metadata)
    loaded_metadata["video_backend"] = "cached_png_nemotron_vl"
    loaded_metadata["do_sample_frames"] = False
    return np.stack(frames), loaded_metadata


def _is_torchcodec_end_of_stream_error(exc: RuntimeError) -> bool:
    return _TORCHCODEC_END_OF_STREAM_ERROR in str(exc)


def _torchcodec_sample_indices(
    *,
    total_frames: int,
    fps: float,
    requested_num_frames: int,
    temporal_patch_size: int,
) -> list[int]:
    _, timestamps_s = _compute_video_timestamps(
        total_frames / fps,
        requested_num_frames,
        total_frames,
        requested_num_frames,
        temporal_patch_size,
    )
    return [
        _timestamp_to_video_frame_index(timestamp, fps, total_frames)
        for timestamp in timestamps_s
    ]


def _find_torchcodec_decodable_frame_count(
    decoder_factory: Callable[[], Any],
    declared_total_frames: int,
) -> int:
    """Find the decodable tail when container metadata overstates frame count."""
    decoder = decoder_factory()

    def can_decode(frame_index: int) -> bool:
        try:
            decoder.get_frame_at(frame_index)
        except RuntimeError as exc:
            if _is_torchcodec_end_of_stream_error(exc):
                return False
            raise
        return True

    last_declared_index = declared_total_frames - 1
    if can_decode(last_declared_index):
        return declared_total_frames
    if not can_decode(0):
        raise ValueError("Video has no decodable frames")

    last_decodable = 0
    first_undecodable = last_declared_index
    while first_undecodable - last_decodable > 1:
        candidate = (last_decodable + first_undecodable) // 2
        if can_decode(candidate):
            last_decodable = candidate
        else:
            first_undecodable = candidate
    return last_decodable + 1


def _decode_torchcodec_video(
    source: Any,
    *,
    requested_num_frames: int,
    temporal_patch_size: int,
    source_description: str,
    initial_decoder: Any | None = None,
) -> tuple[np.ndarray, float, int, list[int]]:
    """Decode sampled frames, recovering from overstated container metadata."""
    from torchcodec.decoders import VideoDecoder

    def decoder_factory() -> Any:
        return VideoDecoder(
            source,
            dimension_order="NHWC",
            num_ffmpeg_threads=0,
            device="cpu",
            seek_mode="exact",
        )

    decoder = initial_decoder if initial_decoder is not None else decoder_factory()
    total_frames = int(decoder.metadata.num_frames or 0)
    fps = float(decoder.metadata.average_fps or 0.0)
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {source_description}")
    if fps <= 0:
        raise ValueError(f"Video has invalid fps ({fps}): {source_description}")

    sampled_indices = _torchcodec_sample_indices(
        total_frames=total_frames,
        fps=fps,
        requested_num_frames=requested_num_frames,
        temporal_patch_size=temporal_patch_size,
    )
    try:
        frames = decoder.get_frames_at(indices=sampled_indices).data
    except RuntimeError as exc:
        if not _is_torchcodec_end_of_stream_error(exc):
            raise

        decodable_frames = _find_torchcodec_decodable_frame_count(
            decoder_factory, total_frames
        )
        if decodable_frames != total_frames:
            print(
                "WARNING: TorchCodec container metadata overstates the decodable "
                f"frame count for {source_description}: declared={total_frames}, "
                f"decodable={decodable_frames}. Resampling over decodable frames.",
                flush=True,
            )
            total_frames = decodable_frames
            sampled_indices = _torchcodec_sample_indices(
                total_frames=total_frames,
                fps=fps,
                requested_num_frames=requested_num_frames,
                temporal_patch_size=temporal_patch_size,
            )

        retry_decoder = decoder_factory()
        try:
            frames = retry_decoder.get_frames_at(indices=sampled_indices).data
        except RuntimeError as retry_exc:
            if not _is_torchcodec_end_of_stream_error(retry_exc):
                raise
            # Some FFmpeg inputs fail only in batched indexed decoding. Decode the
            # same indices individually so rollout and policy inputs stay aligned.
            individual_decoder = decoder_factory()
            frames = torch.stack(
                [
                    individual_decoder.get_frame_at(frame_index).data
                    for frame_index in sampled_indices
                ]
            )

    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError(
            "TorchCodec returned invalid RGB video frames "
            f"with shape {frames.shape}: {source_description}"
        )
    return frames, fps, total_frames, sampled_indices


def _load_video_frames_pyav_with_metadata(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    import av

    try:
        container = av.open(video_path)
    except Exception as exc:
        raise ValueError(f"Cannot open video: {video_path}") from exc

    if not container.streams.video:
        container.close()
        raise ValueError(f"No video stream in {video_path}")

    stream = container.streams.video[0]
    stream.codec_context.thread_type = "NONE"
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    if fps <= 0:
        container.close()
        raise ValueError(f"Video has invalid fps ({fps}): {video_path}")

    total_frames = stream.frames
    if total_frames <= 0:
        if stream.duration and stream.time_base:
            duration_estimate = float(stream.duration * stream.time_base)
        elif container.duration:
            duration_estimate = container.duration / av.time_base
        else:
            duration_estimate = 0.0
        total_frames = max(1, int(duration_estimate * fps))
    total_duration = total_frames / fps

    num_frames, timestamps_s = _compute_video_timestamps(
        total_duration,
        num_frames,
        total_frames,
        num_frames,
        temporal_patch_size,
    )
    time_base = float(stream.time_base) if stream.time_base else 1.0 / fps
    target_pts_list = [int(timestamp / time_base) for timestamp in timestamps_s]
    sampled_indices = [
        _timestamp_to_video_frame_index(timestamp, fps, total_frames)
        for timestamp in timestamps_s
    ]

    frames: list[np.ndarray] = []
    try:
        if target_pts_list:
            container.seek(max(0, target_pts_list[0]), stream=stream, any_frame=False)
        target_idx = 0
        best_frame = None
        frame_counter = 0
        for frame in container.decode(video=0):
            if target_idx >= len(target_pts_list):
                break
            best_frame = frame
            frame_counter += 1
            if frame.pts is None:
                while (
                    target_idx < len(target_pts_list)
                    and frame_counter >= target_idx + 1
                ):
                    frames.append(frame.reformat(format="rgb24").to_ndarray())
                    target_idx += 1
                continue
            frame_end = frame.pts + (frame.duration if frame.duration else 1)
            while (
                target_idx < len(target_pts_list)
                and target_pts_list[target_idx] < frame_end
            ):
                frames.append(frame.reformat(format="rgb24").to_ndarray())
                target_idx += 1
        if best_frame is not None:
            last_frame = best_frame.reformat(format="rgb24").to_ndarray()
            while len(frames) < len(target_pts_list):
                frames.append(last_frame.copy())
    finally:
        container.close()

    if not frames:
        raise ValueError(f"Failed to extract any frames from video: {video_path}")
    metadata = _build_video_metadata(
        fps=fps,
        total_frames=total_frames,
        sampled_indices=sampled_indices,
        backend="pyav",
    )
    return np.stack(frames), metadata


def _load_video_frames_decord_with_metadata(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    from decord import VideoReader
    from decord import cpu as decord_cpu

    reader = VideoReader(video_path, ctx=decord_cpu(), num_threads=1)
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    fps = reader.get_avg_fps()
    if fps <= 0:
        raise ValueError(f"Video has invalid fps ({fps}): {video_path}")

    num_frames, timestamps_s = _compute_video_timestamps(
        total_frames / fps,
        num_frames,
        total_frames,
        num_frames,
        temporal_patch_size,
    )
    indices = [
        _timestamp_to_video_frame_index(timestamp, fps, total_frames)
        for timestamp in timestamps_s
    ]
    metadata = _build_video_metadata(
        fps=fps,
        total_frames=total_frames,
        sampled_indices=indices,
        backend="decord",
    )
    return reader.get_batch(indices).asnumpy(), metadata


def _load_video_frames_vllm_with_metadata(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode video with the same loader used by raw Gym vLLM requests."""
    del temporal_patch_size
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY

    loader_name = os.environ.get("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    loader = VIDEO_LOADER_REGISTRY.load(loader_name)
    frames, metadata = loader.load_bytes(
        Path(video_path).read_bytes(), num_frames=int(num_frames)
    )
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError(
            f"vLLM video loader {loader_name!r} returned invalid frames "
            f"with shape {frames.shape}: {video_path}"
        )
    metadata = dict(metadata)
    metadata["video_sampling_style"] = _get_video_sampling_style()
    return frames, metadata


def _load_video_frames_torchcodec_with_metadata(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode video with the repository's optional TorchCodec dependency."""
    try:
        frames, fps, total_frames, sampled_indices = _decode_torchcodec_video(
            video_path,
            requested_num_frames=num_frames,
            temporal_patch_size=temporal_patch_size,
            source_description=video_path,
        )
    except ImportError as exc:
        raise ImportError(
            "Gym video preprocessing requires the optional video dependencies. "
            "Run `bash tools/install_audio_deps.sh` before training."
        ) from exc

    metadata = _build_video_metadata(
        fps=fps,
        total_frames=total_frames,
        sampled_indices=sampled_indices,
        backend="torchcodec",
    )
    return frames, metadata


def register_torchcodec_vllm_video_loader() -> bool:
    """Use TorchCodec for raw Nemotron video bytes parsed by vLLM's HTTP server.

    vLLM's ``nemotron_vl`` loader defaults to OpenCV, while NeMo-RL deliberately
    does not ship OpenCV or PyAV. Registering this structural ``VideoLoader``
    implementation under the same extension name keeps vLLM's media connector
    contract and makes rollout decoding match policy-logprob preprocessing.

    Returns:
        Whether the TorchCodec loader was registered.
    """
    video_backend = os.environ.get("NRL_VIDEO_BACKEND", "torchcodec").strip().lower()
    vllm_loader = os.environ.get("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    if video_backend != "torchcodec" or vllm_loader != "nemotron_vl":
        return False

    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY

    class TorchCodecNemotronVLVideoBackend:
        @classmethod
        def load_bytes(
            cls,
            data: bytes,
            num_frames: int = -1,
            fps: int = -1,
            max_duration: int = 300,
            frame_recovery: bool = False,
            **kwargs: Any,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            del cls, max_duration, kwargs
            if frame_recovery:
                raise ValueError(
                    "frame_recovery is not supported by the TorchCodec video loader"
                )

            cached_video = _load_cached_video_frame_manifest(
                data, num_frames=int(num_frames)
            )
            if cached_video is not None:
                return cached_video

            try:
                from torchcodec.decoders import VideoDecoder
            except ImportError as exc:
                raise ImportError(
                    "Gym video generation requires the optional video dependencies. "
                    "Run `bash tools/install_audio_deps.sh` before training."
                ) from exc

            decoder = VideoDecoder(
                data,
                dimension_order="NHWC",
                num_ffmpeg_threads=0,
                device="cpu",
                seek_mode="exact",
            )
            total_frames = int(decoder.metadata.num_frames or 0)
            source_fps = float(decoder.metadata.average_fps or 0.0)
            if total_frames <= 0:
                raise ValueError("Video has no frames")
            if source_fps <= 0:
                raise ValueError(f"Video has invalid fps ({source_fps})")

            requested_num_frames = (
                total_frames if int(num_frames) < 0 else int(num_frames)
            )
            if fps > 0:
                duration_limited_frames = max(
                    1, int((total_frames / source_fps) * float(fps))
                )
                requested_num_frames = min(
                    requested_num_frames, duration_limited_frames
                )

            frames, source_fps, total_frames, sampled_indices = (
                _decode_torchcodec_video(
                    data,
                    requested_num_frames=requested_num_frames,
                    temporal_patch_size=_get_positive_int_env(
                        _VIDEO_TEMPORAL_PATCH_SIZE_ENV, 1
                    ),
                    source_description="in-memory video",
                    initial_decoder=decoder,
                )
            )

            metadata = _build_video_metadata(
                fps=source_fps,
                total_frames=total_frames,
                sampled_indices=sampled_indices,
                backend="torchcodec_nemotron_vl",
            )
            metadata["original_video_bytes"] = data
            return frames, metadata

    VIDEO_LOADER_REGISTRY.register("nemotron_vl")(TorchCodecNemotronVLVideoBackend)
    return True


def load_video_frames_with_metadata(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load sampled RGB frames and the sampling metadata used by vLLM."""
    backend = os.environ.get("NRL_VIDEO_BACKEND", "torchcodec").strip().lower()
    if backend == "torchcodec":
        return _load_video_frames_torchcodec_with_metadata(
            video_path, num_frames, temporal_patch_size
        )
    if backend == "vllm":
        return _load_video_frames_vllm_with_metadata(
            video_path, num_frames, temporal_patch_size
        )
    if backend == "decord":
        return _load_video_frames_decord_with_metadata(
            video_path, num_frames, temporal_patch_size
        )
    if backend != "pyav":
        raise ValueError(
            f"Unsupported NRL_VIDEO_BACKEND={backend!r}; "
            "supported: 'torchcodec', 'pyav', 'decord', 'vllm'"
        )
    return _load_video_frames_pyav_with_metadata(
        video_path, num_frames, temporal_patch_size
    )


def load_video_frames(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> np.ndarray:
    """Load sampled RGB video frames without returning metadata."""
    frames, _ = load_video_frames_with_metadata(
        video_path, num_frames, temporal_patch_size
    )
    return frames


# The expert-id range vs carry dtype is model-constant, so it is verified on the
# first non-empty routed-experts tensor per process and skipped afterwards.
G_ROUTED_EXPERTS_RANGE_CHECKED = False


def _as_routed_experts_tensor(
    value: Any, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Convert backend routed-expert ids to the resolved carry dtype.

    Guards against expert ids overflowing ``dtype`` before the narrowing cast,
    which would otherwise wrap silently (e.g. if the expert count was
    mis-detected when resolving the dtype).
    """
    global G_ROUTED_EXPERTS_RANGE_CHECKED
    tensor = torch.as_tensor(value, device=device)
    if not G_ROUTED_EXPERTS_RANGE_CHECKED and tensor.numel() > 0:
        max_id = int(tensor.max())
        limit = torch.iinfo(dtype).max
        if max_id > limit:
            raise ValueError(
                f"routed expert id {max_id} exceeds the resolved carry dtype "
                f"{dtype} (max {limit}); the model's expert count was likely "
                "mis-detected (see resolve_routed_experts_dtype in "
                "nemo_rl.models.generation.interfaces)."
            )
        G_ROUTED_EXPERTS_RANGE_CHECKED = True
    return tensor.to(dtype=dtype)


def format_prompt_for_vllm_generation(
    data: BatchedDataDict[GenerationDatumSpec], sample_idx: Optional[int] = None
) -> list[dict[str, Any]]:
    """Format a list of prompts for vllm generation (which requires a specific format for its own `generate` method).

    See https://docs.vllm.ai/en/v0.9.1/features/multimodal_inputs.html for prompt format for multimodal inputs.
    """
    # Prepare prompts for vLLM (removing padding)
    prompts = []

    input_ids = data["input_ids"]
    batch_size = input_ids.shape[0]
    input_lengths = data["input_lengths"]

    # if sample_idx is None, return list of all prompts for the entire batch
    # else, return the prompt for the single sample specified by sample_idx
    return_all = sample_idx is None
    if sample_idx is None:
        start_idx = 0
        end_idx = batch_size
    else:
        start_idx = sample_idx
        end_idx = sample_idx + 1

    def _get_regular_prompt(index: int):
        valid_length = input_lengths[index].item()
        valid_ids = (
            input_ids[index, :valid_length]
            if valid_length > 0
            else input_ids[index, :0]
        )
        token_ids = valid_ids.tolist()
        return {"prompt_token_ids": token_ids}

    # Check if this is VLM generation by looking for message_log with images
    # Support for videos/audio/etc. can be added here
    # if 'message_log' in data and any('images' in msg for msg in data['message_log']):
    if "vllm_content" in data:
        # VLM generation using content and multi_modal_data
        for i in range(start_idx, end_idx):
            msg = data["vllm_content"][i]
            # if msg is None, this conversation had no multimodal content, fallback to regular prompt
            if msg is None:
                prompts.append(_get_regular_prompt(i))
                continue
            # init prompt dict
            prompt_dict = {"prompt": msg}
            # collect multi_modal_data from images, audios, and videos
            multi_modal_data = {}
            images = data.get("vllm_images", None)
            if images is not None and len(images[i]) > 0:
                multi_modal_data["image"] = (
                    images[i][0] if len(images[i]) == 1 else images[i]
                )
            audios = data.get("vllm_audios", None)
            if audios is not None and len(audios[i]) > 0:
                multi_modal_data["audio"] = (
                    audios[i][0] if len(audios[i]) == 1 else audios[i]
                )
            videos = data.get("vllm_videos", None)
            if videos is not None and len(videos[i]) > 0:
                multi_modal_data["video"] = (
                    videos[i][0] if len(videos[i]) == 1 else videos[i]
                )
            if not multi_modal_data:
                prompts.append(_get_regular_prompt(i))
                continue
            prompt_dict["multi_modal_data"] = multi_modal_data
            prompts.append(prompt_dict)
    else:
        # Regular LLM generation using token_ids (pre-tokenized).
        # Note: eval.py uses raw prompt strings instead of token IDs because its
        # collate function produces message_log dicts, not tokenized tensors.
        # Both are valid vLLM input formats but may tokenize slightly differently.
        for i in range(start_idx, end_idx):
            # Use input_lengths to get only valid tokens (not padding)
            prompts.append(_get_regular_prompt(i))

    return prompts if return_all else prompts[0]


def pad_and_align_routed_expert_indices(
    request_output: Any,
    completion_output: Any,
    *,
    valid_length: int,
    padded_length: int,
    device: torch.device,
    require_complete_routed_experts: bool = False,
    allow_missing_routed_experts_fallback: bool = True,
    return_stats: bool = False,
    routed_experts_dtype: torch.dtype = ROUTED_EXPERTS_FALLBACK_DTYPE,
) -> Optional[torch.Tensor] | tuple[Optional[torch.Tensor], dict[str, int]]:
    """Return full-sequence-aligned routed experts as ``[S, L, topk]`` in ``routed_experts_dtype``."""
    routed = getattr(completion_output, "routed_experts", None)
    prompt_routed = getattr(request_output, "prompt_routed_experts", None)

    if prompt_routed is not None:
        prompt_routed = _as_routed_experts_tensor(
            prompt_routed, device=device, dtype=routed_experts_dtype
        )
    if routed is not None:
        routed = _as_routed_experts_tensor(
            routed, device=device, dtype=routed_experts_dtype
        )

    if prompt_routed is not None and routed is not None:
        routed = torch.cat((prompt_routed, routed), dim=0)
    elif prompt_routed is not None:
        routed = prompt_routed

    expected_routes = min(max(valid_length - 1, 0), padded_length)
    stats = {
        "actual_routes": 0,
        "expected_routes": expected_routes,
        "missing_routes": 0,
        "surplus_routes": 0,
    }

    if routed is None:
        return (None, stats) if return_stats else None
    if routed.dim() != 3:
        raise ValueError(
            "vLLM routed_experts must have shape [tokens, num_moe_layers, topk], "
            f"got {tuple(routed.shape)}"
        )

    stats["actual_routes"] = int(routed.shape[0])
    stats["missing_routes"] = max(expected_routes - int(routed.shape[0]), 0)
    stats["surplus_routes"] = max(int(routed.shape[0]) - (expected_routes + 1), 0)
    if (
        require_complete_routed_experts
        and stats["missing_routes"] > 0
        and not allow_missing_routed_experts_fallback
    ):
        # This has only been observed rarely with vLLM prefix caching plus
        # chunked prefill: a small number of samples can omit routed-expert
        # rows even though most requests are complete. Keep
        # tools/model_diagnostics/6.vllm_routed_experts_completeness.py as a
        # standalone reproducer for upstream vLLM bug reports.
        num_cached_tokens = getattr(request_output, "num_cached_tokens", None)
        raise ValueError(
            "vLLM returned incomplete routed_experts for router replay: "
            f"routes={routed.shape[0]}, expected_at_least={expected_routes}, "
            f"valid_length={valid_length}, padded_length={padded_length}, "
            f"num_cached_tokens={num_cached_tokens}. This usually means the "
            "generation backend did not return routed experts for every "
            "non-final token in the prompt+response sequence."
        )
    max_allowed_routes = expected_routes + 1
    if require_complete_routed_experts and routed.shape[0] > max_allowed_routes:
        num_cached_tokens = getattr(request_output, "num_cached_tokens", None)
        raise ValueError(
            "vLLM returned too many routed_experts routes for router replay: "
            f"routes={routed.shape[0]}, expected={expected_routes}, "
            f"max_allowed={max_allowed_routes}, valid_length={valid_length}, "
            f"padded_length={padded_length}, num_cached_tokens={num_cached_tokens}. "
            "Router replay allows at most one surplus final-token route."
        )

    default_route = torch.arange(
        routed.shape[2],
        dtype=routed_experts_dtype,
        device=device,
    )
    full = (
        default_route.view(1, 1, -1)
        .expand(padded_length, routed.shape[1], routed.shape[2])
        .clone()
    )
    routes_to_copy = min(expected_routes, routed.shape[0])
    if routes_to_copy > 0:
        full[:routes_to_copy] = routed[:routes_to_copy].to(device=device)
    if stats["missing_routes"] > 0:
        full[routes_to_copy:expected_routes] = R3_MISSING_ROUTE_SENTINEL
    return (full, stats) if return_stats else full


def attach_routed_experts_to_chat_response_choices(
    response: Any,
    final_request_output: Any,
    *,
    device: torch.device,
    logger: Any = None,
    routed_experts_dtype: torch.dtype = ROUTED_EXPERTS_FALLBACK_DTYPE,
) -> Any:
    """Attach aligned routed experts to OpenAI chat response choices."""
    outputs_by_index = {
        output.index: output for output in getattr(final_request_output, "outputs", [])
    }
    prompt_token_count = len(
        getattr(final_request_output, "prompt_token_ids", []) or []
    )

    choices = list(getattr(response, "choices", []))
    attached_choice_indices = set()
    for choice in choices:
        generation_details = outputs_by_index.get(choice.index)
        if generation_details is None:
            continue
        attached_choice_indices.add(choice.index)

        generation_token_count = len(getattr(generation_details, "token_ids", []) or [])
        routed_result = pad_and_align_routed_expert_indices(
            final_request_output,
            generation_details,
            valid_length=prompt_token_count + generation_token_count,
            padded_length=prompt_token_count + generation_token_count,
            device=device,
            require_complete_routed_experts=True,
            return_stats=True,
            routed_experts_dtype=routed_experts_dtype,
        )
        if not isinstance(routed_result, tuple):
            raise RuntimeError(
                "Expected routed_experts alignment to return stats for the "
                "OpenAI-compatible chat endpoint."
            )
        routed_experts, r3_stats = routed_result
        if routed_experts is None:
            raise RuntimeError(
                "vLLM was asked to return routed experts for the "
                "OpenAI-compatible chat endpoint but the generation "
                "output did not include routed_experts."
            )
        if r3_stats["missing_routes"] > 0 and logger is not None:
            logger.warning(
                "R3 router replay fallback: vLLM returned incomplete "
                "routed_experts for chat choice_idx=%d, "
                "missing_token_routes=%d, actual_routes=%d, "
                "expected_routes=%d. Megatron will use its own router "
                "for those missing token routes.",
                choice.index,
                r3_stats["missing_routes"],
                r3_stats["actual_routes"],
                r3_stats["expected_routes"],
            )
        # Base64 envelope instead of .tolist(): nested JSON int lists cost
        # ~1s of CPU per serialize/parse hop at long context lengths and get
        # re-validated at every gym HTTP hop; a single string passes through
        # the gym chain opaquely.
        choice.message.routed_experts = encode_routed_experts(
            routed_experts.to(dtype=routed_experts_dtype)
        )

    if len(attached_choice_indices) != len(choices):
        missing_choice_indices = sorted(
            choice.index
            for choice in choices
            if choice.index not in attached_choice_indices
        )
        raise RuntimeError(
            "vLLM was asked to return routed experts for the "
            "OpenAI-compatible chat endpoint but response choices could not be "
            "matched to generation outputs: "
            f"missing_choice_indices={missing_choice_indices}."
        )

    return response


def attach_token_information_to_chat_response_choices(
    response: Any,
    final_request_output: Any,
) -> Any:
    """Attach engine-native token information to OpenAI chat response choices."""
    prompt_token_ids = getattr(final_request_output, "prompt_token_ids", None)
    if prompt_token_ids is None:
        raise RuntimeError(
            "vLLM was asked to return token information for the "
            "OpenAI-compatible chat endpoint but the final request output did "
            "not include prompt_token_ids."
        )

    generation_outputs = list(getattr(final_request_output, "outputs", []))
    generation_output_indices = [output.index for output in generation_outputs]
    outputs_by_index = {output.index: output for output in generation_outputs}
    if len(outputs_by_index) != len(generation_outputs):
        raise RuntimeError(
            "vLLM returned duplicate generation output indices while attaching "
            "token information to the OpenAI-compatible chat response."
        )

    choices = list(getattr(response, "choices", []))
    choice_indices = [choice.index for choice in choices]
    if len(set(choice_indices)) != len(choice_indices):
        raise RuntimeError(
            "vLLM returned duplicate response choice indices while attaching "
            "token information to the OpenAI-compatible chat response."
        )

    choice_index_set = set(choice_indices)
    output_index_set = set(generation_output_indices)
    missing_choice_indices = sorted(choice_index_set - output_index_set)
    unexpected_output_indices = sorted(output_index_set - choice_index_set)
    if missing_choice_indices or unexpected_output_indices:
        raise RuntimeError(
            "vLLM was asked to return token information for the "
            "OpenAI-compatible chat endpoint but response choices could not be "
            "matched to generation outputs: "
            f"missing_choice_indices={missing_choice_indices}, "
            f"unexpected_output_indices={unexpected_output_indices}."
        )

    for choice in choices:
        generation_details = outputs_by_index[choice.index]
        output_token_ids = getattr(generation_details, "token_ids", None)
        if output_token_ids is None:
            raise RuntimeError(
                "vLLM was asked to return token information for the "
                "OpenAI-compatible chat endpoint but generation output "
                f"choice_idx={choice.index} did not include token_ids."
            )
        generation_token_ids = list(output_token_ids)

        generation_logprob_details = getattr(generation_details, "logprobs", None)
        if generation_logprob_details is None:
            if generation_token_ids:
                raise RuntimeError(
                    "vLLM was asked to return token information for the "
                    "OpenAI-compatible chat endpoint but generation output "
                    f"choice_idx={choice.index} did not include logprobs."
                )
            generation_log_probs = []
        else:
            if len(generation_token_ids) != len(generation_logprob_details):
                raise RuntimeError(
                    "vLLM returned mismatched generation token IDs and log "
                    "probabilities for the OpenAI-compatible chat endpoint: "
                    f"choice_idx={choice.index}, "
                    f"token_count={len(generation_token_ids)}, "
                    f"logprob_count={len(generation_logprob_details)}."
                )
            generation_log_probs = []
            for token_id, position_logprobs in zip(
                generation_token_ids, generation_logprob_details
            ):
                selected_token_logprob = position_logprobs.get(token_id)
                if selected_token_logprob is None:
                    raise RuntimeError(
                        "vLLM generation log probabilities did not include the "
                        "selected token while attaching token information to "
                        "the OpenAI-compatible chat response: "
                        f"choice_idx={choice.index}, token_id={token_id}."
                    )
                generation_log_probs.append(
                    max(float(selected_token_logprob.logprob), VLLM_LOGPROB_FLOOR)
                )

        choice.message.prompt_token_ids = list(prompt_token_ids)
        choice.message.generation_token_ids = generation_token_ids
        choice.message.generation_log_probs = generation_log_probs

    return response


def model_dump_chat_response_with_dynamic_message_fields(
    response: Any,
) -> dict[str, Any]:
    """Dump a vLLM OpenAI chat response while preserving dynamic message fields."""
    response_dict = response.model_dump()
    for choice, choice_dict in zip(
        getattr(response, "choices", []), response_dict.get("choices", [])
    ):
        message = getattr(choice, "message", None)
        for field_name in (
            "routed_experts",
            "prompt_token_ids",
            "generation_token_ids",
            "generation_log_probs",
        ):
            field_value = getattr(message, field_name, None)
            if field_value is not None:
                choice_dict.setdefault("message", {})[field_name] = field_value
    return response_dict


def aggregate_spec_decode_counters(
    worker_metrics: list[dict[str, float | list[float]]],
) -> dict[str | tuple[str, int], float]:
    """Aggregate speculative decoding counters from multiple workers.

    Combines spec decode metrics collected from DP leader workers into
    a single aggregated counter dictionary.

    Args:
        worker_metrics: List of metric dictionaries from each worker.
            Each dict maps metric names to float values or lists of floats
            (for per-position metrics).

    Returns:
        Dictionary mapping metric names to their aggregated float values.
        Per-position metrics use (name, position) tuples as keys.

    Example:
        >>> metrics_from_workers = policy_generation.get_metrics()
        >>> counters = aggregate_spec_decode_counters(metrics_from_workers)
        >>> print(counters.get("vllm:spec_decode_num_drafts", 0))
        1234.0
    """
    counters: dict[str | tuple[str, int], float] = defaultdict(float)

    for report in worker_metrics:
        for metric_name, value in report.items():
            if "spec_decode" in metric_name:
                if isinstance(value, list):
                    # Per-position metrics (e.g., acceptance counts at each draft position)
                    for position, pos_value in enumerate(value, 1):
                        counters[metric_name, position] += pos_value
                else:
                    counters[metric_name] += value

    return dict(counters)


def compute_spec_decode_metrics(
    start_counters: dict[str | tuple[str, int], float],
    end_counters: dict[str | tuple[str, int], float],
) -> dict[str, float]:
    """Compute delta and derived metrics for speculative decoding.

    Calculates the difference between two counter snapshots and derives
    acceptance rate and acceptance length metrics for logging.

    Args:
        start_counters: Counter snapshot taken before generation.
        end_counters: Counter snapshot taken after generation.

    Returns:
        Dictionary of metrics suitable for logging to wandb/tensorboard.
        Keys are prefixed with "vllm/" for namespace consistency.
        Includes:
            - vllm/spec_num_drafts: Total number of draft batches
            - vllm/spec_num_draft_tokens: Total draft tokens generated
            - vllm/spec_num_accepted_tokens: Total tokens accepted
            - vllm/spec_acceptance_length: Average accepted tokens per draft + 1
            - vllm/spec_acceptance_rate: Ratio of accepted to draft tokens
            - vllm/{metric}-{position}: Per-position acceptance counts
            - vllm/spec_acceptance_rate-pos-{position}: Per-position acceptance rates
    """
    keys = set(start_counters) | set(end_counters)
    delta = {k: end_counters.get(k, 0.0) - start_counters.get(k, 0.0) for k in keys}

    num_drafts = delta.get("vllm:spec_decode_num_drafts", 0.0)
    num_draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    num_accepted_tokens = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)

    # acceptance_length = 1 + (accepted / drafts) represents average tokens
    # generated per draft batch (1 target model token + accepted draft tokens)
    acceptance_length = (
        1.0 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0
    )
    acceptance_rate = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0.0
    )

    spec_metrics: dict[str, float] = {
        "vllm/spec_num_drafts": num_drafts,
        "vllm/spec_num_draft_tokens": num_draft_tokens,
        "vllm/spec_num_accepted_tokens": num_accepted_tokens,
        "vllm/spec_acceptance_length": acceptance_length,
        "vllm/spec_acceptance_rate": acceptance_rate,
    }

    # Add per-position metrics for detailed analysis
    for key, value in delta.items():
        if isinstance(key, tuple):
            metric_name, position = key
            spec_metrics[f"vllm/{metric_name}-{position}"] = value
            if num_drafts > 0:
                spec_metrics[f"vllm/spec_acceptance_rate-pos-{position}"] = (
                    value / num_drafts
                )

    return spec_metrics


# TODO: Replace this hard-coded map with a generic plugin-registration
# hook on ``VllmGeneration`` (e.g. a ``worker_cls_overrides`` registry populated
# by ``nemo_rl.modelopt`` on import) so core has no knowledge of ModelOpt-specific
# worker classes.
GENERATION_WORKER_OVERRIDES = {
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker",
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker",
}


def resolve_generation_worker_cls(default_cls: str, config: dict) -> str:
    """Return the quantized vLLM generation worker FQN if ``quant_cfg`` is set, else ``default_cls``.

    Safe to call even when ModelOpt is not installed — returns ``default_cls``
    unchanged whenever ``quant_cfg`` is ``None``, so the core generation path
    stays import-free of ModelOpt.
    """
    if config.get("quant_cfg") is None:
        return default_cls
    return GENERATION_WORKER_OVERRIDES.get(default_cls, default_cls)

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
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from PIL import Image

VideoSamplingStyle = Literal["nemotron_vl"]

_TORCHCODEC_END_OF_STREAM_ERROR = (
    "Requested next frame while there are no more frames left to decode."
)
_CACHED_VIDEO_FRAME_MANIFEST_MAGIC = b"NEMO_RL_CACHED_VIDEO_FRAMES_V1\n"
_CACHED_VIDEO_FRAME_MANIFEST_MIME = "video/x-nemo-rl-cached-frames"


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
    timestamp_s: float,
    fps: float,
    total_frames: int,
    sampling_style: VideoSamplingStyle,
) -> int:
    """Convert a timestamp according to the configured sampling contract."""
    if sampling_style != "nemotron_vl":
        raise ValueError(f"Unsupported video sampling style: {sampling_style!r}")
    frame_idx = round(timestamp_s * fps)
    return max(0, min(int(frame_idx), total_frames - 1))


def _select_video_frame_count(
    *,
    requested_num_frames: int,
    total_frames_in_file: int,
    temporal_patch_size: int,
    sampling_style: VideoSamplingStyle,
) -> int:
    requested_num_frames = max(1, int(requested_num_frames))
    if sampling_style != "nemotron_vl":
        raise ValueError(f"Unsupported video sampling style: {sampling_style!r}")
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
    sampling_style: VideoSamplingStyle,
) -> tuple[int, list[float]]:
    num_frames = _select_video_frame_count(
        requested_num_frames=original_num_frames,
        total_frames_in_file=total_frames_in_file,
        temporal_patch_size=temporal_patch_size,
        sampling_style=sampling_style,
    )

    if sampling_style != "nemotron_vl":
        raise ValueError(f"Unsupported video sampling style: {sampling_style!r}")
    if num_frames <= 1 or total_duration <= 0 or total_frames_in_file <= 1:
        num_frames = max(1, num_frames)
        return num_frames, [0.0] * num_frames
    fps = total_frames_in_file / total_duration
    last_timestamp_s = (total_frames_in_file - 1) / fps
    timestamps_s = np.linspace(0.0, last_timestamp_s, num_frames, dtype=float)
    frame_indices = [
        max(0, min(round(float(ts) * fps), total_frames_in_file - 1))
        for ts in timestamps_s
    ]
    return num_frames, [idx / fps for idx in frame_indices]


def _build_video_metadata(
    *,
    fps: float,
    total_frames: int,
    sampled_indices: list[int],
    backend: str,
    sampling_style: VideoSamplingStyle,
) -> dict[str, Any]:
    return {
        "fps": fps,
        "duration": total_frames / fps,
        "total_num_frames": total_frames,
        "frames_indices": sampled_indices,
        "video_backend": backend,
        "video_sampling_style": sampling_style,
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


def build_cached_video_frame_metadata(num_frames: int) -> dict[str, Any]:
    """Return the shared synthetic timing contract for cached video frames."""
    if num_frames < 1:
        raise ValueError("Cached Gym video requires at least one frame.")

    # The cache contains lossless sampled frames but not source timing
    # metadata, and original videos are not guaranteed to remain mounted.
    # Match vLLM's built-in image-sequence contract: one synthetic second
    # per cached frame with stable sequential indices.
    return {
        "fps": 1.0,
        "duration": float(num_frames),
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "cached_png_sequence",
        "do_sample_frames": False,
    }


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
        "metadata": build_cached_video_frame_metadata(len(resolved_frames)),
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
    sampling_style: VideoSamplingStyle,
) -> list[int]:
    _, timestamps_s = _compute_video_timestamps(
        total_frames / fps,
        requested_num_frames,
        total_frames,
        requested_num_frames,
        temporal_patch_size,
        sampling_style,
    )
    return [
        _timestamp_to_video_frame_index(timestamp, fps, total_frames, sampling_style)
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
    sampling_style: VideoSamplingStyle,
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
        sampling_style=sampling_style,
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
                sampling_style=sampling_style,
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


def _load_video_frames_torchcodec_with_metadata(
    video_path: str,
    *,
    num_frames: int,
    temporal_patch_size: int,
    sampling_style: VideoSamplingStyle,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode video with the repository's optional TorchCodec dependency."""
    try:
        frames, fps, total_frames, sampled_indices = _decode_torchcodec_video(
            video_path,
            requested_num_frames=num_frames,
            temporal_patch_size=temporal_patch_size,
            sampling_style=sampling_style,
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
        sampling_style=sampling_style,
    )
    return frames, metadata


def register_torchcodec_vllm_video_loader(
    *,
    sampling_style: VideoSamplingStyle,
    temporal_patch_size: int,
) -> None:
    """Use TorchCodec for raw Nemotron video bytes parsed by vLLM's HTTP server.

    vLLM's ``nemotron_vl`` loader defaults to OpenCV, while NeMo-RL deliberately
    does not ship OpenCV or PyAV. Registering this structural ``VideoLoader``
    implementation under the same extension name keeps vLLM's media connector
    contract and makes rollout decoding match policy-logprob preprocessing.

    The caller supplies the same validated values materialized into the policy
    data configuration. Registration is therefore not conditional on process
    environment state.
    """
    if temporal_patch_size <= 0:
        raise ValueError(
            f"video temporal_patch_size must be positive, got {temporal_patch_size}"
        )

    # vLLM reads this setting when it builds the media connector. Set it from
    # the validated NeMo RL config before importing the registry/engine.
    os.environ["VLLM_VIDEO_LOADER_BACKEND"] = "nemotron_vl"

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
                frames, metadata = cached_video
                metadata["video_sampling_style"] = sampling_style
                return frames, metadata

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
                    temporal_patch_size=temporal_patch_size,
                    sampling_style=sampling_style,
                    source_description="in-memory video",
                    initial_decoder=decoder,
                )
            )

            metadata = _build_video_metadata(
                fps=source_fps,
                total_frames=total_frames,
                sampled_indices=sampled_indices,
                backend="torchcodec_nemotron_vl",
                sampling_style=sampling_style,
            )
            metadata["original_video_bytes"] = data
            return frames, metadata

    VIDEO_LOADER_REGISTRY.register("nemotron_vl")(TorchCodecNemotronVLVideoBackend)


def load_video_frames_with_metadata(
    video_path: str,
    *,
    num_frames: int,
    temporal_patch_size: int,
    sampling_style: VideoSamplingStyle,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load sampled RGB frames through the supported TorchCodec backend."""
    return _load_video_frames_torchcodec_with_metadata(
        video_path,
        num_frames=num_frames,
        temporal_patch_size=temporal_patch_size,
        sampling_style=sampling_style,
    )

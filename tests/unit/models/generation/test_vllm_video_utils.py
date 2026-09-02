# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import base64
import builtins
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from nemo_rl.environments.nemotron_utils import _nemotron_video_target_resolution
from nemo_rl.models.generation.vllm import video_utils as utils
from nemo_rl.models.generation.vllm.config import (
    materialize_vllm_video_config,
    resolve_vllm_video_config,
)


def test_nemotron_vl_timestamps_use_rounded_uniform_frame_indices():
    num_frames, timestamps = utils._compute_video_timestamps(
        total_duration=10.0,
        num_frames=4,
        total_frames_in_file=20,
        original_num_frames=4,
        temporal_patch_size=2,
        sampling_style="nemotron_vl",
    )

    assert num_frames == 4
    assert timestamps == [0.0, 3.0, 6.5, 9.5]


def test_nemotron_vl_single_frame_is_repeated_for_temporal_patch():
    num_frames, timestamps = utils._compute_video_timestamps(
        total_duration=0.5,
        num_frames=1,
        total_frames_in_file=1,
        original_num_frames=1,
        temporal_patch_size=2,
        sampling_style="nemotron_vl",
    )

    assert num_frames == 2
    assert timestamps == [0.0, 0.0]


def test_public_video_loader_uses_torchcodec(monkeypatch):
    expected_frames = np.zeros((4, 1, 1, 3), dtype=np.uint8)
    expected_metadata = {"video_backend": "torchcodec"}
    calls = []

    def fake_torchcodec_loader(
        video_path, *, num_frames, temporal_patch_size, sampling_style
    ):
        calls.append((video_path, num_frames, temporal_patch_size, sampling_style))
        return expected_frames, expected_metadata

    monkeypatch.setattr(
        utils,
        "_load_video_frames_torchcodec_with_metadata",
        fake_torchcodec_loader,
    )

    frames, metadata = utils.load_video_frames_with_metadata(
        "video.mp4",
        num_frames=4,
        temporal_patch_size=2,
        sampling_style="nemotron_vl",
    )

    assert frames is expected_frames
    assert metadata == expected_metadata
    assert calls == [("video.mp4", 4, 2, "nemotron_vl")]


def _install_fake_video_modules(monkeypatch, decoder_cls):
    registered = {}

    class FakeRegistry:
        def register(self, name):
            def decorator(loader):
                registered[name] = loader
                return loader

            return decorator

    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    multimodal_module = ModuleType("vllm.multimodal")
    multimodal_module.__path__ = []
    video_module = ModuleType("vllm.multimodal.video")
    video_module.VIDEO_LOADER_REGISTRY = FakeRegistry()
    torchcodec_module = ModuleType("torchcodec")
    torchcodec_module.__path__ = []
    torchcodec_decoders_module = ModuleType("torchcodec.decoders")
    torchcodec_decoders_module.VideoDecoder = decoder_cls
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", multimodal_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal.video", video_module)
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec_module)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", torchcodec_decoders_module)
    return registered


def test_registered_torchcodec_loader_matches_policy_sampling(monkeypatch):
    captured = []

    class FakeVideoDecoder:
        def __init__(self, source, **kwargs):
            del kwargs
            self.source = source
            self.metadata = SimpleNamespace(num_frames=20, average_fps=2.0)

        def get_frames_at(self, *, indices):
            captured.append((self.source, list(indices)))
            return SimpleNamespace(
                data=torch.zeros((len(indices), 2, 4, 3), dtype=torch.uint8)
            )

    registered = _install_fake_video_modules(monkeypatch, FakeVideoDecoder)
    utils.register_torchcodec_vllm_video_loader(
        sampling_style="nemotron_vl", temporal_patch_size=2
    )

    rollout_frames, metadata = registered["nemotron_vl"].load_bytes(
        b"video-bytes", num_frames=4
    )

    assert captured == [(b"video-bytes", [0, 6, 13, 19])]
    assert rollout_frames.shape == (4, 2, 4, 3)
    assert metadata["frames_indices"] == [0, 6, 13, 19]
    assert metadata["video_sampling_style"] == "nemotron_vl"
    assert metadata["original_video_bytes"] == b"video-bytes"
    assert os.environ["VLLM_VIDEO_LOADER_BACKEND"] == "nemotron_vl"


def test_registered_loader_reads_cached_frame_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_RL_VIDEO_MEDIA_ROOT", str(tmp_path))

    class DecoderMustNotRun:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("cached frames must not instantiate TorchCodec")

    registered = _install_fake_video_modules(monkeypatch, DecoderMustNotRun)
    utils.register_torchcodec_vllm_video_loader(
        sampling_style="nemotron_vl", temporal_patch_size=2
    )

    expected_frames = []
    frame_paths = []
    for index in range(2):
        frame = np.full((3, 4, 3), index * 127, dtype=np.uint8)
        frame_path = tmp_path / f"frame_{index:04d}.png"
        Image.fromarray(frame).save(frame_path)
        expected_frames.append(frame)
        frame_paths.append(str(frame_path))
    payload = utils.CACHED_VIDEO_FRAME_MANIFEST_MAGIC + json.dumps(
        {
            "frame_paths": frame_paths,
            "metadata": {
                "fps": 1.0,
                "duration": 2.0,
                "total_num_frames": 2,
                "frames_indices": [0, 1],
            },
        }
    ).encode("utf-8")

    frames, metadata = registered["nemotron_vl"].load_bytes(payload, num_frames=2)

    np.testing.assert_array_equal(frames, np.stack(expected_frames))
    assert metadata["video_backend"] == "cached_png_nemotron_vl"
    assert metadata["video_sampling_style"] == "nemotron_vl"


def test_cached_video_data_url_requires_no_driver_decoder(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_RL_VIDEO_MEDIA_ROOT", str(tmp_path))
    frame_paths = []
    for index in range(4):
        frame_path = tmp_path / f"frame_{index:04d}.png"
        Image.new("RGB", (2, 2), color=(index, 0, 0)).save(frame_path)
        frame_paths.append(str(frame_path))

    data_url = utils.build_cached_video_frame_data_url(frame_paths)

    _, encoded = data_url.split(",", 1)
    payload = base64.b64decode(encoded)
    manifest = json.loads(payload[len(utils.CACHED_VIDEO_FRAME_MANIFEST_MAGIC) :])
    assert manifest["frame_paths"] == frame_paths
    assert manifest["metadata"]["frames_indices"] == [0, 1, 2, 3]
    assert manifest["metadata"]["fps"] == 1.0
    assert manifest["metadata"]["duration"] == 4.0
    assert manifest["metadata"]["total_num_frames"] == 4


def test_cached_video_manifest_does_not_import_torchcodec(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_RL_VIDEO_MEDIA_ROOT", str(tmp_path))
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (2, 2)).save(frame_path)
    payload = utils.CACHED_VIDEO_FRAME_MANIFEST_MAGIC + json.dumps(
        {
            "frame_paths": [str(frame_path)],
            "metadata": {
                "fps": 1.0,
                "duration": 1.0,
                "total_num_frames": 1,
                "frames_indices": [0],
            },
        }
    ).encode("utf-8")
    original_import = builtins.__import__

    def reject_torchcodec_import(name, *args, **kwargs):
        if name == "torchcodec.decoders":
            raise AssertionError("cached manifest must not import TorchCodec")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torchcodec_import)
    loaded = utils._load_cached_video_frame_manifest(payload, num_frames=1)

    assert loaded is not None
    assert loaded[0].shape == (1, 2, 2, 3)


def test_cached_video_media_root_is_required(monkeypatch, tmp_path):
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (2, 2)).save(frame_path)
    monkeypatch.delenv("NEMO_RL_VIDEO_MEDIA_ROOT", raising=False)

    with pytest.raises(ValueError, match="NEMO_RL_VIDEO_MEDIA_ROOT"):
        utils.build_cached_video_frame_data_url([str(frame_path)])


def test_cached_video_frame_cannot_escape_media_root(monkeypatch, tmp_path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    outside_frame = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside_frame)
    monkeypatch.setenv("NEMO_RL_VIDEO_MEDIA_ROOT", str(media_root))

    with pytest.raises(ValueError, match="must be under"):
        utils.build_cached_video_frame_data_url([str(outside_frame)])


@pytest.mark.parametrize(
    ("width", "height", "patches", "maintain_aspect_ratio"),
    [
        (1280, 720, 1024, True),
        (720, 1280, 1024, True),
        (640, 640, 1024, False),
        (320, 180, 256, True),
    ],
)
def test_nemotron_target_resolution_matches_vllm(
    width, height, patches, maintain_aspect_ratio
):
    upstream = pytest.importorskip(
        "vllm.transformers_utils.processors.nano_nemotron_vl"
    )
    expected_width, expected_height, _ = (
        upstream.get_video_target_size_and_feature_size(
            width,
            height,
            patches,
            maintain_aspect_ratio,
            16,
            0.5,
        )
    )

    assert _nemotron_video_target_resolution(
        original_width=width,
        original_height=height,
        target_num_patches=patches,
        patch_size=16,
        downsample_ratio=0.5,
        maintain_aspect_ratio=maintain_aspect_ratio,
    ) == (expected_width, expected_height)


def test_materialize_video_config_is_single_source_of_sampling_values():
    generation = {
        "backend": "vllm",
        "vllm_cfg": {
            "video": {
                "sampling_style": "nemotron_vl",
                "num_frames": 32,
                "temporal_patch_size": 2,
            }
        },
        "vllm_kwargs": {
            "limit_mm_per_prompt": {"video": {"count": 1, "num_frames": 8}}
        },
    }
    policy = {"generation": generation, "tokenizer": {}}
    data = {}

    materialize_vllm_video_config(policy, data)

    resolved = resolve_vllm_video_config(generation)
    assert resolved is not None
    assert policy["tokenizer"]["video"]["num_frames"] == 32
    assert data["default"] == {
        "num_frames": 32,
        "video_sampling_style": "nemotron_vl",
        "video_temporal_patch_size": 2,
    }
    assert generation["vllm_kwargs"]["limit_mm_per_prompt"]["video"]["num_frames"] == 32
    assert generation["vllm_kwargs"]["media_io_kwargs"]["video"]["num_frames"] == 32


def test_materialize_video_config_requires_video_limit_mapping():
    policy = {
        "generation": {
            "backend": "vllm",
            "vllm_cfg": {
                "video": {
                    "sampling_style": "nemotron_vl",
                    "num_frames": 32,
                    "temporal_patch_size": 2,
                }
            },
            "vllm_kwargs": {"limit_mm_per_prompt": {}},
        },
        "tokenizer": {},
    }

    with pytest.raises(ValueError, match="limit_mm_per_prompt.video"):
        materialize_vllm_video_config(policy, {})


def test_video_config_rejects_unknown_sampling_fields():
    generation = {
        "vllm_cfg": {
            "video": {
                "sampling_style": "nemotron_vl",
                "num_frames": 32,
                "temporal_patch_size": 2,
                "sampling_stlye": "typo",
            }
        }
    }

    with pytest.raises(ValueError, match="sampling_stlye"):
        resolve_vllm_video_config(generation)

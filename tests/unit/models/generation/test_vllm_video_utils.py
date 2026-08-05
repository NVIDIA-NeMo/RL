# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import base64
import builtins
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest
import torch
from PIL import Image

from nemo_rl.models.generation.vllm import utils
from nemo_rl.models.generation.vllm.config import (
    should_reset_mm_cache_after_refit,
)


def test_sync_post_init_initializes_folded_radio_layerscale_before_numa_binding():
    from nemo_rl.models.generation.vllm.vllm_worker import (
        VllmGenerationWorkerImpl,
    )

    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.llm = SimpleNamespace(collective_rpc=MagicMock())
    worker.report_device_id = MagicMock(return_value=["gpu-0"])
    worker._mtp_load_from_disk = False
    worker._sparse_refit_receiver = None

    worker.post_init()

    assert worker.llm.collective_rpc.call_args_list == [
        call("_initialize_nemotron_omni_radio_layerscale"),
        call("bind_numa", args=tuple()),
    ]
    assert worker.vllm_device_ids == ["gpu-0"]


@pytest.mark.asyncio
async def test_async_post_init_awaits_folded_radio_layerscale_initialization():
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.llm = SimpleNamespace(collective_rpc=AsyncMock())
    worker.report_device_id_async = AsyncMock(return_value=["gpu-0"])
    worker._mtp_load_from_disk = False
    worker._sparse_refit_receiver = None

    await worker.post_init_async()

    assert worker.llm.collective_rpc.await_args_list == [
        call("_initialize_nemotron_omni_radio_layerscale"),
        call("bind_numa", args=tuple()),
    ]
    assert worker.vllm_device_ids == ["gpu-0"]


def test_nemotron_vl_timestamps_use_rounded_uniform_frame_indices(monkeypatch):
    monkeypatch.setenv("NRL_VIDEO_SAMPLING_STYLE", "nemotron_vl")

    num_frames, timestamps = utils._compute_video_timestamps(
        total_duration=10.0,
        num_frames=4,
        total_frames_in_file=20,
        original_num_frames=4,
        temporal_patch_size=2,
    )

    assert num_frames == 4
    assert timestamps == [0.0, 3.0, 6.5, 9.5]


def test_public_video_loader_honors_vllm_backend(monkeypatch):
    monkeypatch.setenv("NRL_VIDEO_SAMPLING_STYLE", "nemotron_vl")
    monkeypatch.setenv("NRL_VIDEO_BACKEND", "vllm")
    expected_frames = np.zeros((4, 1, 1, 3), dtype=np.uint8)
    expected_metadata = {
        "frames_indices": [0, 3, 6, 9],
        "video_sampling_style": "nemotron_vl",
    }
    calls = []

    def fake_vllm_loader(video_path, num_frames, temporal_patch_size):
        calls.append((video_path, num_frames, temporal_patch_size))
        return expected_frames, expected_metadata

    monkeypatch.setattr(
        utils,
        "_load_video_frames_vllm_with_metadata",
        fake_vllm_loader,
    )

    frames, metadata = utils.load_video_frames_with_metadata(
        "video.mp4", num_frames=4, temporal_patch_size=2
    )

    assert frames is expected_frames
    assert metadata == expected_metadata
    assert calls == [("video.mp4", 4, 2)]


def test_public_video_loader_defaults_to_torchcodec(monkeypatch):
    monkeypatch.delenv("NRL_VIDEO_BACKEND", raising=False)
    expected_frames = np.zeros((4, 1, 1, 3), dtype=np.uint8)
    expected_metadata = {"video_backend": "torchcodec"}
    calls = []

    def fake_torchcodec_loader(video_path, num_frames, temporal_patch_size):
        calls.append((video_path, num_frames, temporal_patch_size))
        return expected_frames, expected_metadata

    monkeypatch.setattr(
        utils,
        "_load_video_frames_torchcodec_with_metadata",
        fake_torchcodec_loader,
    )

    frames, metadata = utils.load_video_frames_with_metadata(
        "video.mp4", num_frames=4, temporal_patch_size=2
    )

    assert frames is expected_frames
    assert metadata == expected_metadata
    assert calls == [("video.mp4", 4, 2)]


def test_torchcodec_loader_uses_nemotron_sampling_and_returns_rgb(monkeypatch):
    monkeypatch.setenv("NRL_VIDEO_SAMPLING_STYLE", "nemotron_vl")
    captured = {}

    class FakeVideoDecoder:
        def __init__(self, source, **kwargs):
            captured["source"] = source
            captured["decoder_kwargs"] = kwargs
            self.metadata = SimpleNamespace(num_frames=20, average_fps=2.0)

        def get_frames_at(self, *, indices):
            captured["indices"] = indices
            return SimpleNamespace(
                data=torch.zeros((len(indices), 2, 4, 3), dtype=torch.uint8)
            )

    torchcodec_module = ModuleType("torchcodec")
    torchcodec_module.__path__ = []
    torchcodec_decoders_module = ModuleType("torchcodec.decoders")
    torchcodec_decoders_module.VideoDecoder = FakeVideoDecoder
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec_module)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", torchcodec_decoders_module)

    frames, metadata = utils._load_video_frames_torchcodec_with_metadata(
        "video.mp4", num_frames=4, temporal_patch_size=2
    )

    assert captured == {
        "source": "video.mp4",
        "decoder_kwargs": {
            "dimension_order": "NHWC",
            "num_ffmpeg_threads": 0,
            "device": "cpu",
            "seek_mode": "exact",
        },
        "indices": [0, 6, 13, 19],
    }
    assert frames.shape == (4, 2, 4, 3)
    assert metadata["frames_indices"] == [0, 6, 13, 19]
    assert metadata["video_sampling_style"] == "nemotron_vl"
    assert not metadata["do_sample_frames"]


def test_register_torchcodec_vllm_loader_decodes_raw_bytes_with_matching_frames(
    monkeypatch,
):
    monkeypatch.setenv("NRL_VIDEO_BACKEND", "torchcodec")
    monkeypatch.setenv("NRL_VIDEO_SAMPLING_STYLE", "nemotron_vl")
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "nemotron_vl")
    registered = {}
    captured = {}

    class FakeRegistry:
        def register(self, name):
            def decorator(loader):
                registered[name] = loader
                return loader

            return decorator

    class FakeVideoDecoder:
        def __init__(self, source, **kwargs):
            captured["source"] = source
            captured["decoder_kwargs"] = kwargs
            self.metadata = SimpleNamespace(num_frames=20, average_fps=2.0)

        def get_frames_at(self, *, indices):
            captured["indices"] = indices
            return SimpleNamespace(
                data=torch.zeros((len(indices), 2, 4, 3), dtype=torch.uint8)
            )

    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    multimodal_module = ModuleType("vllm.multimodal")
    multimodal_module.__path__ = []
    video_module = ModuleType("vllm.multimodal.video")
    video_module.VIDEO_LOADER_REGISTRY = FakeRegistry()
    torchcodec_module = ModuleType("torchcodec")
    torchcodec_module.__path__ = []
    torchcodec_decoders_module = ModuleType("torchcodec.decoders")
    torchcodec_decoders_module.VideoDecoder = FakeVideoDecoder
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", multimodal_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal.video", video_module)
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec_module)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", torchcodec_decoders_module)

    assert utils.register_torchcodec_vllm_video_loader()
    frames, metadata = registered["nemotron_vl"].load_bytes(
        b"video-bytes", num_frames=4
    )

    assert captured == {
        "source": b"video-bytes",
        "decoder_kwargs": {
            "dimension_order": "NHWC",
            "num_ffmpeg_threads": 0,
            "device": "cpu",
            "seek_mode": "exact",
        },
        "indices": [0, 6, 13, 19],
    }
    assert frames.shape == (4, 2, 4, 3)
    assert metadata["frames_indices"] == [0, 6, 13, 19]
    assert metadata["video_backend"] == "torchcodec_nemotron_vl"
    assert metadata["video_sampling_style"] == "nemotron_vl"
    assert metadata["original_video_bytes"] == b"video-bytes"


def test_registered_vllm_loader_reads_lossless_cached_frame_manifest(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("NRL_VIDEO_BACKEND", "torchcodec")
    monkeypatch.setenv("NRL_VIDEO_SAMPLING_STYLE", "nemotron_vl")
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "nemotron_vl")
    monkeypatch.setenv("NEMO_RL_VIDEO_MEDIA_ROOT", str(tmp_path))
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
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", multimodal_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal.video", video_module)
    original_import = builtins.__import__

    def reject_torchcodec_import(name, *args, **kwargs):
        if name == "torchcodec.decoders":
            raise AssertionError("cached frames must not import TorchCodec")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torchcodec_import)

    expected_frames = []
    frame_paths = []
    for index in range(2):
        frame = np.full((3, 4, 3), index * 127, dtype=np.uint8)
        frame_path = tmp_path / f"frame_{index:04d}.png"
        Image.fromarray(frame).save(frame_path)
        expected_frames.append(frame)
        frame_paths.append(str(frame_path))
    manifest = {
        "frame_paths": frame_paths,
        "metadata": {
            "fps": 1.0,
            "duration": 2.0,
            "total_num_frames": 2,
            "frames_indices": [0, 1],
        },
    }
    payload = utils._CACHED_VIDEO_FRAME_MANIFEST_MAGIC + json.dumps(manifest).encode(
        "utf-8"
    )

    assert utils.register_torchcodec_vllm_video_loader()
    frames, metadata = registered["nemotron_vl"].load_bytes(payload, num_frames=2)

    np.testing.assert_array_equal(frames, np.stack(expected_frames))
    assert metadata["frames_indices"] == [0, 1]
    assert metadata["video_backend"] == "cached_png_nemotron_vl"
    assert metadata["do_sample_frames"] is False


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
    manifest = json.loads(payload[len(utils._CACHED_VIDEO_FRAME_MANIFEST_MAGIC) :])
    assert manifest == {
        "frame_paths": frame_paths,
        "metadata": {
            "fps": 1.0,
            "duration": 4.0,
            "total_num_frames": 4,
            "frames_indices": [0, 1, 2, 3],
            "video_backend": "cached_png_sequence",
            "do_sample_frames": False,
        },
    }


def test_cached_video_manifest_rejects_empty_frame_list():
    manifest = {
        "frame_paths": [],
        "metadata": {
            "fps": 1.0,
            "duration": 0.0,
            "total_num_frames": 0,
            "frames_indices": [],
        },
    }
    payload = utils._CACHED_VIDEO_FRAME_MANIFEST_MAGIC + json.dumps(manifest).encode(
        "utf-8"
    )

    with pytest.raises(ValueError, match="requires non-empty frame_paths"):
        utils._load_cached_video_frame_manifest(payload, num_frames=-1)


def test_torchcodec_loaders_resample_over_decodable_frames(monkeypatch):
    monkeypatch.setenv("NRL_VIDEO_BACKEND", "torchcodec")
    monkeypatch.setenv("NRL_VIDEO_SAMPLING_STYLE", "nemotron_vl")
    monkeypatch.setenv("NRL_VIDEO_TEMPORAL_PATCH_SIZE", "2")
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "nemotron_vl")
    registered = {}
    batch_calls = []

    class FakeRegistry:
        def register(self, name):
            def decorator(loader):
                registered[name] = loader
                return loader

            return decorator

    class FakeVideoDecoder:
        def __init__(self, source, **kwargs):
            del kwargs
            self.source = source
            self.metadata = SimpleNamespace(num_frames=20, average_fps=2.0)

        def get_frame_at(self, frame_index):
            if frame_index >= 18:
                raise RuntimeError(utils._TORCHCODEC_END_OF_STREAM_ERROR)
            return SimpleNamespace(
                data=torch.full((2, 4, 3), frame_index, dtype=torch.uint8)
            )

        def get_frames_at(self, *, indices):
            batch_calls.append((self.source, list(indices)))
            if any(frame_index >= 18 for frame_index in indices):
                raise RuntimeError(utils._TORCHCODEC_END_OF_STREAM_ERROR)
            return SimpleNamespace(
                data=torch.stack(
                    [self.get_frame_at(frame_index).data for frame_index in indices]
                )
            )

    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    multimodal_module = ModuleType("vllm.multimodal")
    multimodal_module.__path__ = []
    video_module = ModuleType("vllm.multimodal.video")
    video_module.VIDEO_LOADER_REGISTRY = FakeRegistry()
    torchcodec_module = ModuleType("torchcodec")
    torchcodec_module.__path__ = []
    torchcodec_decoders_module = ModuleType("torchcodec.decoders")
    torchcodec_decoders_module.VideoDecoder = FakeVideoDecoder
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", multimodal_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal.video", video_module)
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec_module)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", torchcodec_decoders_module)

    policy_frames, policy_metadata = utils._load_video_frames_torchcodec_with_metadata(
        "video.mp4", num_frames=4, temporal_patch_size=2
    )
    assert utils.register_torchcodec_vllm_video_loader()
    rollout_frames, rollout_metadata = registered["nemotron_vl"].load_bytes(
        b"video-bytes", num_frames=4
    )

    assert policy_metadata["total_num_frames"] == 18
    assert rollout_metadata["total_num_frames"] == 18
    assert policy_metadata["frames_indices"] == [0, 6, 11, 17]
    assert rollout_metadata["frames_indices"] == [0, 6, 11, 17]
    np.testing.assert_array_equal(policy_frames, rollout_frames)
    assert batch_calls == [
        ("video.mp4", [0, 6, 13, 19]),
        ("video.mp4", [0, 6, 11, 17]),
        (b"video-bytes", [0, 6, 13, 19]),
        (b"video-bytes", [0, 6, 11, 17]),
    ]


def test_register_torchcodec_vllm_loader_leaves_other_backends_unchanged(
    monkeypatch,
):
    monkeypatch.setenv("NRL_VIDEO_BACKEND", "pyav")
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "nemotron_vl")

    assert not utils.register_torchcodec_vllm_video_loader()


def test_vllm_loader_preserves_odd_valid_frame_count(monkeypatch, tmp_path):
    frames = np.zeros((29, 1, 1, 3), dtype=np.uint8)
    metadata = {"frames_indices": list(range(29))}
    loader = SimpleNamespace(load_bytes=lambda *args, **kwargs: (frames, metadata))
    registry = SimpleNamespace(load=lambda name: loader)

    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    multimodal_module = ModuleType("vllm.multimodal")
    multimodal_module.__path__ = []
    video_module = ModuleType("vllm.multimodal.video")
    video_module.VIDEO_LOADER_REGISTRY = registry
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", multimodal_module)
    monkeypatch.setitem(sys.modules, "vllm.multimodal.video", video_module)
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "nemotron_vl")

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    actual_frames, actual_metadata = utils._load_video_frames_vllm_with_metadata(
        str(video_path), num_frames=32, temporal_patch_size=2
    )

    assert actual_frames is frames
    assert actual_metadata["frames_indices"] == list(range(29))


def test_multimodal_cache_reset_defaults_to_safe_invalidation():
    assert should_reset_mm_cache_after_refit(
        {"vllm_kwargs": {"limit_mm_per_prompt": {"video": 1}}}
    )
    assert not should_reset_mm_cache_after_refit({})


def test_multimodal_cache_reset_honors_frozen_encoder_override():
    config = {
        "vllm_cfg": {"reset_mm_cache_after_refit": False},
        "vllm_kwargs": {"limit_mm_per_prompt": {"video": 1}},
    }

    assert not should_reset_mm_cache_after_refit(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_mm_cache_after_refit", [True, False])
@pytest.mark.parametrize(
    "method_name",
    [
        "update_weights_from_collective_async",
        "update_weights_via_ipc_zmq_async",
    ],
)
async def test_async_refit_honors_multimodal_cache_reset_policy(
    reset_mm_cache_after_refit,
    method_name,
):
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "vllm_cfg": {
            "async_engine": True,
            "reset_mm_cache_after_refit": reset_mm_cache_after_refit,
        }
    }
    worker.llm = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=[True]),
        reset_mm_cache=AsyncMock(),
        reset_encoder_cache=AsyncMock(),
    )

    assert await getattr(worker, method_name)()
    worker.llm.collective_rpc.assert_awaited_once_with(
        method_name.removesuffix("_async"),
        args=(reset_mm_cache_after_refit,),
    )
    if reset_mm_cache_after_refit:
        worker.llm.reset_mm_cache.assert_awaited_once_with()
        worker.llm.reset_encoder_cache.assert_awaited_once_with()
    else:
        worker.llm.reset_mm_cache.assert_not_awaited()
        worker.llm.reset_encoder_cache.assert_not_awaited()

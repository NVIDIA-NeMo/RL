# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the bounded resize with the actual installed RADIO implementation."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.radio import ViTPatchGenerator

from nemo_rl.models.generation.vllm.radio_cpe import (
    cropped_cpe_position_embeddings,
    install_radio_cpe_crop,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dimensions", [(1, 37), (37, 1), (7, 37), (37, 7), (8, 37), (37, 8), (8, 8)]
)
def test_crop_matches_original_square_coordinates(dtype, dimensions):
    position = torch.linspace(-1, 1, 8 * 8 * 5, dtype=dtype).reshape(1, 64, 5)
    model = SimpleNamespace(pos_embed=position, num_rows=8, num_cols=8, cpe_mode=True)
    # The original method is the reference; neither its interpolation nor the
    # downstream metrics are replaced by a looser checker.
    original = ViTPatchGenerator._get_pos_embeddings
    expected = original(model, 1, dimensions)
    actual = cropped_cpe_position_embeddings(position, 8, 8, dimensions)
    assert torch.equal(actual, expected)
    assert actual.dtype == dtype
    assert actual.shape == expected.shape


def test_crop_storage_does_not_keep_the_unused_square():
    position = torch.linspace(-1, 1, 8 * 8 * 5).reshape(1, 64, 5)
    actual = cropped_cpe_position_embeddings(position, 8, 8, (8, 257))
    assert actual.untyped_storage().nbytes() <= 9 * 257 * 5 * 4
    assert actual.untyped_storage().nbytes() < 257 * 257 * 5 * 4


def test_runtime_installation_is_local_idempotent_and_preserves_other_modes(
    monkeypatch,
):
    original = ViTPatchGenerator._get_pos_embeddings
    source = Path(__import__(ViTPatchGenerator.__module__, fromlist=["x"]).__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(ViTPatchGenerator, "_get_pos_embeddings", original)
    install_radio_cpe_crop()
    installed = ViTPatchGenerator._get_pos_embeddings
    install_radio_cpe_crop()
    assert ViTPatchGenerator._get_pos_embeddings is installed
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    position = torch.linspace(-1, 1, 8 * 8 * 5).reshape(1, 64, 5)
    for cpe_mode, dims in (
        (True, (8, 37)),
        (True, (8, 8)),
        (True, (5, 5)),
        (False, (5, 13)),
    ):
        model = SimpleNamespace(
            pos_embed=position, num_rows=8, num_cols=8, cpe_mode=cpe_mode
        )
        assert torch.equal(installed(model, 1, dims), original(model, 1, dims))


def test_native_worker_mixin_installs_before_constructor(monkeypatch):
    from vllm.v1.worker.gpu_worker import Worker

    from nemo_rl.models.generation.vllm import radio_cpe
    from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension

    installed = []
    monkeypatch.setattr(
        radio_cpe, "install_radio_cpe_crop", lambda: installed.append(True)
    )
    # Match vLLM WorkerWrapperBase.init_worker's actual mixin injection. Invoke
    # only __new__; native device/model initialization is deliberately absent.
    original_bases = Worker.__bases__
    try:
        if VllmInternalWorkerExtension not in Worker.__bases__:
            Worker.__bases__ = Worker.__bases__ + (VllmInternalWorkerExtension,)
        for vision in ({"model_type": "radio"}, SimpleNamespace(model_type="radio")):
            config = SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(vision_config=vision)
                )
            )
            instance = Worker.__new__(Worker, vllm_config=config)
            assert type(instance) is Worker
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace())
        )
        Worker.__new__(Worker, vllm_config=config)
        assert installed == [True, True]
        assert not torch.cuda.is_initialized()
    finally:
        Worker.__bases__ = original_bases

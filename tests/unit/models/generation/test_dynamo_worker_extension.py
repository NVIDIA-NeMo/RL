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
"""The worker extension managed Dynamo engines load for NeMo RL refit."""

import pytest

pytest.importorskip("vllm")

from nemo_rl.models.generation.vllm.dynamo_worker_extension import (  # noqa: E402
    DynamoWorkerExtension,
)
from nemo_rl.utils.staged_payload import encode_staged_payload  # noqa: E402

pytestmark = pytest.mark.vllm


def _stage(ext: DynamoWorkerExtension, payload, *, skip: int | None = None):
    staged = encode_staged_payload(payload)
    for index, data in enumerate(staged.chunks):
        if index != skip:
            ext.nrl_stage_chunk(staged.key, index, len(staged.chunks), data)
    return staged


def test_staged_reshard_plan_reaches_prepare_unchanged() -> None:
    ext = DynamoWorkerExtension()  # no __init__, as when vLLM mixes it in
    received = []
    ext.prepare_nccl_reshard_refit_info = received.append
    plan = {"per_layer_params": {"layers.0": [{"name": "w"}]}, "misc_meta": {}}

    staged = _stage(ext, plan)
    ext.nrl_prepare_nccl_reshard_refit_info_from_stage(staged.key, staged.sha256)

    assert received == [plan]
    # The buffer is consumed: the same key cannot be prepared twice.
    with pytest.raises(RuntimeError, match="No payload is staged"):
        ext.nrl_prepare_nccl_reshard_refit_info_from_stage(staged.key, staged.sha256)


def test_a_new_key_discards_a_partially_staged_payload() -> None:
    ext = DynamoWorkerExtension()
    stale = _stage(ext, {"old": True})
    fresh = _stage(ext, {"new": True})
    received = []
    ext.prepare_nccl_reshard_refit_info = received.append

    with pytest.raises(RuntimeError, match="No payload is staged"):
        ext.nrl_prepare_nccl_reshard_refit_info_from_stage(stale.key, stale.sha256)
    ext.nrl_prepare_nccl_reshard_refit_info_from_stage(fresh.key, fresh.sha256)
    assert received == [{"new": True}]


def test_a_missing_chunk_or_bad_digest_is_refused() -> None:
    ext = DynamoWorkerExtension()
    ext.prepare_nccl_reshard_refit_info = lambda info: None
    payload = {"blob": "x" * (3 << 20), "salt": list(range(1000))}

    staged = _stage(ext, payload, skip=0)
    if len(staged.chunks) > 1:
        with pytest.raises(RuntimeError, match="missing chunks"):
            ext.nrl_prepare_nccl_reshard_refit_info_from_stage(
                staged.key, staged.sha256
            )

    staged = _stage(ext, {"a": 1})
    with pytest.raises(ValueError, match="digest"):
        ext.nrl_prepare_nccl_reshard_refit_info_from_stage(staged.key, "0" * 64)


def test_a_chunk_that_does_not_fit_the_payload_is_refused() -> None:
    ext = DynamoWorkerExtension()
    ext.nrl_stage_chunk("k", 0, 2, "a")
    with pytest.raises(ValueError, match="does not fit"):
        ext.nrl_stage_chunk("k", 2, 2, "b")
    with pytest.raises(ValueError, match="does not fit"):
        ext.nrl_stage_chunk("k", 0, 3, "c")


def test_a_refit_that_reports_failure_raises() -> None:
    # Dynamo discards the RPC's return value, so False would read as success.
    ext = DynamoWorkerExtension()
    seen = []
    ext.nccl_reshard_refit = lambda refit_timeout_s: seen.append(refit_timeout_s)
    with pytest.raises(RuntimeError, match="reported failure"):
        ext.nrl_nccl_reshard_refit(30.0)
    ext.nccl_reshard_refit = lambda refit_timeout_s: True
    assert ext.nrl_nccl_reshard_refit(None) is True
    assert seen == [30.0]


def test_new_methods_do_not_shadow_the_vllm_worker() -> None:
    # vLLM refuses an extension that shadows a Worker attribute.
    from vllm.v1.worker.gpu_worker import Worker

    added = {name for name in vars(DynamoWorkerExtension) if not name.startswith("__")}
    assert "nrl_stage_chunk" in added
    assert not added & set(dir(Worker))


def test_staged_collective_metadata_is_restored_in_order_with_dtypes() -> None:
    import torch

    ext = DynamoWorkerExtension()
    received = []
    ext.prepare_refit_info = received.append
    entries = [["z.weight", [4, 8], "bfloat16"], ["a.bias", [8], "float32"]]

    staged = _stage(ext, entries)
    ext.nrl_prepare_refit_info_from_stage(staged.key, staged.sha256)

    assert len(received) == 1
    assert list(received[0].items()) == [
        ("z.weight", (torch.Size([4, 8]), torch.bfloat16)),
        ("a.bias", (torch.Size([8]), torch.float32)),
    ]


@pytest.mark.parametrize("dtype_name", ["not_a_dtype", "Tensor"])
def test_staged_collective_metadata_rejects_unknown_dtypes(dtype_name) -> None:
    ext = DynamoWorkerExtension()
    ext.prepare_refit_info = lambda info: None
    staged = _stage(ext, [["w", [2], dtype_name]])
    with pytest.raises(ValueError, match="unknown dtype"):
        ext.nrl_prepare_refit_info_from_stage(staged.key, staged.sha256)


def test_a_collective_refit_that_reports_failure_raises() -> None:
    ext = DynamoWorkerExtension()
    calls = []
    ext.update_weights_from_collective = lambda **kwargs: calls.append(kwargs)
    with pytest.raises(RuntimeError, match="reported failure"):
        ext.nrl_update_weights_from_collective(10.0, True)
    assert calls == [{"refit_timeout_s": 10.0, "refit_with_reload_api": True}]
    ext.update_weights_from_collective = lambda **kwargs: True
    assert ext.nrl_update_weights_from_collective(None, False) is True


@pytest.fixture()
def fp8_module(monkeypatch):
    """The FP8 module with its process-global state restored afterwards."""
    import types

    from nemo_rl.models.generation.vllm.quantization import fp8

    monkeypatch.setattr(fp8, "global_fp8_config", None)
    monkeypatch.setattr(fp8, "fp8_state", fp8.FP8State())
    monkeypatch.setattr(fp8, "fp8_patches_applied", False)
    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    return fp8


def test_build_fp8_config_has_no_process_side_effects(fp8_module, monkeypatch) -> None:
    import os

    fp8 = fp8_module
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    run_engine_core = fp8.EngineCoreProc.run_engine_core
    vllm_cfg = {
        "precision": "fp8",
        "kv_cache_dtype": "auto",
        "use_deep_gemm": True,
        "async_engine": True,
    }

    config, kwargs = fp8.build_fp8_config(vllm_cfg, "dummy-model", 2)

    assert config.model_parallel_size == 2 and config.use_fp8_weights
    assert kwargs["quantization"] == "fp8"
    assert fp8.global_fp8_config is None
    assert fp8.EngineCoreProc.run_engine_core is run_engine_core
    assert "VLLM_USE_DEEP_GEMM" not in os.environ
    assert fp8.fp8_engine_env(vllm_cfg, config) == {
        "VLLM_USE_DEEP_GEMM": "1",
        "VLLM_USE_DEEP_GEMM_E8M0": "0",
    }


def test_dynamo_fp8_engine_args_match_the_vllm_backend(fp8_module) -> None:
    import dataclasses
    import json

    from nemo_rl.models.generation.dynamo.dynamo_worker import _fp8_engine_args
    from nemo_rl.models.generation.vllm.quantization.fp8_kwargs import (
        FP8_CONFIG_ENV_VAR,
    )

    vllm_cfg = {"precision": "fp8", "kv_cache_dtype": "auto", "is_mx": True}
    user_kwargs = {"hf_overrides": {"max_position_embeddings": 4096}}

    cfg, kwargs, defaults, fp8_env = _fp8_engine_args(
        "dummy-model", vllm_cfg, user_kwargs, model_parallel_size=2
    )

    expected_config, expected_kwargs = fp8_module.build_fp8_config(
        vllm_cfg, "dummy-model", 2
    )
    # vLLM's working dtype; the quantization config is what makes weights FP8.
    assert cfg["precision"] == "bfloat16"
    assert kwargs["quantization"] == "fp8"
    # kv_cache_dtype already reaches dynamo.vllm from vllm_cfg.
    assert "kv_cache_dtype" not in kwargs
    assert kwargs["hf_overrides"]["max_position_embeddings"] == 4096
    assert (
        kwargs["hf_overrides"]["quantization_config"]
        == expected_kwargs["hf_overrides"]["quantization_config"]
    )
    assert user_kwargs == {"hf_overrides": {"max_position_embeddings": 4096}}
    assert defaults == {}
    assert json.loads(fp8_env[FP8_CONFIG_ENV_VAR]) == dataclasses.asdict(
        expected_config
    )


def test_the_extension_applies_fp8_patches_from_its_environment(
    fp8_module, monkeypatch
) -> None:
    import dataclasses
    import json

    from nemo_rl.models.generation.vllm import dynamo_worker_extension
    from nemo_rl.models.generation.vllm.quantization.fp8_kwargs import (
        FP8_CONFIG_ENV_VAR,
    )

    fp8 = fp8_module
    applied = []
    monkeypatch.setattr(
        fp8, "apply_fp8_patches", lambda self, config: applied.append(config)
    )
    config = fp8.FP8Config(model_parallel_size=2, is_mx=True)

    monkeypatch.delenv(FP8_CONFIG_ENV_VAR, raising=False)
    dynamo_worker_extension._apply_fp8_patches_from_env()
    assert applied == []

    monkeypatch.setenv(FP8_CONFIG_ENV_VAR, json.dumps(dataclasses.asdict(config)))
    dynamo_worker_extension._apply_fp8_patches_from_env()
    assert applied == [config]

    monkeypatch.setattr(fp8, "fp8_patches_applied", True)
    dynamo_worker_extension._apply_fp8_patches_from_env()
    assert applied == [config]

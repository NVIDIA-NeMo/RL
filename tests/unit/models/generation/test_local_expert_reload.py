# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.vllm.local_expert_reload import (
    LocalBf16ExpertReload,
    LocalExpertBinding,
    copy_local_bf16_expert,
)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_local_projection_replay(gated: bool, padded: bool, reverse: bool) -> None:
    width, hidden, experts = 8, 16, 3
    padded_width, padded_hidden = (12, 20) if padded else (width, hidden)
    w13 = torch.full(
        (experts, padded_width * (2 if gated else 1), padded_hidden),
        -1,
        dtype=torch.bfloat16,
    )
    w2 = torch.full((experts, padded_hidden, padded_width), -1, dtype=torch.bfloat16)
    projections = (
        ["gate_proj", "up_proj", "down_proj"] if gated else ["up_proj", "down_proj"]
    )
    if reverse:
        projections.reverse()
    for update in (1, 2, 3):
        for projection in projections:
            is_down = projection == "down_proj"
            shape = (experts, hidden, width) if is_down else (experts, width, hidden)
            value = (
                update * 10 + {"gate_proj": 1, "up_proj": 2, "down_proj": 3}[projection]
            )
            weight = torch.full(shape, value, dtype=torch.bfloat16)
            copy_local_bf16_expert(
                w2 if is_down else w13, weight, projection=projection, gated=gated
            )
        expected13 = torch.zeros_like(w13)
        if gated:
            expected13[:, :width, :hidden] = update * 10 + 1
        start = padded_width if gated else 0
        expected13[:, start : start + width, :hidden] = update * 10 + 2
        expected2 = torch.zeros_like(w2)
        expected2[:, :hidden, :width] = update * 10 + 3
        torch.testing.assert_close(w13, expected13, rtol=0, atol=0)
        torch.testing.assert_close(w2, expected2, rtol=0, atol=0)


def test_invalid_payload_does_not_modify_destination() -> None:
    target = torch.full((2, 16, 8), 7, dtype=torch.bfloat16)
    before = target.clone()
    with pytest.raises(ValueError, match="exceeds"):
        copy_local_bf16_expert(
            target,
            torch.ones((2, 9, 8), dtype=torch.bfloat16),
            projection="up_proj",
            gated=True,
        )
    torch.testing.assert_close(target, before, rtol=0, atol=0)


def test_meta_pass_does_not_skip_real_padding_initialization() -> None:
    meta = torch.empty((2, 24, 20), dtype=torch.bfloat16, device="meta")
    payload = torch.ones((2, 8, 16), dtype=torch.bfloat16)
    copy_local_bf16_expert(meta, payload, projection="up_proj", gated=True)
    actual = torch.full((2, 24, 20), -1, dtype=torch.bfloat16)
    copy_local_bf16_expert(actual, payload, projection="up_proj", gated=True)
    assert torch.all(actual[:, :12] == -1)
    assert torch.all(actual[:, 20:] == 0)
    assert torch.all(actual[:, 12:20, 16:] == 0)


def test_vllm_counter_counts_only_payload() -> None:
    pytest.importorskip("vllm")
    from vllm.model_executor.model_loader.reload.meta import get_numel_loaded

    meta = torch.empty((2, 24, 20), dtype=torch.bfloat16, device="meta")
    payload = torch.ones((2, 8, 16), dtype=torch.bfloat16)
    bound = inspect.signature(copy_local_bf16_expert).bind(
        meta, payload, projection="up_proj", gated=True
    )
    count, _ = get_numel_loaded(copy_local_bf16_expert, bound)
    assert count == payload.numel()


def _checkpoint_model(gated: bool, padded: bool) -> torch.nn.Module:
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.moe_config = SimpleNamespace(is_act_and_mul=gated)
    width, hidden = (12, 20) if padded else (8, 16)
    model.experts.w13 = torch.nn.Parameter(
        torch.full((2, width * (2 if gated else 1), hidden), -1, dtype=torch.bfloat16),
        requires_grad=False,
    )
    model.experts.w2 = torch.nn.Parameter(
        torch.full((2, hidden, width), -1, dtype=torch.bfloat16), requires_grad=False
    )
    return model


def _bindings(gated: bool) -> dict[str, LocalExpertBinding]:
    result = {
        "up": LocalExpertBinding("experts.w13", "up_proj", (2, 8, 16)),
        "down": LocalExpertBinding("experts.w2", "down_proj", (2, 16, 8)),
    }
    if gated:
        result["gate"] = LocalExpertBinding("experts.w13", "gate_proj", (2, 8, 16))
    return result


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_installed_online_reload_preserves_values_and_storage(
    gated: bool, padded: bool
) -> None:
    pytest.importorskip("vllm")
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    model = _checkpoint_model(gated, padded)
    record_metadata_for_reloading(model)
    bindings = _bindings(gated)
    originals = dict(model.named_parameters())
    for update in (1, 2, 3):
        reload = LocalBf16ExpertReload(model, bindings)
        initialize_layerwise_reload(model)
        expected13 = torch.zeros_like(originals["experts.w13"])
        expected2 = torch.zeros_like(originals["experts.w2"])
        for name, binding in bindings.items():
            value = update * 10 + {"gate": 1, "up": 2, "down": 3}[name]
            payload = torch.full(binding.local_shape, value, dtype=torch.bfloat16)
            reload.load(name, payload)
            payload.zero_()  # Deferred loading must not retain a borrowed receive buffer.
            if name == "down":
                expected2[:, :16, :8] = value
            else:
                start = expected13.shape[1] // 2 if name == "up" and gated else 0
                expected13[:, start : start + 8, :16] = value
        reload.require_complete()
        finalize_layerwise_reload(model, None)
        reload.verify_runtime_storage()
        torch.testing.assert_close(model.experts.w13, expected13, rtol=0, atol=0)
        torch.testing.assert_close(model.experts.w2, expected2, rtol=0, atol=0)


def test_missing_component_poisoning() -> None:
    reload = LocalBf16ExpertReload(_checkpoint_model(True, True), _bindings(True))
    with pytest.raises(RuntimeError, match="Missing local expert"):
        reload.require_complete()
    with pytest.raises(RuntimeError, match="unusable"):
        reload.require_complete()


def test_load_before_initialization_rejected() -> None:
    reload = LocalBf16ExpertReload(_checkpoint_model(True, True), _bindings(True))
    with pytest.raises(RuntimeError, match="active checkpoint storage"):
        reload.load("up", torch.ones((2, 8, 16), dtype=torch.bfloat16))


def test_duplicate_destination_rejected() -> None:
    bindings = _bindings(True)
    bindings["alias"] = bindings["up"]
    with pytest.raises(ValueError, match="Duplicate local expert destination"):
        LocalBf16ExpertReload(_checkpoint_model(True, True), bindings)


@pytest.mark.parametrize("bad_shape", [False, True])
def test_incomplete_or_inconsistent_plan_rejected(bad_shape: bool) -> None:
    bindings = _bindings(True)
    if bad_shape:
        bindings["down"] = LocalExpertBinding("experts.w2", "down_proj", (2, 16, 7))
    else:
        del bindings["gate"]
    with pytest.raises(ValueError, match="plan"):
        LocalBf16ExpertReload(_checkpoint_model(True, True), bindings)


def test_duplicate_load_poisoning() -> None:
    pytest.importorskip("vllm")
    from vllm.model_executor.model_loader.reload.layerwise import (
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    model = _checkpoint_model(True, True)
    record_metadata_for_reloading(model)
    reload = LocalBf16ExpertReload(model, _bindings(True))
    initialize_layerwise_reload(model)
    payload = torch.ones((2, 8, 16), dtype=torch.bfloat16)
    reload.load("up", payload)
    with pytest.raises(ValueError, match="Duplicate local expert component"):
        reload.load("up", payload)
    with pytest.raises(RuntimeError, match="unusable"):
        reload.load("gate", payload)

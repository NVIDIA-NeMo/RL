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

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from nemo_rl.modelopt.calibration_artifact import load_nvfp4_calibration
from nemo_rl.modelopt.models.generation import nvfp4_refit

_GATE = "model.layers.0.mlp.experts.3.gate_proj.weight"
_UP = "model.layers.0.mlp.experts.3.up_proj.weight"
_DOWN = "model.layers.0.mlp.experts.3.down_proj.weight"


def _write_calibration(path: Path) -> None:
    save_file(
        {_GATE: torch.tensor(10.0), _UP: torch.tensor(20.0)},
        str(path),
        metadata={
            "model_id": json.dumps("Qwen/Qwen3-30B-A3B"),
            "model_revision": json.dumps("revision-1"),
            "quant_cfg": json.dumps("NVFP4_TEST_CFG"),
        },
    )


def test_calibration_artifact_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "calibration.safetensors"
    _write_calibration(path)

    calibration = load_nvfp4_calibration(
        path,
        model_id="Qwen/Qwen3-30B-A3B",
        model_revision="revision-1",
        quant_cfg="NVFP4_TEST_CFG",
        expected_projection_names={_GATE, _UP},
    )

    assert {name: value.item() for name, value in calibration.input_amax.items()} == {
        _GATE: 10.0,
        _UP: 20.0,
    }


def test_calibration_artifact_rejects_identity_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "calibration.safetensors"
    _write_calibration(path)

    with pytest.raises(ValueError, match="model_revision"):
        load_nvfp4_calibration(
            path,
            model_id="Qwen/Qwen3-30B-A3B",
            model_revision="revision-2",
            quant_cfg="NVFP4_TEST_CFG",
        )


def test_calibration_artifact_rejects_projection_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "calibration.safetensors"
    _write_calibration(path)

    with pytest.raises(ValueError, match="projection names do not match"):
        load_nvfp4_calibration(
            path,
            model_id="Qwen/Qwen3-30B-A3B",
            model_revision="revision-1",
            quant_cfg="NVFP4_TEST_CFG",
            expected_projection_names={_GATE},
        )


def test_calibration_artifact_requires_digest_for_file_config(tmp_path: Path) -> None:
    quant_cfg = tmp_path / "nvfp4.yaml"
    quant_cfg.write_text("quant_cfg: []\n")
    artifact_path = tmp_path / "calibration.safetensors"
    save_file(
        {_GATE: torch.tensor(10.0)},
        str(artifact_path),
        metadata={
            "model_id": json.dumps("Qwen/Qwen3-30B-A3B"),
            "model_revision": json.dumps("revision-1"),
            "quant_cfg": json.dumps(str(quant_cfg)),
        },
    )

    with pytest.raises(ValueError, match="quant_cfg_sha256.*required"):
        load_nvfp4_calibration(
            artifact_path,
            model_id="Qwen/Qwen3-30B-A3B",
            model_revision="revision-1",
            quant_cfg=str(quant_cfg),
        )


def _recording_exporter(
    calls: list[tuple[str, Any]],
) -> Callable[[str, torch.Tensor, Any], Iterator[tuple[str, torch.Tensor]]]:
    def export(
        name: str, weight: torch.Tensor, meta: Any
    ) -> Iterator[tuple[str, torch.Tensor]]:
        calls.append((name, meta))
        yield name, weight

    return export


@pytest.mark.parametrize(
    ("mode", "calibration_values", "expected_export_mode", "expected_amaxes"),
    [
        ("w4a16", None, "w4a16_nvfp4", (None, None)),
        ("w4a4", (10.0, 20.0), "nvfp4", (10.0, 20.0)),
    ],
)
def test_serialize_routed_gate_up_uses_mode_and_named_calibration(
    monkeypatch: pytest.MonkeyPatch,
    mode: nvfp4_refit.NVFP4RefitMode,
    calibration_values: tuple[float, float] | None,
    expected_export_mode: str,
    expected_amaxes: tuple[float | None, float | None],
) -> None:
    calls: list[tuple[str, Any]] = []
    requested_modes: list[str] = []

    def get_exporter(quant_mode: str) -> tuple[str, Callable[..., Any]]:
        requested_modes.append(quant_mode)
        return f"format-{quant_mode}", _recording_exporter(calls)

    monkeypatch.setattr(nvfp4_refit, "get_modelopt_quant_exporter", get_exporter)
    monkeypatch.setattr(
        nvfp4_refit,
        "compute_nvfp4_input_scale",
        lambda input_amax: input_amax,
    )
    calibration = None
    if calibration_values is not None:
        calibration = nvfp4_refit.NVFP4Calibration(
            input_amax={
                _GATE: torch.tensor(calibration_values[0]),
                _UP: torch.tensor(calibration_values[1]),
            }
        )

    serialized = nvfp4_refit.serialize_bf16_nvfp4_group(
        {
            _GATE: torch.ones((32, 16), dtype=torch.bfloat16),
            _UP: torch.ones((32, 16), dtype=torch.bfloat16),
        },
        mode=mode,
        calibration=calibration,
    )

    observed_amaxes = tuple(
        None if meta.input_amax is None else meta.input_amax.item() for _, meta in calls
    )
    assert (
        requested_modes,
        [name for name, _ in serialized],
        observed_amaxes,
    ) == ([expected_export_mode], [_GATE, _UP], expected_amaxes)


def test_serialize_gate_up_shares_largest_weight_amax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        nvfp4_refit,
        "get_modelopt_quant_exporter",
        lambda _mode: ("w4a16", _recording_exporter(calls)),
    )

    nvfp4_refit.serialize_bf16_nvfp4_group(
        {
            _GATE: torch.full((32, 16), 2.0, dtype=torch.bfloat16),
            _UP: torch.full((32, 16), 3.0, dtype=torch.bfloat16),
        },
        mode="w4a16",
        calibration=None,
    )

    assert [meta.weight_amax.item() for _, meta in calls] == [3.0, 3.0]


@pytest.mark.parametrize("name", [_UP, _DOWN])
@pytest.mark.parametrize("mode", ["w4a16", "w4a4"])
def test_serialize_non_gated_group_preserves_original_name(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    mode: nvfp4_refit.NVFP4RefitMode,
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        nvfp4_refit,
        "get_modelopt_quant_exporter",
        lambda _mode: ("nvfp4", _recording_exporter(calls)),
    )
    monkeypatch.setattr(
        nvfp4_refit,
        "compute_nvfp4_input_scale",
        lambda input_amax: input_amax,
    )
    calibration = (
        nvfp4_refit.NVFP4Calibration(input_amax={name: torch.tensor(20.0)})
        if mode == "w4a4"
        else None
    )

    serialized = nvfp4_refit.serialize_bf16_nvfp4_group(
        {name: torch.ones((32, 16), dtype=torch.bfloat16)},
        mode=mode,
        calibration=calibration,
        projection_family="non_gated",
    )

    assert [serialized_name for serialized_name, _ in serialized] == [name]
    assert [called_name for called_name, _ in calls] == [name]


def test_resolve_routed_expert_families_accepts_non_gated_experts() -> None:
    names = {
        f"model.layers.0.mlp.experts.{expert}.{projection}_proj.weight"
        for expert in range(2)
        for projection in ("up", "down")
    }

    assert nvfp4_refit.resolve_nvfp4_routed_expert_families(names) == {
        "model.layers.0.mlp.experts": "non_gated"
    }


def test_resolve_routed_expert_families_rejects_mixed_family_per_prefix() -> None:
    names = {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    }

    with pytest.raises(ValueError, match="mixes gated and non-gated"):
        nvfp4_refit.resolve_nvfp4_routed_expert_families(names)


def test_resolve_routed_expert_families_rejects_incomplete_expert() -> None:
    with pytest.raises(ValueError, match="incomplete routed-expert family"):
        nvfp4_refit.resolve_nvfp4_routed_expert_families({_UP})


def test_validate_routed_expert_selection_rejects_ignored_gate() -> None:
    available_names = {_GATE, _UP, _DOWN}

    with pytest.raises(ValueError, match="partially selects gated"):
        nvfp4_refit.validate_nvfp4_routed_expert_selection(
            {_UP, _DOWN},
            available_names=available_names,
        )


def test_validate_routed_expert_selection_accepts_non_gated_family() -> None:
    assert nvfp4_refit.validate_nvfp4_routed_expert_selection(
        {_UP, _DOWN},
        available_names={_UP, _DOWN},
    ) == {"model.layers.0.mlp.experts": "non_gated"}


def test_serialize_rejects_incomplete_routed_gate_up_group() -> None:
    with pytest.raises(ValueError, match=r"not complete: missing .*up_proj\.weight"):
        nvfp4_refit.serialize_bf16_nvfp4_group(
            {_GATE: torch.ones((32, 16), dtype=torch.bfloat16)},
            mode="w4a16",
            calibration=None,
        )


def test_serialize_rejects_non_expert_projection() -> None:
    with pytest.raises(ValueError, match="routed-expert"):
        nvfp4_refit.serialize_bf16_nvfp4_group(
            {"model.layers.0.self_attn.q_proj.weight": torch.ones(16, 16)},
            mode="w4a16",
            calibration=None,
        )


def test_serialize_w4a4_requires_named_calibration() -> None:
    with pytest.raises(ValueError, match="Missing input amax.*W4A4"):
        nvfp4_refit.serialize_bf16_nvfp4_group(
            {
                _GATE: torch.ones((32, 16), dtype=torch.bfloat16),
                _UP: torch.ones((32, 16), dtype=torch.bfloat16),
            },
            mode="w4a4",
            calibration=None,
        )


@pytest.mark.parametrize(
    "projection",
    ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"],
)
def test_bf16_nvfp4_receiver_rejects_qkvo_scope(projection: str) -> None:
    backend = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_backend",
    )
    name = f"model.layers.0.self_attn.{projection}.weight"

    with pytest.raises(ValueError, match="supports routed experts only"):
        backend._classify_bf16_routed_experts(
            {name: ((32, 16), torch.bfloat16)},
            ignore_patterns=[],
        )


def test_bf16_nvfp4_receiver_rejects_unignored_projection_scope() -> None:
    backend = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_backend",
    )

    with pytest.raises(ValueError, match="supports routed experts only"):
        backend._classify_bf16_routed_experts(
            {
                "backbone.layers.0.mixer.in_proj.weight": (
                    (32, 16),
                    torch.bfloat16,
                )
            },
            ignore_patterns=[],
        )


def test_bf16_nvfp4_receiver_rejects_pre_grouped_experts() -> None:
    backend = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_backend",
    )

    with pytest.raises(ValueError, match="pre-grouped routed experts"):
        backend._classify_bf16_routed_experts(
            {
                "model.layers.0.mlp.experts.gate_up_proj": (
                    (8, 64, 16),
                    torch.bfloat16,
                ),
                "model.layers.0.mlp.experts.down_proj": (
                    (8, 16, 32),
                    torch.bfloat16,
                ),
            },
            ignore_patterns=[],
        )


def test_bf16_nvfp4_receiver_ignores_embedding_weight() -> None:
    backend = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_backend",
    )

    assert not backend._classify_bf16_routed_experts(
        {
            "model.embed_tokens.weight": (
                (1024, 256),
                torch.bfloat16,
            )
        },
        ignore_patterns=[],
    )


def test_bf16_nvfp4_receiver_accepts_non_gated_experts() -> None:
    backend = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_backend",
    )
    state_dict_info = {
        f"model.layers.0.mlp.experts.{expert}.{projection}_proj.weight": (
            (32, 16),
            torch.bfloat16,
        )
        for expert in range(2)
        for projection in ("up", "down")
    }

    assert backend._classify_bf16_routed_experts(
        state_dict_info,
        ignore_patterns=[],
    ) == frozenset(state_dict_info)


def test_non_gated_receiver_builds_up_only_w13_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_backend",
    )
    prefix = "model.layers.0.mlp.experts"
    grouped_up = f"{prefix}.up_proj.weight"
    grouped_down = f"{prefix}.down_proj.weight"
    expert_names = {
        f"{prefix}.{expert}.{projection}_proj.weight"
        for expert in range(2)
        for projection in ("up", "down")
    }
    base_map = backend.HFToLocalParamMap(
        specs={
            grouped_up: backend.LocalParamSpec(base=torch.empty(0)),
            grouped_down: backend.LocalParamSpec(base=torch.empty(0)),
        }
    )
    monkeypatch.setattr(
        backend.VllmInternalWorkerExtension,
        "build_hf_to_local_param_map",
        lambda _self, _refit_info: base_map,
    )
    monkeypatch.setattr(
        backend.VllmQuantInternalWorkerExtension,
        "_is_real_quant_model",
        lambda _self: True,
    )

    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension.device = torch.device("cpu")
    extension._nrl_bf16_nvfp4_names = frozenset(expert_names)
    extension._nrl_bf16_nvfp4_projection_families = {prefix: "non_gated"}
    mesh = SimpleNamespace(mesh=torch.empty(1))
    placement = SimpleNamespace(dim=None)
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": grouped_up,
                    "global_shape": (2, 32, 16),
                    "dst_mesh_info": mesh,
                    "dst_placements": (placement,),
                },
                {
                    "name": grouped_down,
                    "global_shape": (2, 16, 32),
                    "dst_mesh_info": mesh,
                    "dst_placements": (placement,),
                },
            ]
        },
    }

    result = extension.build_hf_to_local_param_map(refit_info)

    assert set(result.specs) == {grouped_up, grouped_down}
    assert extension._nrl_bf16_nvfp4_group_members == {
        f"{prefix}.w13": (grouped_up,),
        f"{prefix}.w2": (grouped_down,),
    }
    assert extension._nrl_bf16_nvfp4_group_families == {
        f"{prefix}.w13": "non_gated",
        f"{prefix}.w2": "non_gated",
    }


def test_non_gated_receiver_serializes_each_expert_without_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_backend",
    )
    prefix = "model.layers.0.mlp.experts"
    grouped_up = f"{prefix}.up_proj.weight"
    calls: list[tuple[dict[str, torch.Tensor], str]] = []
    loaded: list[tuple[str, torch.Tensor]] = []

    def serialize(tensors, *, mode, calibration, projection_family):
        assert mode == "w4a16"
        assert calibration is None
        calls.append((dict(tensors), projection_family))
        return list(tensors.items())

    monkeypatch.setattr(backend, "serialize_bf16_nvfp4_group", serialize)
    extension = object.__new__(backend.VllmQuantInternalWorkerExtension)
    extension._nrl_bf16_nvfp4_group_members = {f"{prefix}.w13": (grouped_up,)}
    extension._nrl_bf16_nvfp4_group_families = {f"{prefix}.w13": "non_gated"}
    extension._nrl_bf16_nvfp4_staging = {}
    extension._nrl_bf16_nvfp4_mode = "w4a16"
    extension._nrl_bf16_nvfp4_calibration = None
    extension._load_weights = loaded.extend

    weight = torch.arange(64, dtype=torch.bfloat16).reshape(2, 2, 16)
    extension._stage_bf16_nvfp4_group(
        completion_key=f"{prefix}.w13",
        grouped_name=grouped_up,
        weight=weight,
    )

    assert [family for _, family in calls] == ["non_gated", "non_gated"]
    assert [list(tensors) for tensors, _ in calls] == [
        [f"{prefix}.0.up_proj.weight"],
        [f"{prefix}.1.up_proj.weight"],
    ]
    assert [name for name, _ in loaded] == [
        f"{prefix}.0.up_proj.weight",
        f"{prefix}.1.up_proj.weight",
    ]


def test_w4a4_calibration_uses_resolved_revision_without_explicit_revision() -> None:
    backend = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_backend",
    )
    model_config = SimpleNamespace(
        model="Qwen/Qwen3-30B-A3B",
        revision=None,
        hf_config=SimpleNamespace(_commit_hash="resolved-commit"),
    )

    assert backend._vllm_calibration_provenance(model_config) == (
        "Qwen/Qwen3-30B-A3B",
        "resolved-commit",
    )


@pytest.mark.parametrize("mode", ["w4a4", "w4a16"])
def test_worker_propagates_calibration_only_for_w4a4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    worker = pytest.importorskip(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker",
    )
    from nemo_rl.modelopt import utils as modelopt_utils
    from nemo_rl.modelopt.models.generation import vllm_modelopt

    calibration_path = tmp_path / "calibration.safetensors"
    quant_cfg = "NVFP4_TEST_CFG"
    calibration_keys = (
        "VLLM_MODELOPT_CALIBRATION_PATH",
        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG",
    )
    for key in calibration_keys:
        monkeypatch.setenv(key, "/stale/value")
    monkeypatch.setattr(vllm_modelopt, "register_nemo_modelopt_nvfp4", lambda: None)
    monkeypatch.setattr(
        vllm_modelopt,
        "quantization_method_for_mode",
        lambda quant_mode: f"quant-{quant_mode}",
    )
    monkeypatch.setattr(
        modelopt_utils,
        "resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: mode,
    )
    monkeypatch.setattr(
        modelopt_utils,
        "build_vllm_modelopt_nvfp4_config",
        lambda **kwargs: kwargs,
    )

    worker._configure_quant_engine_kwargs(
        {
            "quant_cfg": quant_cfg,
            "real_quant": True,
            "real_quant_calibration_path": str(calibration_path),
        },
        {},
    )

    expected_values = (
        (str(calibration_path.resolve()), quant_cfg) if mode == "w4a4" else (None, None)
    )
    assert (
        tuple(os.environ.get(key) for key in calibration_keys),
        all(key in worker._EXTRA_ENV_VARS for key in calibration_keys),
    ) == (expected_values, True)

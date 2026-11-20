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

from typing import Any

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.mcore

_WEIGHTS = [
    ("fc.weight", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
    ("norm.weight", torch.ones(4, dtype=torch.bfloat16)),
]


def _refit_module() -> Any:
    from nemo_rl.models.megatron.draft import refit as draft_refit

    return draft_refit


def _patch_exporter(monkeypatch: pytest.MonkeyPatch, exporter: Any) -> None:
    import nemo_rl.models.megatron.draft as draft_pkg

    monkeypatch.setattr(draft_pkg, "export_eagle_weights_to_hf", exporter)


@pytest.fixture
def pp1(monkeypatch: pytest.MonkeyPatch) -> Any:
    draft_refit = _refit_module()
    monkeypatch.setattr(
        draft_refit.parallel_state,
        "get_pipeline_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(draft_refit, "broadcast_obj_from_pp_rank", lambda obj: obj)
    monkeypatch.setattr(dist, "get_rank", lambda *_args, **_kwargs: 0)
    return draft_refit


def test_pp1_yields_local_weights_with_prefix(
    pp1: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_exporter(monkeypatch, lambda _model: iter(_WEIGHTS))

    result = list(pp1.stream_draft_weights_for_refit(torch.nn.Linear(1, 1), None))

    assert [name for name, _ in result] == ["draft.fc.weight", "draft.norm.weight"]
    for (_, actual), (_, wanted) in zip(result, _WEIGHTS, strict=True):
        assert actual is wanted


def test_metadata_only_yields_meta_tensors(
    pp1: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_exporter(monkeypatch, lambda _model: iter(_WEIGHTS))

    result = list(
        pp1.stream_draft_weights_for_refit(
            torch.nn.Linear(1, 1), None, metadata_only=True
        )
    )

    assert [name for name, _ in result] == ["draft.fc.weight", "draft.norm.weight"]
    for (_, actual), (_, wanted) in zip(result, _WEIGHTS, strict=True):
        assert actual.device.type == "meta"
        assert actual.shape == wanted.shape and actual.dtype == wanted.dtype


def test_custom_exporter_replaces_eagle_default(
    pp1: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_default(_model: object) -> None:
        raise AssertionError("custom exporter must replace the EAGLE default")

    _patch_exporter(monkeypatch, fail_default)
    custom = [("block.weight", torch.zeros(2))]

    result = list(
        pp1.stream_draft_weights_for_refit(
            torch.nn.Linear(1, 1), None, export_weights=lambda _model: custom
        )
    )

    assert [name for name, _ in result] == ["draft.block.weight"]
    assert result[0][1] is custom[0][1]


def test_disabled_draft_without_model_yields_nothing(
    pp1: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_broadcast(_obj: object) -> None:
        raise AssertionError("no collective when draft refit is disabled")

    monkeypatch.setattr(pp1, "broadcast_obj_from_pp_rank", fail_broadcast)

    assert list(pp1.stream_draft_weights_for_refit(None, None)) == []
    assert list(pp1.stream_draft_weights_for_refit(None, {"enabled": False})) == []


def _raise_export(_model: object) -> None:
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("exporter", "code", "detail"),
    [
        (_raise_export, "EXPORTER_ERROR", "RuntimeError: boom"),
        (lambda _model: [], "MANIFEST_INVALID", "no tensors"),
        (lambda _model: [("a", 1.0)], "MANIFEST_INVALID", "non-tensor"),
        (
            lambda _model: [("a", torch.ones(1)), ("a", torch.ones(1))],
            "MANIFEST_INVALID",
            "duplicate",
        ),
    ],
)
def test_owner_export_failure_raises_coded_error(
    pp1: Any,
    monkeypatch: pytest.MonkeyPatch,
    exporter: Any,
    code: str,
    detail: str,
) -> None:
    _patch_exporter(monkeypatch, exporter)

    with pytest.raises(ValueError, match=rf"\[draft refit {code}\].*{detail}"):
        list(pp1.stream_draft_weights_for_refit(torch.nn.Linear(1, 1), None))


def test_check_draft_refit_metadata_accepts_matching_tensor() -> None:
    draft_refit = _refit_module()
    meta = {"draft.w": {"shape": [2, 3], "dtype": "torch.float32"}}

    draft_refit.check_draft_refit_metadata("draft.w", torch.zeros(2, 3), meta)


@pytest.mark.parametrize(
    "tensor",
    [torch.zeros(3, 2), torch.zeros(2, 3, dtype=torch.bfloat16)],
)
def test_check_draft_refit_metadata_rejects_drift(tensor: torch.Tensor) -> None:
    draft_refit = _refit_module()
    meta = {"draft.w": {"shape": [2, 3], "dtype": "torch.float32"}}

    with pytest.raises(ValueError, match="METADATA_DRIFT.*changed after prepare"):
        draft_refit.check_draft_refit_metadata("draft.w", tensor, meta)


def test_check_draft_refit_metadata_rejects_unknown_name() -> None:
    draft_refit = _refit_module()

    with pytest.raises(ValueError, match="METADATA_DRIFT.*not in the prepared"):
        draft_refit.check_draft_refit_metadata("draft.w", torch.zeros(1), {})


def _run_pp2_stream(rank: int, world_size: int) -> None:
    import nemo_rl.models.megatron.draft as draft_pkg
    from nemo_rl.models.megatron import pipeline_parallel

    draft_refit = _refit_module()
    assert world_size == 2
    pp_group = dist.new_group([0, 1], backend="gloo")
    # Spawned processes are discarded after the test, so plain patches are safe.
    for module in (draft_refit.parallel_state, pipeline_parallel):
        module.get_pipeline_model_parallel_world_size = lambda: 2
        module.get_pipeline_model_parallel_group = lambda: pp_group
    draft_refit.draft_refit_enabled = lambda _cfg: True

    owner_rank = 1
    draft_model = torch.nn.Linear(1, 1) if rank == owner_rank else None
    draft_pkg.export_eagle_weights_to_hf = lambda _model: iter(_WEIGHTS)

    for metadata_only in (True, False):
        result = list(
            draft_refit.stream_draft_weights_for_refit(
                draft_model, None, metadata_only=metadata_only
            )
        )
        assert [name for name, _ in result] == [f"draft.{n}" for n, _ in _WEIGHTS]
        for (_, actual), (_, wanted) in zip(result, _WEIGHTS, strict=True):
            assert actual.shape == wanted.shape and actual.dtype == wanted.dtype
            if metadata_only:
                assert actual.device.type == "meta"
            else:
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    draft_pkg.export_eagle_weights_to_hf = _raise_export
    with pytest.raises(ValueError, match="EXPORTER_ERROR"):
        list(draft_refit.stream_draft_weights_for_refit(draft_model, None))


def test_pp2_streams_owner_weights_to_every_lane_rank(
    distributed_test_runner: Any,
) -> None:
    distributed_test_runner(_run_pp2_stream, world_size=2, backend="gloo")

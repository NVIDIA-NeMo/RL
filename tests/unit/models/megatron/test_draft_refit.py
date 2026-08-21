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

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from nemo_rl.models.megatron.draft import utils as draft_utils


class _FalseyExporter:
    def __init__(self, value: list[tuple[str, torch.Tensor]]) -> None:
        self.value = value
        self.calls = 0

    def __bool__(self) -> bool:
        return False

    def __call__(self) -> list[tuple[str, torch.Tensor]]:
        self.calls += 1
        return self.value


@pytest.mark.mcore
def test_dtype_bucket_pp1_preserves_current_validated_export() -> None:
    calls = 0
    expected = [
        ("draft.bf16", torch.arange(4, dtype=torch.bfloat16)),
        ("draft.fp32", torch.arange(3, dtype=torch.float32)),
    ]

    def exporter() -> list[tuple[str, torch.Tensor]]:
        nonlocal calls
        calls += 1
        return expected

    result = draft_utils.broadcast_draft_weights_from_pp_owner(
        local_exporter=exporter,
        metadata_only=False,
    )

    assert calls == 1
    assert [name for name, _ in result] == [name for name, _ in expected]
    assert result[0][1] is expected[0][1]
    assert result[1][1] is expected[1][1]

    metadata = draft_utils.broadcast_draft_weights_from_pp_owner(
        local_exporter=lambda: expected,
        metadata_only=True,
    )
    assert all(tensor.device.type == "meta" for _, tensor in metadata)

    with pytest.raises(ValueError, match="MANIFEST_INVALID.*duplicate"):
        draft_utils.broadcast_draft_weights_from_pp_owner(
            local_exporter=lambda: [expected[0], expected[0]],
            metadata_only=False,
        )


def _lane_groups(rank: int) -> tuple[dist.ProcessGroup, dist.ProcessGroup, int]:
    pp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    cp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    cp_rank = rank % 2
    pp_stage = rank // 2
    return pp_groups[cp_rank], cp_groups[pp_stage], cp_rank


def _current_export(
    *,
    refit_step: int,
) -> list[tuple[str, torch.Tensor]]:
    device = torch.device("cuda", torch.cuda.current_device())
    return [
        (
            "draft.layers.0.norm.weight",
            torch.full((2,), refit_step, dtype=torch.bfloat16, device=device),
        ),
        (
            "draft.layers.0.proj.weight",
            torch.full((2, 2), refit_step + 10, dtype=torch.float32, device=device),
        ),
        (
            "draft.layers.1.norm.weight",
            torch.full((3,), refit_step + 20, dtype=torch.bfloat16, device=device),
        ),
    ]


def _assert_lane_error_matches(
    *,
    call: Callable[[], list[tuple[str, torch.Tensor]]],
    match: str,
    pp_group: dist.ProcessGroup,
    original_all_reduce: Callable[..., object],
) -> None:
    with pytest.raises(ValueError, match=match) as error:
        call()

    encoded = str(error.value).encode("utf-8")
    fingerprint = torch.tensor(
        [len(encoded), sum(encoded)],
        dtype=torch.int64,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    fingerprint_min = fingerprint.clone()
    fingerprint_max = fingerprint.clone()
    original_all_reduce(fingerprint_min, op=dist.ReduceOp.MIN, group=pp_group)
    original_all_reduce(fingerprint_max, op=dist.ReduceOp.MAX, group=pp_group)
    torch.testing.assert_close(fingerprint_min, fingerprint_max, rtol=0, atol=0)


def _run_cp2_dtype_bucket_owner_preflight(rank: int, world_size: int) -> None:
    assert world_size == 4
    pp_group, cp_group, cp_rank = _lane_groups(rank)
    pp_ranks = tuple(dist.get_process_group_ranks(pp_group))
    owner_rank = pp_ranks[-1]
    payload_broadcasts = 0
    corrupt_manifest = False
    integer_broadcasts = 0
    original_broadcast = dist.broadcast
    original_all_reduce = dist.all_reduce
    original_all_gather_object = dist.all_gather_object
    original_broadcast_object_list = dist.broadcast_object_list

    def reject_object_collective(*_args, **_kwargs):
        raise AssertionError("draft refit transport must not use object collectives")

    def checked_broadcast(
        tensor: torch.Tensor,
        src: int,
        group: dist.ProcessGroup | None = None,
        **kwargs,
    ):
        nonlocal integer_broadcasts, payload_broadcasts
        assert group is pp_group
        assert group is not cp_group
        result = original_broadcast(tensor, src=src, group=group, **kwargs)
        if tensor.dtype in {torch.bfloat16, torch.float32}:
            payload_broadcasts += 1
        elif tensor.dtype == torch.int64:
            integer_broadcasts += 1
            if corrupt_manifest and integer_broadcasts == 2 and rank == pp_ranks[0]:
                tensor[-1] += 1
        return result

    def checked_all_reduce(
        tensor: torch.Tensor,
        op=dist.ReduceOp.SUM,
        group: dist.ProcessGroup | None = None,
        **kwargs,
    ):
        assert group is pp_group
        assert group is not cp_group
        return original_all_reduce(tensor, op=op, group=group, **kwargs)

    dist.broadcast = checked_broadcast
    dist.all_reduce = checked_all_reduce
    dist.all_gather_object = reject_object_collective
    dist.broadcast_object_list = reject_object_collective
    try:
        for refit_step in (1, 2):
            before = payload_broadcasts
            result = draft_utils.broadcast_draft_weights_from_pp_owner(
                local_exporter=(
                    (
                        lambda refit_step=refit_step: _current_export(
                            refit_step=refit_step
                        )
                    )
                    if rank == owner_rank
                    else None
                ),
                metadata_only=False,
                pp_group=pp_group,
                expected_pp_size=2,
                cp_rank=cp_rank,
            )

            assert payload_broadcasts - before == 2
            assert [
                (name, tuple(tensor.shape), tensor.dtype) for name, tensor in result
            ] == [
                ("draft.layers.0.norm.weight", (2,), torch.bfloat16),
                ("draft.layers.0.proj.weight", (2, 2), torch.float32),
                ("draft.layers.1.norm.weight", (3,), torch.bfloat16),
            ]
            expected = _current_export(refit_step=refit_step)
            for (_, actual), (_, wanted) in zip(result, expected, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

        before = payload_broadcasts
        metadata = draft_utils.broadcast_draft_weights_from_pp_owner(
            local_exporter=(
                (lambda: _current_export(refit_step=2)) if rank == owner_rank else None
            ),
            metadata_only=True,
            pp_group=pp_group,
            expected_pp_size=2,
            cp_rank=cp_rank,
        )
        assert payload_broadcasts == before
        assert all(tensor.device.type == "meta" for _, tensor in metadata)

        falsey_exporter = _FalseyExporter(_current_export(refit_step=3))
        falsey_result = draft_utils.broadcast_draft_weights_from_pp_owner(
            local_exporter=falsey_exporter if rank == owner_rank else None,
            metadata_only=False,
            pp_group=pp_group,
            expected_pp_size=2,
            cp_rank=cp_rank,
        )
        assert falsey_exporter.calls == int(rank == owner_rank)
        for (_, actual), (_, wanted) in zip(
            falsey_result,
            _current_export(refit_step=3),
            strict=True,
        ):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

        exporter_calls = 0

        def counted_exporter() -> list[tuple[str, torch.Tensor]]:
            nonlocal exporter_calls
            exporter_calls += 1
            return _current_export(refit_step=3)

        error_cases: list[
            tuple[
                str,
                int,
                draft_utils.DraftRefitExporter | None,
                bool,
                str,
            ]
        ] = [
            ("pp_size", 3, None, False, "TOPOLOGY_MISMATCH"),
            ("zero_owner", 2, None, False, "OWNER_COUNT"),
            (
                "two_owners",
                2,
                counted_exporter,
                False,
                "OWNER_COUNT",
            ),
            (
                "exporter_error",
                2,
                lambda: (_ for _ in ()).throw(RuntimeError("broken owner export")),
                False,
                "EXPORTER_ERROR.*broken owner export",
            ),
            (
                "manifest_mismatch",
                2,
                lambda: _current_export(refit_step=3),
                True,
                "MANIFEST_MISMATCH",
            ),
        ]
        for (
            case,
            expected_pp_size,
            exporter,
            inject_corruption,
            error_match,
        ) in error_cases:
            before = payload_broadcasts
            calls_before = exporter_calls
            corrupt_manifest = inject_corruption
            integer_broadcasts = 0
            if case in {"exporter_error", "manifest_mismatch"} and rank != owner_rank:
                exporter = None

            _assert_lane_error_matches(
                call=lambda exporter=exporter, expected_pp_size=expected_pp_size: (
                    draft_utils.broadcast_draft_weights_from_pp_owner(
                        local_exporter=exporter,
                        metadata_only=False,
                        pp_group=pp_group,
                        expected_pp_size=expected_pp_size,
                        cp_rank=cp_rank,
                    )
                ),
                match=error_match,
                pp_group=pp_group,
                original_all_reduce=original_all_reduce,
            )
            assert payload_broadcasts == before
            if case in {"zero_owner", "two_owners"}:
                assert exporter_calls == calls_before

        request_cases = [
            (rank == pp_ranks[0], 2, cp_rank),
            (False, 3 if rank == pp_ranks[0] else 2, cp_rank),
            (False, 2, cp_rank + int(rank == pp_ranks[-1])),
        ]
        for metadata_only, expected_pp_size, request_cp_rank in request_cases:
            before = payload_broadcasts
            _assert_lane_error_matches(
                call=lambda: draft_utils.broadcast_draft_weights_from_pp_owner(
                    local_exporter=(
                        (lambda: _current_export(refit_step=3))
                        if rank == owner_rank
                        else None
                    ),
                    metadata_only=metadata_only,
                    pp_group=pp_group,
                    expected_pp_size=expected_pp_size,
                    cp_rank=request_cp_rank,
                ),
                match="LANE_CONFIG_MISMATCH",
                pp_group=pp_group,
                original_all_reduce=original_all_reduce,
            )
            assert payload_broadcasts == before
    finally:
        dist.broadcast = original_broadcast
        dist.all_reduce = original_all_reduce
        dist.all_gather_object = original_all_gather_object
        dist.broadcast_object_list = original_broadcast_object_list


@pytest.mark.mcore
def test_cp2_dtype_bucket_owner_preflight(distributed_test_runner) -> None:
    distributed_test_runner(_run_cp2_dtype_bucket_owner_preflight, world_size=4)


class _EmptyBridge:
    def export_hf_weights(self, *_args: object, **_kwargs: object):
        return iter(())


def _draft_refit_worker(*, draft_model: object | None) -> Any:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.cfg = {"draft": {"enabled": True}}
    worker.model = object()
    worker.megatron_bridge = _EmptyBridge()
    worker.refit_conversion_tasks = []
    worker.draft_model = draft_model
    return worker


def _run_tp2_pp2_cp2_worker_draft_refit(rank: int, world_size: int) -> None:
    from megatron.core import parallel_state

    import nemo_rl.models.megatron.draft as draft_module

    assert world_size == 8
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
    )

    pp_group = parallel_state.get_pipeline_model_parallel_group()
    expected_owner_rank = parallel_state.get_pipeline_model_parallel_last_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    lane_value = cp_rank * 10 + tp_rank
    draft_model = (
        SimpleNamespace(lane_value=lane_value)
        if parallel_state.is_pipeline_last_stage()
        else None
    )
    worker = _draft_refit_worker(draft_model=draft_model)

    payload_sources: list[int] = []
    original_export = draft_module.export_eagle_weights_to_hf
    original_broadcast = dist.broadcast
    original_all_reduce = dist.all_reduce
    original_all_gather_object = dist.all_gather_object
    original_broadcast_object_list = dist.broadcast_object_list

    def export_lane(model: Any) -> list[tuple[str, torch.Tensor]]:
        return [
            (
                "model.layers.0.norm.weight",
                torch.full(
                    (2,),
                    model.lane_value,
                    dtype=torch.float32,
                    device=torch.device("cuda", torch.cuda.current_device()),
                ),
            )
        ]

    def checked_broadcast(
        tensor: torch.Tensor,
        src: int,
        group: dist.ProcessGroup | None = None,
        **kwargs,
    ):
        assert group is pp_group
        if tensor.dtype == torch.float32:
            payload_sources.append(src)
        return original_broadcast(tensor, src=src, group=group, **kwargs)

    def checked_all_reduce(
        tensor: torch.Tensor,
        op=dist.ReduceOp.SUM,
        group: dist.ProcessGroup | None = None,
        **kwargs,
    ):
        if group is dist.group.WORLD:
            assert tensor.shape == ()
            assert tensor.dtype == torch.int64
            assert op == dist.ReduceOp.MIN
        else:
            assert group is pp_group
        return original_all_reduce(tensor, op=op, group=group, **kwargs)

    def reject_object_collective(*_args, **_kwargs):
        raise AssertionError("worker draft refit must not use object collectives")

    draft_module.export_eagle_weights_to_hf = export_lane
    dist.broadcast = checked_broadcast
    dist.all_reduce = checked_all_reduce
    dist.all_gather_object = reject_object_collective
    dist.broadcast_object_list = reject_object_collective
    try:
        metadata = list(
            worker._iter_params_with_optional_kv_scales(draft_metadata_only=True)
        )
        assert [
            (name, tuple(tensor.shape), tensor.dtype) for name, tensor in metadata
        ] == [("draft.model.layers.0.norm.weight", (2,), torch.float32)]
        assert metadata[0][1].device.type == "meta"
        assert payload_sources == []

        exported = list(worker._iter_params_with_optional_kv_scales())
        assert [name for name, _ in exported] == ["draft.model.layers.0.norm.weight"]
        torch.testing.assert_close(
            exported[0][1],
            torch.full_like(exported[0][1], lane_value),
            rtol=0,
            atol=0,
        )
        assert payload_sources == [expected_owner_rank]
    finally:
        draft_module.export_eagle_weights_to_hf = original_export
        dist.broadcast = original_broadcast
        dist.all_reduce = original_all_reduce
        dist.all_gather_object = original_all_gather_object
        dist.broadcast_object_list = original_broadcast_object_list
        parallel_state.destroy_model_parallel()


@pytest.mark.mcore
def test_tp2_pp2_cp2_worker_uses_exact_mcore_draft_lane(
    distributed_test_runner,
) -> None:
    distributed_test_runner(_run_tp2_pp2_cp2_worker_draft_refit, world_size=8)


def _run_tp2_pp2_cp2_worker_draft_failure_consensus(rank: int, world_size: int) -> None:
    from megatron.core import parallel_state

    import nemo_rl.models.megatron.draft as draft_module

    assert world_size == 8
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
    )

    cp_rank = parallel_state.get_context_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    draft_model = object() if parallel_state.is_pipeline_last_stage() else None
    worker = _draft_refit_worker(draft_model=draft_model)
    worker._calculate_refit_param_info = lambda: []
    original_export = draft_module.export_eagle_weights_to_hf

    def export_lane(_model: object) -> list[tuple[str, torch.Tensor]]:
        if cp_rank == 0 and tp_rank == 0:
            raise RuntimeError("injected single-lane draft export failure")
        return [
            (
                "model.layers.0.norm.weight",
                torch.ones(
                    2,
                    dtype=torch.float32,
                    device=torch.device("cuda", torch.cuda.current_device()),
                ),
            )
        ]

    draft_module.export_eagle_weights_to_hf = export_lane
    try:
        with pytest.raises(RuntimeError) as error_info:
            worker.prepare_nccl_reshard_refit_info(
                train_parallelism={
                    "tp_size": 2,
                    "ep_size": 1,
                    "cp_size": 2,
                    "pp_size": 2,
                },
                gen_parallelism={
                    "tp_size": 2,
                    "ep_size": 1,
                    "cp_size": 1,
                    "pp_size": 1,
                },
                train_world_size=8,
                gen_world_size=2,
            )
        error_text = str(error_info.value)
        assert "injected single-lane draft export failure" in error_text
        encoded = error_text.encode("utf-8")
        fingerprint = torch.tensor(
            [len(encoded), sum(encoded)],
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        fingerprint_min = fingerprint.clone()
        fingerprint_max = fingerprint.clone()
        dist.all_reduce(fingerprint_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(fingerprint_max, op=dist.ReduceOp.MAX)
        torch.testing.assert_close(fingerprint_min, fingerprint_max, rtol=0, atol=0)
    finally:
        draft_module.export_eagle_weights_to_hf = original_export
        parallel_state.destroy_model_parallel()


@pytest.mark.mcore
def test_tp2_pp2_cp2_single_lane_draft_failure_reaches_every_rank(
    distributed_test_runner,
) -> None:
    distributed_test_runner(
        _run_tp2_pp2_cp2_worker_draft_failure_consensus,
        world_size=8,
    )


def _run_cp_lane_manifest_mismatch(rank: int, world_size: int) -> None:
    from megatron.core import parallel_state

    import nemo_rl.models.megatron.draft as draft_module
    from nemo_rl.weight_sync import nccl_reshard_utils

    assert world_size == 8
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
    )

    cp_rank = parallel_state.get_context_parallel_rank()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    draft_model = object() if parallel_state.is_pipeline_last_stage() else None
    worker = _draft_refit_worker(draft_model=draft_model)
    worker._calculate_refit_param_info = lambda: []

    class _MetadataBridge:
        def export_hf_weights(self, *_args: object, **_kwargs: object):
            return iter(
                [
                    (
                        "model.layers.0.mlp.gate_proj.weight",
                        torch.empty((2, 2), dtype=torch.bfloat16, device="meta"),
                    )
                ]
            )

    worker.megatron_bridge = _MetadataBridge()
    payload_broadcasts = 0
    build_calls = 0
    descriptor_reductions = 0
    original_export = draft_module.export_eagle_weights_to_hf
    original_build = nccl_reshard_utils.build_nccl_reshard_refit_info
    original_broadcast = dist.broadcast
    original_all_reduce = dist.all_reduce
    original_all_gather_object = dist.all_gather_object
    original_broadcast_object_list = dist.broadcast_object_list

    def export_lane(_model: object) -> list[tuple[str, torch.Tensor]]:
        names = (
            ("model.layers.0.a.weight", "model.layers.0.b.weight")
            if cp_rank == 0
            else ("model.layers.0.b.weight", "model.layers.0.a.weight")
        )
        direct_weight = torch.ones(
            1,
            dtype=torch.float32,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        return [
            (names[0], direct_weight),
            (names[1], torch.empty_like(direct_weight)),
        ]

    def unexpected_build(*_args, **_kwargs):
        nonlocal build_calls
        build_calls += 1
        raise AssertionError("refit builder ran before manifest agreement")

    def checked_broadcast(*args, **kwargs):
        nonlocal payload_broadcasts
        tensor = args[0] if args else kwargs["tensor"]
        if tensor.dtype == torch.float32:
            payload_broadcasts += 1
        return original_broadcast(*args, **kwargs)

    def checked_all_reduce(
        tensor: torch.Tensor,
        op=dist.ReduceOp.SUM,
        group: dist.ProcessGroup | None = None,
        **kwargs,
    ):
        nonlocal descriptor_reductions
        if group is dist.group.WORLD:
            assert tensor.dtype == torch.int64
            if tensor.shape == (6,):
                descriptor_reductions += 1
            else:
                assert tensor.shape == ()
                assert op == dist.ReduceOp.MIN
        else:
            assert group is pp_group
        return original_all_reduce(tensor, op=op, group=group, **kwargs)

    def reject_object_collective(*_args, **_kwargs):
        raise AssertionError("manifest agreement must not use object collectives")

    draft_module.export_eagle_weights_to_hf = export_lane
    nccl_reshard_utils.build_nccl_reshard_refit_info = unexpected_build
    dist.broadcast = checked_broadcast
    dist.all_reduce = checked_all_reduce
    dist.all_gather_object = reject_object_collective
    dist.broadcast_object_list = reject_object_collective
    try:
        with pytest.raises(
            ValueError, match="refit weight manifest differs across ranks"
        ):
            worker.prepare_nccl_reshard_refit_info(
                train_parallelism={
                    "tp_size": 2,
                    "ep_size": 1,
                    "cp_size": 2,
                    "pp_size": 2,
                },
                gen_parallelism={
                    "tp_size": 2,
                    "ep_size": 1,
                    "cp_size": 1,
                    "pp_size": 1,
                },
                train_world_size=8,
                gen_world_size=2,
            )
        assert build_calls == 0
        assert payload_broadcasts == 0
        assert descriptor_reductions == 2
    finally:
        draft_module.export_eagle_weights_to_hf = original_export
        nccl_reshard_utils.build_nccl_reshard_refit_info = original_build
        dist.broadcast = original_broadcast
        dist.all_reduce = original_all_reduce
        dist.all_gather_object = original_all_gather_object
        dist.broadcast_object_list = original_broadcast_object_list
        parallel_state.destroy_model_parallel()


@pytest.mark.mcore
def test_cp_lane_manifest_mismatch_fails_before_refit_builder(
    distributed_test_runner,
) -> None:
    distributed_test_runner(_run_cp_lane_manifest_mismatch, world_size=8)

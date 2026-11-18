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

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from nemo_rl.models.megatron.draft import utils as draft_utils
from nemo_rl.models.policy.draft_config import Eagle3DraftConfig


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


@pytest.mark.mcore
@pytest.mark.parametrize("metadata_only", [False, True])
def test_pp_owner_transport_batches_dtypes_in_pytest_process(
    monkeypatch: pytest.MonkeyPatch, metadata_only: bool
) -> None:
    pp_group = object()
    expected = [
        ("draft.bf16_a", torch.arange(2, dtype=torch.bfloat16)),
        ("draft.fp32", torch.arange(3, dtype=torch.float32)),
        ("draft.bf16_b", torch.arange(4, dtype=torch.bfloat16)),
    ]
    broadcast_dtypes: list[torch.dtype] = []

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, *, op, group: None)

    def record_broadcast(tensor: torch.Tensor, *, src: int, group: object) -> None:
        assert src == 1
        assert group is pp_group
        broadcast_dtypes.append(tensor.dtype)

    monkeypatch.setattr(dist, "broadcast", record_broadcast)

    result = draft_utils.broadcast_draft_weights_from_pp_owner(
        local_exporter=lambda: expected,
        metadata_only=metadata_only,
        pp_group=pp_group,
        expected_pp_size=2,
        cp_rank=0,
    )

    assert [name for name, _ in result] == [name for name, _ in expected]
    if metadata_only:
        assert all(tensor.device.type == "meta" for _, tensor in result)
        assert broadcast_dtypes == [torch.int64, torch.int64, torch.uint8]
    else:
        for (_, actual), (_, wanted) in zip(result, expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        assert broadcast_dtypes == [
            torch.int64,
            torch.int64,
            torch.uint8,
            torch.bfloat16,
            torch.float32,
        ]


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


def _run_cp2_non_owner_payload_materialization_failure(
    rank: int, world_size: int
) -> None:
    assert world_size == 4
    pp_group, _cp_group, cp_rank = _lane_groups(rank)
    pp_ranks = tuple(dist.get_process_group_ranks(pp_group))
    owner_rank = pp_ranks[-1]
    original_empty = torch.empty
    original_broadcast = dist.broadcast
    payload_broadcasts = 0

    def fail_non_owner_bf16_bucket(*args: object, **kwargs: object) -> torch.Tensor:
        size = args[0] if args else kwargs.get("size")
        dtype = kwargs.get("dtype")
        if rank != owner_rank and size == 5 and dtype == torch.bfloat16:
            raise RuntimeError("injected non-owner payload allocation failure")
        return original_empty(*args, **kwargs)

    def count_payload_broadcasts(
        tensor: torch.Tensor,
        src: int,
        group: dist.ProcessGroup | None = None,
        **kwargs: object,
    ) -> object:
        nonlocal payload_broadcasts
        if tensor.dtype in {torch.bfloat16, torch.float32}:
            payload_broadcasts += 1
        return original_broadcast(tensor, src=src, group=group, **kwargs)

    torch.empty = fail_non_owner_bf16_bucket
    dist.broadcast = count_payload_broadcasts
    try:
        _assert_lane_error_matches(
            call=lambda: draft_utils.broadcast_draft_weights_from_pp_owner(
                local_exporter=(
                    (lambda: _current_export(refit_step=4))
                    if rank == owner_rank
                    else None
                ),
                metadata_only=False,
                pp_group=pp_group,
                expected_pp_size=2,
                cp_rank=cp_rank,
            ),
            match="PAYLOAD_INVALID",
            pp_group=pp_group,
            original_all_reduce=dist.all_reduce,
        )
        assert payload_broadcasts == 0
    finally:
        torch.empty = original_empty
        dist.broadcast = original_broadcast


@pytest.mark.mcore
def test_cp2_non_owner_payload_failure_reaches_every_lane_rank(
    distributed_test_runner,
) -> None:
    distributed_test_runner(
        _run_cp2_non_owner_payload_materialization_failure,
        world_size=4,
    )


class _EmptyBridge:
    def get_conversion_tasks(self, *_args: object, **_kwargs: object) -> list[object]:
        return []

    def export_hf_weights(self, *_args: object, **_kwargs: object):
        return iter(())


def _draft_refit_worker(*, draft_model: object | None) -> Any:
    import nemo_rl.models.megatron.draft as draft_module
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    # Policy.__init__ hands workers the validated model, not the raw mapping.
    worker.cfg = {"draft": Eagle3DraftConfig(enabled=True, model_name=None)}
    worker.model = object()
    worker.megatron_bridge = _EmptyBridge()
    worker.refit_conversion_tasks = []
    worker._refit_prequant_names = set()
    worker.draft_model = draft_model
    worker.draft_provider = SimpleNamespace(
        export_weights=lambda model: draft_module.export_eagle_weights_to_hf(model)
    )
    return worker


@pytest.mark.mcore
def test_ipc_refit_preflights_and_caches_draft_before_zmq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.models.policy import utils as policy_utils

    worker = _draft_refit_worker(draft_model=None)
    worker.rank = 0
    worker.zmq_socket = object()
    calls: list[str] = []
    cached_draft = (("draft.model.weight", torch.ones(1)),)

    def preflight(
        *, metadata_only: bool = False
    ) -> tuple[tuple[tuple[str, torch.Tensor], ...], None]:
        assert metadata_only is False
        calls.append("preflight")
        return cached_draft, None

    def params(
        *, kv_scales: object, draft_weights: object
    ) -> Iterator[tuple[str, torch.Tensor]]:
        calls.append("params")
        assert kv_scales == {"scale": 1.0}
        assert draft_weights is cached_draft
        yield from ()

    def stream(**kwargs: object) -> None:
        calls.append("stream")
        list(kwargs["params_generator"])

    worker._preflight_draft_weights_for_refit = preflight
    worker.maybe_init_zmq = lambda: calls.append("init_zmq")
    worker._iter_params_with_optional_kv_scales = params
    monkeypatch.setattr(policy_utils, "stream_weights_via_ipc_zmq_impl", stream)

    worker.stream_weights_via_ipc_zmq(
        buffer_size_bytes=123,
        kv_scales={"scale": 1.0},
    )

    assert calls == ["preflight", "init_zmq", "stream", "params"]


@pytest.mark.mcore
def test_ipc_refit_preflight_failure_stops_before_zmq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.models.policy import utils as policy_utils

    worker = _draft_refit_worker(draft_model=None)
    calls: list[str] = []
    failure = RuntimeError("draft preflight failed")
    worker._preflight_draft_weights_for_refit = lambda **_kwargs: ((), failure)
    worker.maybe_init_zmq = lambda: calls.append("init_zmq")
    monkeypatch.setattr(
        policy_utils,
        "stream_weights_via_ipc_zmq_impl",
        lambda **_kwargs: calls.append("stream"),
    )

    with pytest.raises(RuntimeError, match="draft preflight failed"):
        worker.stream_weights_via_ipc_zmq()

    assert calls == []


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
            worker._iter_params_with_optional_kv_scales(
                draft_weights=worker._draft_weights_for_refit(metadata_only=True)
            )
        )
        assert [
            (name, tuple(tensor.shape), tensor.dtype) for name, tensor in metadata
        ] == [("draft.model.layers.0.norm.weight", (2,), torch.float32)]
        assert metadata[0][1].device.type == "meta"
        assert payload_sources == []

        exported = list(
            worker._iter_params_with_optional_kv_scales(
                draft_weights=worker._draft_weights_for_refit()
            )
        )
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

    class _MetadataBridge(_EmptyBridge):
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


def test_decode_wire_manifest_flags_malformed_metadata_as_wire_invalid() -> None:
    from nemo_rl.models.megatron.draft.utils import (
        _DRAFT_REFIT_STATUS_OK,
        _DRAFT_REFIT_STATUS_WIRE_INVALID,
        DraftRefitTensorSpec,
        _decode_wire_manifest,
        _wire_manifest,
    )

    spec = DraftRefitTensorSpec(
        name="model.layers.0.norm.weight",
        shape=(2,),
        dtype=torch.float32,
        device_type="cuda",
    )
    metadata, utf8_bytes = _wire_manifest((spec,))
    status, detail, manifest = _decode_wire_manifest(
        tensor_count=1, metadata=metadata, utf8_bytes=utf8_bytes
    )
    assert status == _DRAFT_REFIT_STATUS_OK and len(manifest) == 1

    # Trailing metadata values must surface as WIRE_INVALID, not a crash.
    status, detail, manifest = _decode_wire_manifest(
        tensor_count=1, metadata=metadata + [0, 0, 0], utf8_bytes=utf8_bytes
    )
    assert status == _DRAFT_REFIT_STATUS_WIRE_INVALID
    assert "trailing" in detail and manifest == ()

    # Truncated name bytes must surface as WIRE_INVALID.
    status, detail, manifest = _decode_wire_manifest(
        tensor_count=1, metadata=metadata, utf8_bytes=utf8_bytes[:-1]
    )
    assert status == _DRAFT_REFIT_STATUS_WIRE_INVALID
    assert manifest == ()

    # Truncated metadata (count says two tensors) must surface as WIRE_INVALID.
    status, detail, manifest = _decode_wire_manifest(
        tensor_count=2, metadata=metadata, utf8_bytes=utf8_bytes
    )
    assert status == _DRAFT_REFIT_STATUS_WIRE_INVALID
    assert "truncated" in detail and manifest == ()


def test_raise_export_status_maps_codes_and_unknown_to_protocol_error() -> None:
    from nemo_rl.models.megatron.draft.utils import (
        _DRAFT_REFIT_STATUS_WIRE_INVALID,
        _raise_export_status,
    )

    with pytest.raises(ValueError, match="WIRE_INVALID"):
        _raise_export_status(
            _DRAFT_REFIT_STATUS_WIRE_INVALID,
            "bad bytes",
            cp_rank=0,
            pp_ranks=(0, 1),
        )
    with pytest.raises(ValueError, match="PROTOCOL_ERROR"):
        _raise_export_status(99, "unknown", cp_rank=1, pp_ranks=(2,))


def _raise_in_exporter() -> list[tuple[str, torch.Tensor]]:
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("items", "limits", "expected_status", "detail"),
    [
        (_raise_in_exporter, {}, "EXPORTER_ERROR", "exporter raised RuntimeError"),
        ([], {}, "MANIFEST_INVALID", "no tensors"),
        (
            [("a", torch.zeros(1)), ("b", torch.zeros(1))],
            {"_DRAFT_REFIT_MAX_TENSORS": 1},
            "MANIFEST_INVALID",
            "more than 1 tensors",
        ),
        ([("a", torch.zeros(1), 0)], {}, "MANIFEST_INVALID", "not a (name, tensor)"),
        ([("", torch.zeros(1))], {}, "MANIFEST_INVALID", "empty or non-string"),
        ([(7, torch.zeros(1))], {}, "MANIFEST_INVALID", "empty or non-string"),
        (
            [("a", torch.zeros(1)), ("a", torch.zeros(1))],
            {},
            "MANIFEST_INVALID",
            "duplicate tensor name: a",
        ),
        ([("a", [0.0])], {}, "MANIFEST_INVALID", "non-tensor value for a"),
        (
            [("a", torch.zeros(2).to_sparse())],
            {},
            "MANIFEST_INVALID",
            "non-strided tensor for a",
        ),
        (
            [("a", torch.zeros(1, dtype=torch.float8_e4m3fn))],
            {},
            "MANIFEST_INVALID",
            "unsupported dtype",
        ),
        (
            [("a", torch.zeros(1))],
            {"_DRAFT_REFIT_DEVICE_TO_CODE": {"cuda": 1, "meta": 2}},
            "MANIFEST_INVALID",
            "unsupported device cpu for a",
        ),
        (
            [("a", torch.zeros((1,) * 3))],
            {"_DRAFT_REFIT_MAX_NDIM": 2},
            "MANIFEST_INVALID",
            "rank-3 tensor for a",
        ),
        (
            [("a", torch.zeros(1)), ("b", torch.zeros(1, device="meta"))],
            {},
            "MANIFEST_INVALID",
            "multiple device types",
        ),
        ([("\ud800", torch.zeros(1))], {}, "MANIFEST_INVALID", "non-UTF-8"),
        (
            [("abc", torch.zeros(1)), ("def", torch.zeros(1))],
            {"_DRAFT_REFIT_MAX_UTF8_BYTES": 5},
            "MANIFEST_INVALID",
            "UTF-8 metadata limit",
        ),
    ],
)
def test_materialize_draft_refit_export_rejects_invalid_exports(
    monkeypatch, items, limits, expected_status, detail
) -> None:
    for name, value in limits.items():
        monkeypatch.setattr(draft_utils, name, value)
    exporter = items if callable(items) else lambda: iter(items)

    export = draft_utils._materialize_draft_refit_export(exporter)

    assert export.status == getattr(
        draft_utils, f"_DRAFT_REFIT_STATUS_{expected_status}"
    )
    assert detail in export.detail
    assert export.manifest == () and export.tensors == ()


def test_materialize_draft_refit_export_accepts_valid_export() -> None:
    tensors = [("a", torch.zeros(2)), ("b", torch.ones(1, 3, dtype=torch.bfloat16))]

    export = draft_utils._materialize_draft_refit_export(lambda: iter(tensors))

    assert export.status == draft_utils._DRAFT_REFIT_STATUS_OK
    assert [spec.name for spec in export.manifest] == ["a", "b"]
    assert export.manifest[1].shape == (1, 3)
    assert export.manifest[1].dtype == torch.bfloat16
    assert [name for name, _ in export.tensors] == ["a", "b"]


@pytest.mark.parametrize(
    ("tensor_count", "metadata", "utf8_bytes", "detail"),
    [
        (1, [99, 1, 1, 1, 2], b"w", "unsupported dtype code"),
        (1, [8, 9, 1, 1, 2], b"w", "unsupported device code"),
        (1, [8, 1, 17, 1, 2], b"w", "invalid tensor rank"),
        (1, [8, 1, -1, 1, 2], b"w", "invalid tensor rank"),
        (1, [8, 1, 1, 0, 2], b"w", "tensor-name bounds"),
        (1, [8, 1, 2, 1, 2], b"w", "truncated tensor shape"),
        (1, [8, 1, 1, 1, -2], b"w", "negative tensor dimension"),
        (1, [8, 1, 1, 1, 2], b"\xff", "codec"),
        (2, [8, 1, 1, 1, 2, 8, 1, 1, 1, 2], b"ww", "duplicate"),
    ],
    ids=[
        "dtype",
        "device",
        "rank_too_large",
        "rank_negative",
        "empty_name",
        "truncated_shape",
        "negative_dim",
        "bad_utf8",
        "duplicate_name",
    ],
)
def test_decode_wire_manifest_error_branches(
    tensor_count: int, metadata: list[int], utf8_bytes: bytes, detail: str
) -> None:
    status, message, manifest = draft_utils._decode_wire_manifest(
        tensor_count=tensor_count, metadata=metadata, utf8_bytes=utf8_bytes
    )

    assert status == draft_utils._DRAFT_REFIT_STATUS_WIRE_INVALID
    assert detail in message
    assert manifest == ()


def _manifest(shape: tuple[int, ...]) -> list[tuple[str, str, tuple[int, ...], str]]:
    return [("draft.w", "torch.float32", shape, "cuda")]


@pytest.mark.mcore
def test_manifest_rank_agreement_skips_without_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        assert_refit_weight_manifest_rank_agreement,
    )

    def fail_all_reduce(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no collective without a process group")

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "all_reduce", fail_all_reduce)

    assert_refit_weight_manifest_rank_agreement(_manifest((2,)), group=object())


@pytest.mark.mcore
@pytest.mark.parametrize("ranks_agree", [True, False])
def test_manifest_rank_agreement_in_pytest_process(
    monkeypatch: pytest.MonkeyPatch, ranks_agree: bool
) -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        assert_refit_weight_manifest_rank_agreement,
    )

    group = object()
    ops: list[object] = []

    def all_reduce(tensor: torch.Tensor, *, op: object, group: object) -> None:
        assert group is expected_group
        assert tensor.dtype == torch.int64 and tensor.device.type == "cpu"
        ops.append(op)
        # A peer rank with a different manifest moves only the MAX result.
        if not ranks_agree and op == dist.ReduceOp.MAX:
            tensor[-1] += 1

    expected_group = group
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda _group: "gloo")
    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    if ranks_agree:
        assert_refit_weight_manifest_rank_agreement(_manifest((2,)), group=group)
    else:
        with pytest.raises(ValueError, match="differs across ranks"):
            assert_refit_weight_manifest_rank_agreement(_manifest((2,)), group=group)
    assert ops == [dist.ReduceOp.MIN, dist.ReduceOp.MAX]


def _run_manifest_rank_agreement(rank: int, world_size: int) -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        assert_refit_weight_manifest_rank_agreement,
    )

    assert world_size == 2
    group = dist.new_group([0, 1], backend="gloo")
    assert_refit_weight_manifest_rank_agreement(_manifest((2, 3)), group=group)
    with pytest.raises(ValueError, match="differs across ranks"):
        assert_refit_weight_manifest_rank_agreement(
            _manifest((2, 3 + rank)), group=group
        )


@pytest.mark.mcore
def test_manifest_rank_agreement_two_ranks(distributed_test_runner) -> None:
    distributed_test_runner(_run_manifest_rank_agreement, world_size=2, backend="gloo")


def _patch_single_rank_train_world(
    monkeypatch: pytest.MonkeyPatch, worker: Any
) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(dist, "get_rank", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dist, "broadcast", lambda *_args, **_kwargs: None)
    worker._draft_refit_lane_kwargs = lambda: {
        "pp_group": None,
        "expected_pp_size": 1,
        "cp_rank": 0,
    }


@pytest.mark.mcore
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("metadata_only", [False, True])
def test_preflight_draft_weights_world_size_one(
    monkeypatch: pytest.MonkeyPatch, metadata_only: bool
) -> None:
    import nemo_rl.models.megatron.draft as draft_module

    worker = _draft_refit_worker(draft_model=SimpleNamespace())
    _patch_single_rank_train_world(monkeypatch, worker)
    weight = torch.arange(3, dtype=torch.float32)
    monkeypatch.setattr(
        draft_module, "export_eagle_weights_to_hf", lambda _model: [("w", weight)]
    )

    draft_weights, error = worker._preflight_draft_weights_for_refit(
        metadata_only=metadata_only
    )

    assert error is None
    assert [name for name, _ in draft_weights] == ["draft.w"]
    if metadata_only:
        assert draft_weights[0][1].device.type == "meta"
        assert draft_weights[0][1].shape == weight.shape
    else:
        assert draft_weights[0][1] is weight
    assert worker._draft_weights_for_refit(metadata_only=metadata_only)[0][0] == (
        "draft.w"
    )


@pytest.mark.mcore
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_preflight_draft_weights_world_size_one_returns_export_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nemo_rl.models.megatron.draft as draft_module

    worker = _draft_refit_worker(draft_model=SimpleNamespace())
    _patch_single_rank_train_world(monkeypatch, worker)

    def fail_export(_model: object) -> list[tuple[str, torch.Tensor]]:
        raise ValueError("draft export boom")

    monkeypatch.setattr(draft_module, "export_eagle_weights_to_hf", fail_export)

    draft_weights, error = worker._preflight_draft_weights_for_refit()

    assert draft_weights == ()
    assert isinstance(error, RuntimeError)
    assert "train rank 0" in str(error) and "draft export boom" in str(error)
    with pytest.raises(RuntimeError, match="draft export boom"):
        worker._draft_weights_for_refit()


@pytest.mark.mcore
def test_iter_params_reuses_preflighted_draft_weights_without_pp_export() -> None:
    worker = _draft_refit_worker(draft_model=SimpleNamespace())

    def fail_draft_export(**_kwargs: object) -> Iterator[tuple[str, torch.Tensor]]:
        raise AssertionError("cached draft weights re-entered the PP broadcast")

    worker._iter_draft_weights_for_refit = fail_draft_export
    cached = (("draft.w", torch.ones(2)), ("draft.b", torch.zeros(1)))

    exported = list(worker._iter_params_with_optional_kv_scales(draft_weights=cached))

    assert exported == list(cached)
    assert all(
        actual is wanted
        for (_, actual), (_, wanted) in zip(exported, cached, strict=True)
    )


def _run_pp2_lane_gloo(rank: int, world_size: int) -> None:
    assert world_size == 2
    pp_group = dist.new_group([0, 1], backend="gloo")
    owner_rank = 1
    expected = [
        ("draft.fp32", torch.arange(3, dtype=torch.float32) + 10),
        ("draft.i64", torch.arange(4, dtype=torch.int64)),
        ("draft.fp32_b", torch.full((2, 2), 7.0)),
    ]
    exporter = (lambda: expected) if rank == owner_rank else None

    for metadata_only in (True, False):
        result = draft_utils.broadcast_draft_weights_from_pp_owner(
            local_exporter=exporter,
            metadata_only=metadata_only,
            pp_group=pp_group,
            expected_pp_size=2,
            cp_rank=0,
        )
        assert [name for name, _ in result] == [name for name, _ in expected]
        for (_, actual), (_, wanted) in zip(result, expected, strict=True):
            assert actual.shape == wanted.shape and actual.dtype == wanted.dtype
            if metadata_only:
                assert actual.device.type == "meta"
            else:
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def fail_export() -> list[tuple[str, torch.Tensor]]:
        raise RuntimeError("owner export boom")

    with pytest.raises(ValueError, match="EXPORTER_ERROR"):
        draft_utils.broadcast_draft_weights_from_pp_owner(
            local_exporter=fail_export if rank == owner_rank else None,
            metadata_only=False,
            pp_group=pp_group,
            expected_pp_size=2,
            cp_rank=0,
        )
    with pytest.raises(ValueError, match="TOPOLOGY_MISMATCH"):
        draft_utils.broadcast_draft_weights_from_pp_owner(
            local_exporter=exporter,
            metadata_only=False,
            pp_group=pp_group,
            expected_pp_size=4,
            cp_rank=0,
        )


@pytest.mark.mcore
def test_pp2_lane_transport_two_ranks(distributed_test_runner) -> None:
    distributed_test_runner(_run_pp2_lane_gloo, world_size=2, backend="gloo")

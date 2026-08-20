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

import pytest
import torch
import torch.distributed as dist

from nemo_rl.models.megatron.draft import utils as draft_utils


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

        error_cases: list[tuple[str, int, Callable[[], object] | None, bool, str]] = [
            ("pp_size", 3, None, False, "TOPOLOGY_MISMATCH"),
            ("zero_owner", 2, None, False, "OWNER_COUNT"),
            (
                "two_owners",
                2,
                lambda: _current_export(refit_step=3),
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
    finally:
        dist.broadcast = original_broadcast
        dist.all_reduce = original_all_reduce
        dist.all_gather_object = original_all_gather_object
        dist.broadcast_object_list = original_broadcast_object_list


@pytest.mark.mcore
def test_cp2_dtype_bucket_owner_preflight(distributed_test_runner) -> None:
    distributed_test_runner(_run_cp2_dtype_bucket_owner_preflight, world_size=4)

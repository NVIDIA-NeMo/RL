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
"""Policy-worker deferred route assembly tests."""

from __future__ import annotations

from collections import Counter

import torch
from nemo_gym.token_id_capture.staging.digest import compute_extras_digest

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    ROUTE_PASSTHROUGH_FLAG,
    ROUTE_PLAN_TAG,
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_EXPERTS_FIELD,
    ROUTED_EXTRAS_METADATA_FIELD,
)
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.route_plan import (
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    RouteSpan,
    encode_route_plan,
)
from nemo_rl.utils.routed_experts_codec import encode_routed_experts


class _Rows(dict):
    def __init__(self, routed: list[torch.Tensor]) -> None:
        super().__init__(
            {
                ROUTED_EXPERTS_FIELD: routed,
                ROUTED_EXPERTS_ENCODING_FIELD: [
                    torch.tensor([1], dtype=torch.int64) for _ in routed
                ],
                ROUTED_EXTRAS_METADATA_FIELD: [
                    torch.tensor(list(b"{}"), dtype=torch.uint8) for _ in routed
                ],
            }
        )
        self.batch_size = (len(routed),)


class _RouteClient:
    def __init__(self, fragments: dict[str, torch.Tensor]) -> None:
        self.fragments = fragments
        self.calls: list[list[str]] = []

    def get_samples(self, *, sample_ids, partition_id, select_fields):
        assert partition_id == "staging"
        assert select_fields == [
            ROUTED_EXPERTS_FIELD,
            ROUTED_EXPERTS_ENCODING_FIELD,
            ROUTED_EXTRAS_METADATA_FIELD,
        ]
        self.calls.append(list(sample_ids))
        return _Rows([self.fragments[key] for key in sample_ids])


class _Worker(TQWorkerMixin):
    def __init__(self, client: _RouteClient) -> None:
        self._dp_client = client
        self._route_fallback_counts = Counter()

    def _routed_experts_dimensions(self) -> tuple[int, int]:
        return 1, 2


def _plan(
    spans: tuple[RouteSpan, ...], *, expected: int, cleanup: tuple[str, ...]
) -> dict:
    return encode_route_plan(
        RouteAssemblyPlan(
            schema_version=ROUTE_PLAN_SCHEMA_VERSION,
            staging_partition="staging",
            spans=spans,
            cleanup_staging_keys=cleanup,
            expected_token_length=expected,
        )
    )


def _span(
    client: _RouteClient,
    staging_key: str,
    carry_len: int,
    generation_len: int,
    staged_route_len: int,
) -> RouteSpan:
    routes = client.fragments[staging_key]
    extras_digest = compute_extras_digest(
        {ROUTED_EXPERTS_FIELD: encode_routed_experts(routes)}
    )
    return RouteSpan(
        staging_key,
        carry_len,
        generation_len,
        staged_route_len,
        extras_digest_version=1,
        extras_digest=extras_digest,
    )


def _meta(plans: list[dict], lengths: list[int]) -> tuple[KVBatchMeta, BatchedDataDict]:
    meta = KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=[f"row{i}" for i in range(len(plans))],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=lengths,
        tags=[{ROUTE_PLAN_TAG: plan} for plan in plans],
        extra_info={ROUTE_PASSTHROUGH_FLAG: True},
    )
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros((len(plans), max(lengths) + 1), dtype=torch.long),
            "input_lengths": torch.tensor(lengths, dtype=torch.long),
        }
    )
    return meta, data


def test_worker_coalesces_keys_and_replays_full_tail_and_placeholder() -> None:
    fragments = {
        "r/c0": torch.tensor([[[10, 11]], [[12, 13]]], dtype=torch.int16),
        "r/c1": torch.tensor([[[22, 23]]], dtype=torch.int16),
    }
    client = _RouteClient(fragments)
    worker = _Worker(client)
    plans = [
        _plan(
            (
                _span(client, "r/c0", 0, 2, 2),
                _span(client, "r/c1", 1, 1, 1),
            ),
            expected=4,
            cleanup=("r/c0", "r/c1", "r/off_chain"),
        ),
        _plan((), expected=1, cleanup=("r/rejected",)),
    ]
    meta, data = _meta(plans, [4, 1])

    result = worker._maybe_assemble_routed_experts(meta, data)

    assert client.calls == [["r/c0", "r/c1"]]
    routed = result[ROUTED_EXPERTS_FIELD]
    assert routed.dtype == torch.int16
    assert routed.shape == (2, 5, 1, 2)
    assert routed[0, :4, 0].tolist() == [
        [10, 11],
        [12, 13],
        [-1, -1],
        [22, 23],
    ]
    assert bool(routed[0, 4].eq(-1).all())
    assert bool(routed[1].eq(-1).all())
    assert not worker._route_fallback_counts


def test_wrong_model_shape_falls_back_for_entire_rollout() -> None:
    client = _RouteClient({"r/c0": torch.tensor([[[10]], [[11]]], dtype=torch.int16)})
    worker = _Worker(client)
    plan = _plan(
        (_span(client, "r/c0", 0, 2, 2),),
        expected=2,
        cleanup=("r/c0",),
    )
    meta, data = _meta([plan], [2])

    routed = worker._maybe_assemble_routed_experts(meta, data)[ROUTED_EXPERTS_FIELD]

    assert bool(routed.eq(-1).all())
    assert worker._route_fallback_counts == Counter({"fragment_model_shape": 1})


def test_tampered_fragment_falls_back_for_entire_rollout() -> None:
    client = _RouteClient(
        {"r/c0": torch.tensor([[[10, 11]], [[12, 13]]], dtype=torch.int16)}
    )
    worker = _Worker(client)
    span = _span(client, "r/c0", 0, 2, 2)
    client.fragments["r/c0"][0, 0, 0] = 999
    meta, data = _meta(
        [_plan((span,), expected=2, cleanup=("r/c0",))],
        [2],
    )

    routed = worker._maybe_assemble_routed_experts(meta, data)[ROUTED_EXPERTS_FIELD]

    assert bool(routed.eq(-1).all())
    assert worker._route_fallback_counts == Counter({"fragment_integrity": 1})

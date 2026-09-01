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

import numpy as np
import pytest
import torch

from nemo_rl.utils.routed_experts_ref import (
    ROUTED_EXPERTS_REF_DTYPE,
    ROUTED_EXPERTS_REF_KEY,
    ROUTED_EXPERTS_REF_SCHEMA,
    RoutedExpertsMaterializerState,
    RoutedExpertsStoreState,
    materialize_routed_experts_refs,
    routed_experts_ref_lookup_key,
    routed_experts_materialization_key,
    slice_routed_experts_ref,
)


def _ref(*, target: int = 3, shape: tuple[int, int, int] = (5, 2, 2)):
    return {
        "schema": ROUTED_EXPERTS_REF_SCHEMA,
        "store": "store-a",
        "store_instance_id": "instance-a",
        "request_id": "request-a",
        "key": ROUTED_EXPERTS_REF_KEY,
        "task_index": 11,
        "rollout_index": 2,
        "target_weight_version": target,
        "offset": 0,
        "length": shape[0],
        "shape": list(shape),
        "dtype": ROUTED_EXPERTS_REF_DTYPE,
    }


def test_materialize_refs_resolves_one_full_object_for_multiple_message_slices():
    source = np.arange(5 * 2 * 2, dtype=np.int16).reshape(5, 2, 2)
    full_ref = _ref()
    refs = [
        [
            slice_routed_experts_ref(full_ref, offset=0, length=2),
            slice_routed_experts_ref(full_ref, offset=2, length=3),
        ]
    ]
    resolve_calls = []

    def resolve(ref):
        resolve_calls.append(routed_experts_ref_lookup_key(ref))
        return source

    materialized = materialize_routed_experts_refs(
        refs,
        input_ids=torch.zeros(1, 8, dtype=torch.long),
        input_lengths=torch.tensor([5], dtype=torch.int32),
        resolver=resolve,
    )

    assert resolve_calls == [(3, 11, 2, "request-a")]
    assert materialized.dtype == torch.int16
    assert torch.equal(materialized[0, :5], torch.from_numpy(source))
    assert torch.equal(
        materialized[0, 5:], torch.full((3, 2, 2), -1, dtype=torch.int16)
    )


def test_materialize_refs_concatenates_segments_from_multiple_turn_requests():
    first_source = np.arange(3 * 2 * 2, dtype=np.int16).reshape(3, 2, 2)
    second_source = np.arange(4 * 2 * 2, dtype=np.int16).reshape(4, 2, 2) + 100
    first_ref = _ref(shape=(3, 2, 2)) | {"request_id": "request-first"}
    second_ref = _ref(shape=(4, 2, 2)) | {"request_id": "request-second"}
    refs = [
        [
            slice_routed_experts_ref(first_ref, offset=0, length=2),
            slice_routed_experts_ref(second_ref, offset=1, length=3),
        ]
    ]
    sources = {
        "request-first": first_source,
        "request-second": second_source,
    }

    materialized = materialize_routed_experts_refs(
        refs,
        input_ids=torch.zeros(1, 6, dtype=torch.long),
        input_lengths=torch.tensor([5], dtype=torch.int32),
        resolver=lambda ref: sources[ref["request_id"]],
    )

    expected = np.concatenate((first_source[:2], second_source[1:4]), axis=0)
    assert torch.equal(materialized[0, :5], torch.from_numpy(expected))
    assert torch.equal(
        materialized[0, 5:], torch.full((1, 2, 2), -1, dtype=torch.int16)
    )


def test_materialize_refs_rejects_incomplete_sample_coverage():
    with pytest.raises(ValueError, match="do not cover"):
        materialize_routed_experts_refs(
            [[slice_routed_experts_ref(_ref(), offset=0, length=4)]],
            input_ids=torch.zeros(1, 5, dtype=torch.long),
            input_lengths=torch.tensor([5], dtype=torch.int32),
            resolver=lambda ref: np.zeros((5, 2, 2), dtype=np.int16),
        )


def test_materialization_key_covers_order_slices_padding_and_target_version():
    full_ref = _ref()
    first = slice_routed_experts_ref(full_ref, offset=0, length=2)
    second = slice_routed_experts_ref(full_ref, offset=2, length=3)

    def key(refs, *, padded_length=8):
        return routed_experts_materialization_key(
            refs,
            batch_size=1,
            padded_length=padded_length,
            input_lengths=[5],
        )

    baseline = key([[first, second]])

    assert key([[dict(first), dict(second)]]) == baseline
    assert key([[second | {"length": 3}, first | {"length": 2}]]) != baseline
    assert key([[first, second]], padded_length=9) != baseline
    assert (
        key(
            [[first | {"target_weight_version": 4}, second]],
        )
        != baseline
    )

    sample_a = _ref(shape=(2, 2, 2)) | {"request_id": "sample-a"}
    sample_b = _ref(shape=(3, 2, 2)) | {"request_id": "sample-b"}
    ordered = routed_experts_materialization_key(
        [[sample_a], [sample_b]],
        batch_size=2,
        padded_length=4,
        input_lengths=[2, 3],
    )
    reordered = routed_experts_materialization_key(
        [[sample_b], [sample_a]],
        batch_size=2,
        padded_length=4,
        input_lengths=[3, 2],
    )
    assert ordered != reordered


def test_materializer_state_singleflights_sibling_requests_and_retires():
    state = RoutedExpertsMaterializerState(max_entries=2)
    key = "microbatch-a"

    assert state.get(key) is None
    state.put(
        key=key,
        object_ref="dense-ref-a",
        nbytes=80,
        shape=(1, 5, 2, 2),
        dtype="int16",
        target_weight_versions=[3],
    )
    for _ in range(63):
        assert state.get(key).object_ref == "dense-ref-a"

    assert state.stats() == {
        "remaining_materializations": 1,
        "materialization_cache_requests": 64,
        "materialization_cache_hits": 63,
        "materialization_cache_misses": 1,
        "materializations": 1,
        "materialized_bytes": 80,
        "materialization_cache_evictions": 0,
    }

    assert state.retire_through(3) == {
        "remaining_materializations": 0,
        "materialization_cache_requests": 64,
        "materialization_cache_hits": 63,
        "materialization_cache_misses": 1,
        "materializations": 1,
        "materialized_bytes": 80,
        "materialization_cache_evictions": 0,
        "retired_through": 3,
        "retired_materializations": 1,
        "retired_materialized_bytes": 80,
    }


def test_materializer_state_lru_bound_evicts_least_recently_used_entry():
    state = RoutedExpertsMaterializerState(max_entries=2)
    for index in range(2):
        state.put(
            key=f"microbatch-{index}",
            object_ref=f"dense-ref-{index}",
            nbytes=40,
            shape=(1, 5, 2, 2),
            dtype="int16",
            target_weight_versions=[3],
        )
    assert state.get("microbatch-0") is not None
    state.put(
        key="microbatch-2",
        object_ref="dense-ref-2",
        nbytes=40,
        shape=(1, 5, 2, 2),
        dtype="int16",
        target_weight_versions=[4],
    )

    assert state.get("microbatch-1") is None
    assert state.get("microbatch-0") is not None
    assert state.get("microbatch-2") is not None
    assert state.stats()["materialization_cache_evictions"] == 1


def test_store_state_retire_through_is_monotonic_and_rejects_late_puts():
    state = RoutedExpertsStoreState("instance-a")
    target_three = _ref(target=3)
    target_four = _ref(target=4)
    target_four["request_id"] = "request-b"
    state.put(
        key=routed_experts_ref_lookup_key(target_three),
        object_ref="ref-three",
        nbytes=40,
        shape=(5, 2, 2),
        dtype="int16",
    )
    state.put(
        key=routed_experts_ref_lookup_key(target_four),
        object_ref="ref-four",
        nbytes=60,
        shape=(5, 2, 2),
        dtype="int16",
    )

    result = state.retire_through(3)

    assert result == {
        "retired_through": 3,
        "retired_objects": 1,
        "retired_bytes": 40,
        "remaining_objects": 1,
    }
    assert (
        state.get(
            key=routed_experts_ref_lookup_key(target_four),
            store_instance_id="instance-a",
        ).object_ref
        == "ref-four"
    )
    with pytest.raises(RuntimeError, match="retired target-weight version"):
        state.put(
            key=routed_experts_ref_lookup_key(target_three),
            object_ref="late-ref",
            nbytes=40,
            shape=(5, 2, 2),
            dtype="int16",
        )

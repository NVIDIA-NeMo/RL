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
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from nemo_rl.utils.routed_experts_ref import (
    ROUTED_EXPERTS_RAY_NAMESPACE,
    ROUTED_EXPERTS_REF_DTYPE,
    ROUTED_EXPERTS_REF_KEY,
    ROUTED_EXPERTS_REF_SCHEMA,
    RoutedExpertsObjectStore,
    RoutedExpertsStoreState,
    RoutedExpertsStoreWriter,
    _assemble_routed_experts_range_results,
    _materialize_normalized_routed_experts_with_ray_transport,
    _normalize_routed_experts_batch,
    _plan_routed_experts_range_reads,
    materialize_routed_experts_refs,
    registry_actor_name,
    retire_routed_experts_through,
    routed_experts_ref_lookup_key,
    slice_routed_experts_ref,
    validate_routed_experts_ref,
)


def _ref(*, target: int = 3, attempt: int = 0, shape: tuple[int, int, int] = (5, 2, 2)):
    return {
        "schema": ROUTED_EXPERTS_REF_SCHEMA,
        "store": "store-a",
        "store_instance_id": "instance-a",
        "request_id": "request-a",
        "key": ROUTED_EXPERTS_REF_KEY,
        "task_index": 11,
        "rollout_index": 2,
        "attempt_index": attempt,
        "target_weight_version": target,
        "offset": 0,
        "length": shape[0],
        "shape": list(shape),
        "dtype": ROUTED_EXPERTS_REF_DTYPE,
    }


def _put_state(state: RoutedExpertsStoreState, ref: dict, value: np.ndarray) -> None:
    state.put(
        key=routed_experts_ref_lookup_key(ref),
        object_ref=value,
        nbytes=int(value.nbytes),
        shape=value.shape,
        dtype=str(value.dtype),
    )


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

    assert resolve_calls == [(3, 11, 2, 0, "request-a")]
    assert materialized.dtype == torch.int16
    assert torch.equal(materialized[0, :5], torch.from_numpy(source))
    assert torch.equal(
        materialized[0, 5:], torch.full((3, 2, 2), -1, dtype=torch.int16)
    )


def test_materialize_single_full_object_preserves_batch_shape_without_copy():
    source = np.arange(5 * 2 * 2, dtype=np.int16).reshape(5, 2, 2)
    materialized = materialize_routed_experts_refs(
        [[_ref()]],
        input_ids=torch.zeros(1, 5, dtype=torch.long),
        input_lengths=torch.tensor([5], dtype=torch.int32),
        resolver=lambda ref: source,
    )

    assert materialized.shape == (1, 5, 2, 2)
    source[0, 0, 0] = 123
    assert materialized[0, 0, 0, 0].item() == 123


def test_materialize_single_full_object_adds_requested_padding():
    source = np.arange(5 * 2 * 2, dtype=np.int16).reshape(5, 2, 2)
    materialized = materialize_routed_experts_refs(
        [[_ref()]],
        input_ids=torch.zeros(1, 8, dtype=torch.long),
        input_lengths=torch.tensor([5], dtype=torch.int32),
        resolver=lambda ref: source,
    )

    assert torch.equal(materialized[0, :5], torch.from_numpy(source))
    assert torch.all(materialized[0, 5:] == -1)


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


def test_store_state_returns_only_requested_ranges_from_owned_arrays():
    state = RoutedExpertsStoreState("instance-a")
    first_source = np.arange(5 * 2 * 2, dtype=np.int16).reshape(5, 2, 2)
    second_source = np.arange(4 * 2 * 2, dtype=np.int16).reshape(4, 2, 2) + 100
    first_ref = _ref(shape=(5, 2, 2)) | {"request_id": "request-first"}
    second_ref = _ref(shape=(4, 2, 2)) | {"request_id": "request-second"}
    _put_state(state, first_ref, first_source)
    _put_state(state, second_ref, second_source)

    refs = [
        slice_routed_experts_ref(first_ref, offset=1, length=2),
        slice_routed_experts_ref(second_ref, offset=2, length=2),
        slice_routed_experts_ref(first_ref, offset=4, length=1),
    ]
    packed = state.get_ranges(refs)

    expected = np.concatenate(
        (first_source[1:3], second_source[2:4], first_source[4:5]), axis=0
    )
    assert np.array_equal(packed, expected)
    assert packed.nbytes == expected.nbytes
    assert not np.shares_memory(packed, first_source)
    assert not np.shares_memory(packed, second_source)


@pytest.mark.parametrize("attempt", [0, 1, 7])
def test_store_state_rejects_duplicates_within_one_attempt(attempt: int) -> None:
    state = RoutedExpertsStoreState("instance-a")
    ref = _ref(attempt=attempt)
    value = np.zeros(ref["shape"], dtype=np.int16)
    _put_state(state, ref, value)

    with pytest.raises(RuntimeError, match="Duplicate routed-experts object key"):
        _put_state(state, ref, value + 1)
    assert (
        state.get(
            key=routed_experts_ref_lookup_key(ref), store_instance_id="instance-a"
        ).object_ref
        is value
    )


@pytest.mark.parametrize("legacy_first_slice", [False, True])
def test_attempts_with_same_request_id_have_independent_reads_and_slices(
    legacy_first_slice: bool,
) -> None:
    state = RoutedExpertsStoreState("instance-a")
    refs = [_ref(attempt=0, shape=(6, 2, 2)), _ref(attempt=1, shape=(7, 2, 2))]
    sources = [
        np.arange(6 * 2 * 2, dtype=np.int16).reshape(6, 2, 2),
        np.arange(7 * 2 * 2, dtype=np.int16).reshape(7, 2, 2) + 100,
    ]
    for ref, source in zip(refs, sources):
        _put_state(state, ref, source)
    assert [routed_experts_ref_lookup_key(ref) for ref in refs] == [
        (3, 11, 2, 0, "request-a"),
        (3, 11, 2, 1, "request-a"),
    ]
    first_ref = dict(refs[0])
    if legacy_first_slice:
        del first_ref["attempt_index"]
    segments = [
        [
            slice_routed_experts_ref(first_ref, offset=1, length=2),
            slice_routed_experts_ref(refs[0], offset=3, length=2),
        ],
        [slice_routed_experts_ref(refs[1], offset=2, length=4)],
    ]
    resolve_calls = []

    def resolve(ref: dict[str, Any]) -> np.ndarray:
        key = routed_experts_ref_lookup_key(ref)
        resolve_calls.append(key)
        return state.get(key=key, store_instance_id=ref["store_instance_id"]).object_ref

    dense = materialize_routed_experts_refs(
        segments,
        input_ids=torch.zeros(2, 5, dtype=torch.long),
        input_lengths=torch.tensor([4, 4], dtype=torch.int32),
        resolver=resolve,
    )
    assert resolve_calls == [routed_experts_ref_lookup_key(ref) for ref in refs]
    assert torch.equal(dense[0, :4], torch.from_numpy(sources[0][1:5]))
    assert torch.equal(dense[1, :4], torch.from_numpy(sources[1][2:6]))
    assert torch.all(dense[:, 4:] == -1)

    batch = _normalize_routed_experts_batch(
        segments, batch_size=2, padded_length=5, input_lengths=[4, 4]
    )
    groups, stats = _plan_routed_experts_range_reads(batch)
    assert len(groups) == 1
    assert stats["range_read_source_objects"] == 2
    assert stats["full_source_rows_equivalent"] == 13
    values = state.get_ranges([placement.ref for placement in groups[0].placements])
    scattered = _assemble_routed_experts_range_results(
        batch,
        groups,
        [
            {
                "values": values,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "nbytes": int(values.nbytes),
            }
        ],
    )
    assert np.array_equal(scattered, dense.numpy())


def test_store_state_retires_all_attempts_for_target() -> None:
    state = RoutedExpertsStoreState("instance-a")
    value = np.zeros((5, 2, 2), dtype=np.int16)
    refs = [_ref(attempt=attempt) for attempt in (0, 1, 2)]
    future_ref = _ref(target=4, attempt=1)
    for ref in [*refs, future_ref]:
        _put_state(state, ref, value)

    assert state.retire_through(3) == {
        "retired_through": 3,
        "retired_objects": 3,
        "retired_bytes": 3 * value.nbytes,
        "remaining_objects": 1,
    }
    for ref in refs:
        with pytest.raises(KeyError, match="Missing routed-experts object"):
            state.get(
                key=routed_experts_ref_lookup_key(ref), store_instance_id="instance-a"
            )
    with pytest.raises(RuntimeError, match="retired target-weight version"):
        _put_state(state, _ref(attempt=3), value)
    assert (
        state.get(
            key=routed_experts_ref_lookup_key(future_ref),
            store_instance_id="instance-a",
        ).object_ref
        is value
    )


@pytest.mark.parametrize("attempt", [None, -1, True, False, 1.5, "1"])
def test_validate_ref_rejects_invalid_attempt_index(attempt: Any) -> None:
    with pytest.raises(
        ValueError, match="attempt_index must be a non-negative integer"
    ):
        validate_routed_experts_ref(_ref() | {"attempt_index": attempt})


def test_legacy_ref_without_attempt_matches_explicit_zero() -> None:
    ref = _ref()
    del ref["attempt_index"]
    restored = json.loads(json.dumps(ref))

    assert restored["schema"] == "nemo_rl.routed_experts_ref.v1"
    assert validate_routed_experts_ref(restored) == ref
    assert routed_experts_ref_lookup_key(restored) == routed_experts_ref_lookup_key(
        _ref(attempt=0)
    )
    assert routed_experts_ref_lookup_key(restored) != routed_experts_ref_lookup_key(
        _ref(attempt=1)
    )


def test_store_accepts_legacy_reads_but_rejects_legacy_live_inserts() -> None:
    store = RoutedExpertsObjectStore.__ray_metadata__.modified_class("instance-a")
    legacy_ref = _ref()
    del legacy_ref["attempt_index"]
    with pytest.raises(ValueError, match="inserts require an explicit attempt_index"):
        store.put_ref(legacy_ref, [object()], 40)

    zero_object, retry_object = object(), object()
    store.put_ref(_ref(attempt=0), [zero_object], 40)
    store.put_ref(_ref(attempt=1), [retry_object], 40)
    assert store.get_ref(legacy_ref)["object_ref"] is zero_object
    assert store.get_ref(_ref(attempt=1))["object_ref"] is retry_object
    assert "attempt_index" not in legacy_ref


def test_validate_ref_rejects_unknown_schema() -> None:
    with pytest.raises(
        ValueError, match="Expected a routed-experts Ray reference with schema"
    ):
        validate_routed_experts_ref(_ref() | {"schema": "unknown"})


def test_writer_requires_explicit_attempt() -> None:
    writer = RoutedExpertsStoreWriter.__new__(RoutedExpertsStoreWriter)
    with pytest.raises(TypeError, match="attempt_index"):
        writer.put(
            torch.zeros(5, 2, 2, dtype=torch.int16),
            request_id="request-a",
            task_index=11,
            rollout_index=2,
            target_weight_version=3,
        )


@pytest.mark.parametrize("attempt", [0, 2])
def test_writer_emits_attempt_in_reference(
    monkeypatch: pytest.MonkeyPatch, attempt: int
) -> None:
    writer = RoutedExpertsStoreWriter.__new__(RoutedExpertsStoreWriter)
    writer.store_name = "store-a"
    writer.store_instance_id = "instance-a"
    writer.store = MagicMock()
    object_ref = object()
    put = MagicMock(return_value=object_ref)
    monkeypatch.setattr("nemo_rl.utils.routed_experts_ref.ray.put", put)
    monkeypatch.setattr("nemo_rl.utils.routed_experts_ref.ray.get", lambda value: value)

    ref = writer.put(
        torch.zeros(5, 2, 2, dtype=torch.int16),
        request_id="request-a",
        task_index=11,
        rollout_index=2,
        attempt_index=attempt,
        target_weight_version=3,
    )

    assert ref == _ref(attempt=attempt)
    writer.store.put_ref.remote.assert_called_once_with(ref, [object_ref], 40)
    assert put.call_count == 1


def test_range_plan_and_scatter_preserve_interleaved_store_order():
    source_a = np.arange(5 * 2 * 2, dtype=np.int16).reshape(5, 2, 2)
    source_b = np.arange(4 * 2 * 2, dtype=np.int16).reshape(4, 2, 2) + 100
    ref_a = _ref(shape=(5, 2, 2)) | {"request_id": "request-a"}
    ref_b = _ref(shape=(4, 2, 2)) | {
        "store": "store-b",
        "store_instance_id": "instance-b",
        "request_id": "request-b",
    }
    refs = [
        [
            slice_routed_experts_ref(ref_a, offset=0, length=1),
            slice_routed_experts_ref(ref_b, offset=1, length=2),
            slice_routed_experts_ref(ref_a, offset=3, length=1),
        ]
    ]
    batch = _normalize_routed_experts_batch(
        refs,
        batch_size=1,
        padded_length=6,
        input_lengths=[4],
    )
    groups, stats = _plan_routed_experts_range_reads(batch)
    states = {
        "store-a": RoutedExpertsStoreState("instance-a"),
        "store-b": RoutedExpertsStoreState("instance-b"),
    }
    _put_state(states["store-a"], ref_a, source_a)
    _put_state(states["store-b"], ref_b, source_b)
    results = []
    for group in groups:
        values = states[group.store_name].get_ranges(
            [placement.ref for placement in group.placements]
        )
        results.append(
            {
                "values": values,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "nbytes": int(values.nbytes),
            }
        )

    dense = _assemble_routed_experts_range_results(batch, groups, results)

    expected = np.concatenate((source_a[0:1], source_b[1:3], source_a[3:4]))
    assert np.array_equal(dense[0, :4], expected)
    assert np.all(dense[0, 4:] == -1)
    assert stats["range_read_store_calls"] == 2
    assert stats["range_read_rows"] == 4


def test_fifty_turn_range_plan_eliminates_cumulative_prefix_reads():
    turns = 50
    rows_per_turn = 4
    final_rows = turns * rows_per_turn
    refs = []
    for turn in range(turns):
        source_rows = (turn + 1) * rows_per_turn
        full_ref = _ref(shape=(source_rows, 2, 2)) | {"request_id": f"request-{turn}"}
        if turn == 0:
            offset = 0
            length = rows_per_turn - 1
        elif turn == turns - 1:
            offset = turn * rows_per_turn - 1
            length = rows_per_turn + 1
        else:
            offset = turn * rows_per_turn - 1
            length = rows_per_turn
        refs.append(slice_routed_experts_ref(full_ref, offset=offset, length=length))

    batch = _normalize_routed_experts_batch(
        [refs],
        batch_size=1,
        padded_length=final_rows,
        input_lengths=[final_rows],
    )
    groups, stats = _plan_routed_experts_range_reads(batch)

    full_source_rows = rows_per_turn * turns * (turns + 1) // 2
    bytes_per_row = 2 * 2 * 2
    assert len(groups) == 1
    assert groups[0].requested_rows == final_rows
    assert stats == {
        "range_read_requests": 1,
        "range_read_store_calls": 1,
        "range_read_source_objects": turns,
        "range_read_segments": turns,
        "range_read_rows": final_rows,
        "range_read_bytes": final_rows * bytes_per_row,
        "full_source_rows_equivalent": full_source_rows,
        "full_source_bytes_equivalent": full_source_rows * bytes_per_row,
        "range_read_avoided_bytes": (full_source_rows - final_rows) * bytes_per_row,
    }
    assert full_source_rows / final_rows == 25.5


def test_ray_transport_uses_direct_object_path_for_unsliced_ref(monkeypatch):
    source = np.arange(5 * 2 * 2, dtype=np.int16).reshape(5, 2, 2)
    batch = _normalize_routed_experts_batch(
        [[_ref()]],
        batch_size=1,
        padded_length=5,
        input_lengths=[5],
    )
    resolve = MagicMock(return_value=source)
    materialize_ranges = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_ref._resolve_routed_experts_ref_with_ray",
        resolve,
    )
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_ref._materialize_normalized_routed_experts_with_ray_ranges",
        materialize_ranges,
    )

    dense, stats = _materialize_normalized_routed_experts_with_ray_transport(batch)

    resolve.assert_called_once_with(batch.refs_by_sample[0][0])
    materialize_ranges.assert_not_called()
    assert np.shares_memory(dense, source)
    assert stats
    assert all(value == 0 for value in stats.values())


def test_ray_transport_uses_range_path_for_sliced_ref(monkeypatch):
    sliced_ref = slice_routed_experts_ref(_ref(), offset=1, length=3)
    batch = _normalize_routed_experts_batch(
        [[sliced_ref]],
        batch_size=1,
        padded_length=3,
        input_lengths=[3],
    )
    expected = np.arange(3 * 2 * 2, dtype=np.int16).reshape(1, 3, 2, 2)
    expected_stats = {"range_read_requests": 1}
    resolve = MagicMock()
    materialize_ranges = MagicMock(return_value=(expected, expected_stats))
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_ref._resolve_routed_experts_ref_with_ray",
        resolve,
    )
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_ref._materialize_normalized_routed_experts_with_ray_ranges",
        materialize_ranges,
    )

    dense, stats = _materialize_normalized_routed_experts_with_ray_transport(batch)

    resolve.assert_not_called()
    materialize_ranges.assert_called_once_with(batch)
    assert dense is expected
    assert stats is expected_stats


@pytest.mark.parametrize(
    ("input_lengths", "error_type", "match"),
    [
        (torch.tensor([[5]]), ValueError, "one-dimensional"),
        ([True], TypeError, "must be integers"),
        ([-1], ValueError, "must be non-negative"),
    ],
)
def test_normalize_batch_rejects_invalid_input_lengths(
    input_lengths, error_type, match
):
    with pytest.raises(error_type, match=match):
        _normalize_routed_experts_batch(
            [[_ref()]],
            batch_size=1,
            padded_length=5,
            input_lengths=input_lengths,
        )


def test_normalize_batch_rejects_sample_longer_than_padded_output():
    with pytest.raises(ValueError, match="exceeds the padded output length"):
        _normalize_routed_experts_batch(
            [[_ref(shape=(6, 2, 2))]],
            batch_size=1,
            padded_length=5,
            input_lengths=[6],
        )


def test_store_state_rejects_nbytes_inconsistent_with_shape():
    state = RoutedExpertsStoreState("instance-a")

    with pytest.raises(ValueError, match="byte size does not match"):
        state.put(
            key=routed_experts_ref_lookup_key(_ref()),
            object_ref=np.zeros((5, 2, 2), dtype=np.int16),
            nbytes=39,
            shape=(5, 2, 2),
            dtype="int16",
        )


def test_materialize_refs_rejects_incomplete_sample_coverage():
    with pytest.raises(ValueError, match="do not cover"):
        materialize_routed_experts_refs(
            [[slice_routed_experts_ref(_ref(), offset=0, length=4)]],
            input_ids=torch.zeros(1, 5, dtype=torch.long),
            input_lengths=torch.tensor([5], dtype=torch.int32),
            resolver=lambda ref: np.zeros((5, 2, 2), dtype=np.int16),
        )


def test_store_state_retire_through_is_monotonic_and_rejects_late_puts():
    state = RoutedExpertsStoreState("instance-a")
    target_three = _ref(target=3)
    target_four = _ref(target=4)
    target_four["request_id"] = "request-b"
    value_three = np.zeros((5, 2, 2), dtype=np.int16)
    value_four = np.zeros((5, 3, 2), dtype=np.int16)
    _put_state(state, target_three, value_three)
    _put_state(state, target_four, value_four)

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
        is value_four
    )
    with pytest.raises(RuntimeError, match="retired target-weight version"):
        state.put(
            key=routed_experts_ref_lookup_key(target_three),
            object_ref=value_three,
            nbytes=int(value_three.nbytes),
            shape=value_three.shape,
            dtype=str(value_three.dtype),
        )


@pytest.mark.parametrize("run_instance_id", [None, "run-a"])
def test_retire_routed_experts_through_skips_disabled_ray_transport(
    run_instance_id: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    get_actor = MagicMock()
    ray_get = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_ref.ray.get_actor",
        get_actor,
    )
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_ref.ray.get",
        ray_get,
    )
    router_replay = {"enabled": False, "transport": "ray"}
    if run_instance_id is not None:
        router_replay["_store_run_instance_id"] = run_instance_id

    result = retire_routed_experts_through(
        {"router_replay": router_replay},
        target_weight_version=3,
    )

    assert result is None
    get_actor.assert_not_called()
    ray_get.assert_not_called()


def test_retire_routed_experts_through_skips_inline_transport(monkeypatch):
    get_actor = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_ref.ray.get_actor",
        get_actor,
    )

    result = retire_routed_experts_through(
        {"router_replay": {"enabled": True}},
        target_weight_version=3,
    )

    assert result is None
    get_actor.assert_not_called()


def test_retire_routed_experts_through_dispatches_to_run_registry(monkeypatch):
    retired = {
        "retired_through": 7,
        "stores": 2,
        "retired_objects": 8,
        "retired_bytes": 160,
        "remaining_objects": 4,
    }
    retire_ref = object()
    registry = MagicMock()
    registry.retire_through.remote.return_value = retire_ref
    get_actor = MagicMock(return_value=registry)
    ray_get = MagicMock(return_value=retired)
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_ref.ray.get_actor",
        get_actor,
    )
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_ref.ray.get",
        ray_get,
    )

    result = retire_routed_experts_through(
        {
            "router_replay": {
                "enabled": True,
                "transport": "ray",
                "_store_run_instance_id": "run-a",
            }
        },
        target_weight_version=7,
    )

    assert result == retired
    get_actor.assert_called_once_with(
        registry_actor_name("run-a"),
        namespace=ROUTED_EXPERTS_RAY_NAMESPACE,
    )
    registry.retire_through.remote.assert_called_once_with(7)
    ray_get.assert_called_once_with(retire_ref)


def test_retire_routed_experts_through_requires_run_instance_id():
    with pytest.raises(RuntimeError, match="missing its store run instance id"):
        retire_routed_experts_through(
            {"router_replay": {"enabled": True, "transport": "ray"}},
            target_weight_version=3,
        )

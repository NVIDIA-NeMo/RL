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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from nemo_rl.utils.routed_experts_ref import (
    ROUTED_EXPERTS_REF_DTYPE,
    ROUTED_EXPERTS_REF_KEY,
    ROUTED_EXPERTS_REF_SCHEMA,
    ROUTED_EXPERTS_RAY_NAMESPACE,
    RoutedExpertsStoreState,
    materialize_routed_experts_refs,
    registry_actor_name,
    retire_routed_experts_through,
    routed_experts_ref_lookup_key,
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

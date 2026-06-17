# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# ---------------------------------------------------------------------------
# Mock teacher worker group for _compute_teacher_logprobs tests
# ---------------------------------------------------------------------------


class _MockShardingAnnotations:
    def __init__(self, dp_size):
        self._dp_size = dp_size

    def get_axis_size(self, name):
        if name == "data_parallel":
            return self._dp_size
        return 1


class _MockTeacherWorkerGroup:
    """Returns logprobs filled with a constant; validates DP-divisible batch."""

    def __init__(self, fill_value=1.0, dp_size=4):
        self._fill_value = fill_value
        self.sharding_annotations = _MockShardingAnnotations(dp_size)

    def get_logprobs(self, data):
        input_ids = data["input_ids"]
        B, S = input_ids.shape
        # Verify the caller already padded to dp_size
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        assert B % dp_size == 0, (
            f"get_logprobs received batch_size={B} not divisible by dp_size={dp_size}"
        )
        return BatchedDataDict(
            {"reference_logprobs": torch.full((B, S), self._fill_value)}
        )


def _make_collector(**overrides):
    """Build a bare AsyncTrajectoryCollector (bypass Ray) for unit testing."""
    import threading

    from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector

    # AsyncTrajectoryCollector is @ray.remote-decorated; unwrap to the real class.
    real_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    defaults = {
        "teacher_worker_groups": {},
        "alias_to_group_alias": {},
        "on_policy_distillation_cfg": {},
        "_has_non_colocated_teachers": False,
    }
    defaults.update(overrides)
    obj = object.__new__(real_cls)
    for k, v in defaults.items():
        setattr(obj, k, v)
    obj._teacher_locks = {k: threading.Lock() for k in obj.teacher_worker_groups}
    return obj


# ---------------------------------------------------------------------------
# DP padding tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,dp_size",
    [
        (1, 4),  # the exact bug: 1 sample, dp=4
        (2, 4),  # 2 samples, dp=4
        (3, 4),  # 3 samples, dp=4
        (4, 4),  # already aligned
        (1, 8),  # extreme: 1 sample, dp=8
        (5, 4),  # 5 samples → pad to 8
    ],
)
def test_compute_teacher_logprobs_dp_padding(batch_size, dp_size):
    """Teacher logprob computation must pad batch to dp_size multiple."""
    twg = _MockTeacherWorkerGroup(fill_value=2.0, dp_size=dp_size)
    collector = _make_collector(
        teacher_worker_groups={"math": twg},
        alias_to_group_alias={"math_agent": "math"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"math_agent": "/ckpt/math"},
        },
        _has_non_colocated_teachers=True,
    )

    S = 16
    input_ids = torch.randint(0, 100, (batch_size, S))
    agent_refs = [{"name": "math_agent"}] * batch_size

    result, _ = collector._compute_teacher_logprobs(input_ids, agent_refs)

    assert result.shape == (batch_size, S)
    assert torch.allclose(result, torch.tensor(2.0))


def test_compute_teacher_logprobs_routes_to_correct_teacher():
    """Samples are routed to the right teacher and results stitched back."""
    math_twg = _MockTeacherWorkerGroup(fill_value=1.0, dp_size=1)
    code_twg = _MockTeacherWorkerGroup(fill_value=2.0, dp_size=1)

    collector = _make_collector(
        teacher_worker_groups={"math": math_twg, "code": code_twg},
        alias_to_group_alias={"math_agent": "math", "code_agent": "code"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {
                "math_agent": "/ckpt/math",
                "code_agent": "/ckpt/code",
            },
        },
        _has_non_colocated_teachers=True,
    )

    B, S = 4, 8
    input_ids = torch.randint(0, 100, (B, S))
    agent_refs = [
        {"name": "math_agent"},
        {"name": "code_agent"},
        {"name": "math_agent"},
        {"name": "code_agent"},
    ]

    result, _ = collector._compute_teacher_logprobs(input_ids, agent_refs)

    assert result.shape == (B, S)
    assert torch.allclose(result[0], torch.tensor(1.0))
    assert torch.allclose(result[1], torch.tensor(2.0))
    assert torch.allclose(result[2], torch.tensor(1.0))
    assert torch.allclose(result[3], torch.tensor(2.0))


def test_compute_teacher_logprobs_deduplication():
    """alias_to_group_alias routes multiple aliases to one teacher group."""
    shared_twg = _MockTeacherWorkerGroup(fill_value=3.0, dp_size=1)

    collector = _make_collector(
        teacher_worker_groups={"primary": shared_twg},
        alias_to_group_alias={"mcqa": "primary", "terminal": "primary"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {
                "mcqa": "/ckpt/shared",
                "terminal": "/ckpt/shared",
            },
        },
        _has_non_colocated_teachers=True,
    )

    B, S = 2, 4
    input_ids = torch.randint(0, 100, (B, S))
    agent_refs = [{"name": "mcqa"}, {"name": "terminal"}]

    result, _ = collector._compute_teacher_logprobs(input_ids, agent_refs)
    assert result.shape == (B, S)
    assert torch.allclose(result, torch.tensor(3.0))


# ---------------------------------------------------------------------------
# Unsort / reorder_data regression test
# ---------------------------------------------------------------------------


def test_reorder_data_vs_direct_gather():
    """Verify reorder_data inverts the permutation, while direct gather does not.

    This is the root cause of the num_gen>1 teacher logprob misalignment bug:
    shard_by_batch_size returns a forward permutation (sorted_pos → orig_idx).
    To restore original order we need the *inverse* (argsort), which
    reorder_data computes.  A direct gather ``result[indices]`` applies
    the forward permutation and silently produces wrong results.
    """
    # Simulate: 4 samples reordered by sequence packing as [3, 0, 2, 1]
    forward_perm = [3, 0, 2, 1]
    # After inference, results are in sorted order:
    #   position 0 = result for orig sample 3
    #   position 1 = result for orig sample 0  etc.
    sorted_results = BatchedDataDict(
        {"logprobs": torch.tensor([[30.0], [0.0], [20.0], [10.0]])}
    )
    # label: sorted_results[i] holds the value for original sample forward_perm[i]
    #   sorted_results[0]=30 → orig 3,  sorted_results[1]=0 → orig 0, etc.

    # --- WRONG: direct gather (the old bug) ---
    wrong = sorted_results["logprobs"][forward_perm]
    # wrong[0] = sorted_results[3] = 10  (should be 0 for orig 0)
    assert not torch.equal(wrong, torch.tensor([[0.0], [10.0], [20.0], [30.0]])), (
        "Direct gather should NOT produce the correct original order"
    )

    # --- CORRECT: reorder_data (inverse permutation) ---
    correct = BatchedDataDict({"logprobs": sorted_results["logprobs"].clone()})
    correct.reorder_data(forward_perm)
    assert torch.equal(
        correct["logprobs"], torch.tensor([[0.0], [10.0], [20.0], [30.0]])
    ), "reorder_data should restore the original sample order"


# ---------------------------------------------------------------------------
# Teacher logprob alignment with variable-length sequences (num_gen > 1)
# ---------------------------------------------------------------------------


def test_reorder_data_inverse_permutation_various():
    """reorder_data correctly inverts arbitrary permutations, including identity."""
    # Identity permutation
    bdd = BatchedDataDict({"x": torch.tensor([[0.0], [1.0], [2.0]])})
    bdd.reorder_data([0, 1, 2])
    assert torch.equal(bdd["x"], torch.tensor([[0.0], [1.0], [2.0]]))

    # Reversal
    bdd = BatchedDataDict({"x": torch.tensor([[0.0], [1.0], [2.0]])})
    bdd.reorder_data([2, 1, 0])
    # batch_sorted_indices=[2,1,0] means sorted[0] came from orig 2, etc.
    # Inverse: orig[2]=sorted[0]=0.0, orig[1]=sorted[1]=1.0, orig[0]=sorted[2]=2.0
    assert torch.equal(bdd["x"], torch.tensor([[2.0], [1.0], [0.0]]))

    # Non-trivial: simulate 4 samples reordered as [2, 3, 0, 1]
    bdd = BatchedDataDict({"x": torch.tensor([[20.0], [30.0], [0.0], [10.0]])})
    bdd.reorder_data([2, 3, 0, 1])
    assert torch.equal(bdd["x"], torch.tensor([[0.0], [10.0], [20.0], [30.0]])), (
        "After reorder_data, row i should hold the result for original sample i"
    )


def test_is_opd_enabled():
    from nemo_rl.algorithms.opd import is_opd_enabled

    assert is_opd_enabled({"on_policy_distillation": {"enabled": True}})
    assert not is_opd_enabled({"on_policy_distillation": {"enabled": False}})
    assert not is_opd_enabled({})


def test_is_opd_enabled_object_config():
    # _opd_cfg must also handle a config object (not just a dict): math recipes
    # have no on_policy_distillation attribute at all.
    from types import SimpleNamespace

    from nemo_rl.algorithms.opd import is_opd_enabled

    assert is_opd_enabled(SimpleNamespace(on_policy_distillation={"enabled": True}))
    assert not is_opd_enabled(
        SimpleNamespace(on_policy_distillation={"enabled": False})
    )
    assert not is_opd_enabled(SimpleNamespace())
    assert not is_opd_enabled(SimpleNamespace(on_policy_distillation=None))


def test_is_non_colocated_teachers_enabled():
    from nemo_rl.algorithms.opd import is_non_colocated_teachers_enabled

    assert is_non_colocated_teachers_enabled(
        {
            "on_policy_distillation": {
                "enabled": True,
                "non_colocated_teachers": {"enabled": True},
            }
        }
    )
    assert not is_non_colocated_teachers_enabled(
        {
            "on_policy_distillation": {
                "enabled": True,
                "non_colocated_teachers": {"enabled": False},
            }
        }
    )


def test_resolve_reference_aliases_bad_agent_ref():
    from nemo_rl.algorithms.opd import resolve_reference_aliases

    with pytest.raises(KeyError):
        resolve_reference_aliases([{"not_name": "oops"}], {"math": "/ckpt/math"})


def test_resolve_reference_aliases_fallback():
    from nemo_rl.algorithms.opd import resolve_reference_aliases

    aliases = resolve_reference_aliases(
        [{"name": "math_agent"}, {"name": "unknown"}, {"name": "code_agent"}],
        {"math_agent": "/ckpt/math", "code_agent": "/ckpt/code"},
        default_teacher_alias="math_agent",
    )
    assert aliases == ["math_agent", "math_agent", "code_agent"]


def test_resolve_reference_aliases_strict_raises():
    from nemo_rl.algorithms.opd import resolve_reference_aliases

    with pytest.raises(ValueError, match="No teacher model mapping"):
        resolve_reference_aliases(
            [{"name": "unknown"}], {"math": "/ckpt/math"}, strict_agent_name_match=True
        )


def test_get_teacher_routing_metrics():
    from nemo_rl.algorithms.opd import get_teacher_routing_metrics

    metrics = get_teacher_routing_metrics(
        ["math_a", "math_b", "if", "math_a"],
        {"math_a": "t_math", "math_b": "t_math", "if": "t_if"},
    )
    assert metrics["on_policy_distillation/teacher_alias_unique"] == 3.0
    assert metrics["on_policy_distillation/teacher_model_unique"] == 2.0


# ---------------------------------------------------------------------------
# teacher_seq_pad_multiple: teacher pre-pad multiple + packing-mode guard
# ---------------------------------------------------------------------------


def _twg(packed, pad_multiple=1):
    """TeacherWorkerGroup stand-in with the two attrs the helper reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        use_sequence_packing=packed, sequence_length_pad_multiple=pad_multiple
    )


def test_teacher_seq_pad_multiple_no_teachers_is_one():
    from nemo_rl.algorithms.opd import teacher_seq_pad_multiple

    assert teacher_seq_pad_multiple({}, 8) == 1


def test_teacher_seq_pad_multiple_all_packed_is_one():
    from nemo_rl.algorithms.opd import teacher_seq_pad_multiple

    assert teacher_seq_pad_multiple({"a": _twg(True), "b": _twg(True)}, 8) == 1


def test_teacher_seq_pad_multiple_mixed_packing_raises():
    from nemo_rl.algorithms.opd import teacher_seq_pad_multiple

    with pytest.raises(ValueError, match="same sequence-packing mode"):
        teacher_seq_pad_multiple({"a": _twg(True), "b": _twg(False)}, 8)


def test_teacher_seq_pad_multiple_non_packed_uses_policy_divisor():
    # Non-packed teachers pre-pad to the policy divisor when it is a multiple of
    # every teacher's requirement (here 2 and 4 both divide 8).
    from nemo_rl.algorithms.opd import teacher_seq_pad_multiple

    teachers = {"a": _twg(False, pad_multiple=2), "b": _twg(False, pad_multiple=4)}
    assert teacher_seq_pad_multiple(teachers, 8) == 8


def test_teacher_seq_pad_multiple_non_packed_incompatible_divisor_raises():
    # policy divisor 8 is not a multiple of the teacher requirement 16.
    from nemo_rl.algorithms.opd import teacher_seq_pad_multiple

    with pytest.raises(ValueError, match="make_sequence_length_divisible_by"):
        teacher_seq_pad_multiple({"a": _twg(False, pad_multiple=16)}, 8)


# ---------------------------------------------------------------------------
# Teacher-logprob seq-length padding + the "opd" advantage-estimator branch
# ---------------------------------------------------------------------------


def test_pad_teacher_logprobs():
    from nemo_rl.algorithms.grpo import _pad_teacher_logprobs

    # teacher_S < train_S -> right zero-pad
    padded = _pad_teacher_logprobs(torch.ones(2, 3), 5)
    assert padded.shape == (2, 5)
    assert (padded[:, :3] == 1).all() and (padded[:, 3:] == 0).all()
    # teacher_S == train_S -> unchanged
    assert _pad_teacher_logprobs(torch.ones(2, 4), 4).shape == (2, 4)
    # teacher_S > train_S -> raises
    with pytest.raises(ValueError, match="seq length"):
        _pad_teacher_logprobs(torch.ones(2, 6), 4)


def test_create_advantage_estimator_opd_branch():
    import warnings
    from types import SimpleNamespace

    from nemo_rl.algorithms.advantage_estimator import OPDAdvantageEstimator
    from nemo_rl.algorithms.grpo import _create_advantage_estimator

    # loss_fn not MOPD-configured -> the 3 recommendation warnings fire.
    loss_fn = SimpleNamespace(
        disable_ppo_ratio=False,
        use_importance_sampling_correction=False,
        truncated_importance_sampling_type="none",
    )
    master_config = SimpleNamespace(
        grpo={"adv_estimator": {"name": "opd"}}, loss_fn=loss_fn
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator = _create_advantage_estimator(master_config)
    assert isinstance(estimator, OPDAdvantageEstimator)
    assert len(caught) == 3

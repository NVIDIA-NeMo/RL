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
"""Stage-level tests of ``_advantage_stage`` with token-capture segment rows.

Drives the real ``SingleControllerActor._advantage_stage`` (``object.__new__``
on the Ray-modified class, an in-memory data plane, the real
``GRPOAdvantageEstimator``) with tagged ``KVBatchMeta`` rows in the
finalizer's layout: canonical rows ``{group}_g{i}`` first, extra rows
``{group}_g{i}_t{j}`` after, every row tagged with ``rollout_local_idx`` /
``trace_in_rollout_idx`` / ``trace_kind`` / ``segment_index`` /
``rows_in_rollout``.

What the stage must do with extra rows (see
``single_controller_utils/segment_stats.py``):

* the estimator sees one row per rollout (the canonical rows), so extra rows
  never change a baseline or count as siblings;
* every row of a rollout receives the rollout's advantage;
* every row keeps its own loss mask (overlong, seq-logprob gate per row; the
  env flag is copied to all rows by the finalizer);
* the rollout's vote in the baseline is the ``amax`` of its rows' baseline
  masks, so a rollout that trains through a segment row while its canonical
  row is gate-masked still votes (LOO-correct baseline);
* orphans, duplicate canonicals, non-GRPO estimators and incomplete chunks
  fail loudly;
* PPO instead runs GAE on every row (each row has its own critic values), and
  its DP pad rows are left out of the rollout identity and checks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GAEConfig,
    GDPOAdvantageEstimator,
    GeneralizedAdvantageEstimator,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    SegmentRowsConfig,
    TokenCaptureConfig,
)
from nemo_rl.algorithms.single_controller_utils.masking_stats import (
    new_masking_stats_accumulator,
)
from nemo_rl.algorithms.single_controller_utils.rollout_stats import (
    new_rollout_stats_accumulator,
)
from nemo_rl.algorithms.single_controller_utils.segment_stats import (
    new_segment_stats_accumulator,
    reduce_segment_stats,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import INPUT_LENGTHS

SEQ = 6
PROMPT = 3
# |generation - prev| per token; exp(1.0) = 2.718 trips seq_logprob_error_threshold=2.
GATE_ERROR = 1.0


@dataclass
class _Row:
    """One published row in the finalizer's vocabulary."""

    group: str
    rollout: int
    trace: int = 0
    kind: str = "terminal"
    reward: float = 0.0
    tokens: int = 4  # generated tokens; token_mask ones at the tail of SEQ
    mask_sample: bool = False
    truncated: bool = False
    gate_error: float = 0.0
    sample_mask: float = 1.0
    # None: computed from the rows handed to _batch; set to lie in a tag.
    rows_in_rollout: Optional[int] = None
    # None: {group}_g{i}[_t{j}]; set to publish under a different id.
    sample_id: Optional[str] = None

    @property
    def id(self) -> str:
        if self.sample_id is not None:
            return self.sample_id
        base = f"{self.group}_g{self.rollout}"
        return base if self.trace == 0 else f"{base}_t{self.trace}"


class _DataPlane:
    """In-memory stand-in: hands back one TensorDict, records the write-back."""

    def __init__(self, data: TensorDict) -> None:
        self._data = data
        self.selected_fields: list[str] | None = None
        self.written: TensorDict | None = None

    def get_samples(self, *, select_fields, **kwargs):
        del kwargs
        self.selected_fields = list(select_fields)
        return self._data

    def put_samples(self, *, fields, **kwargs) -> None:
        del kwargs
        self.written = fields


def _batch(rows: list[_Row]) -> tuple[TensorDict, KVBatchMeta]:
    n = len(rows)
    groups = list(dict.fromkeys(row.group for row in rows))
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        counts[(row.group, row.rollout)] = counts.get((row.group, row.rollout), 0) + 1
    prompt_ids = torch.zeros(n, PROMPT, dtype=torch.long)
    token_mask = torch.zeros(n, SEQ)
    gen = torch.zeros(n, SEQ)
    for r, row in enumerate(rows):
        prompt_ids[r] = groups.index(row.group) + 1
        if row.tokens:
            token_mask[r, SEQ - row.tokens :] = 1.0
        gen[r, 1:] = row.gate_error
    data = TensorDict(
        {
            "prompt_ids_for_adv": prompt_ids,
            "total_reward": torch.tensor([row.reward for row in rows]),
            "token_mask": token_mask,
            "sample_mask": torch.tensor([row.sample_mask for row in rows]),
            "mask_sample": torch.tensor([row.mask_sample for row in rows]),
            "truncated": torch.tensor([row.truncated for row in rows]),
            "prev_logprobs": torch.zeros(n, SEQ),
            "generation_logprobs": gen,
            INPUT_LENGTHS: torch.full((n,), SEQ, dtype=torch.long),
        },
        batch_size=[n],
    )
    tags = [
        {
            "weight_version": 1,
            "rollout_local_idx": row.rollout,
            "trace_in_rollout_idx": row.trace,
            "trace_kind": row.kind,
            "segment_index": row.trace,
            "rows_in_rollout": (
                row.rows_in_rollout
                if row.rows_in_rollout is not None
                else counts[(row.group, row.rollout)]
            ),
        }
        for row in rows
    ]
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[row.id for row in rows],
        fields=list(data.keys()),
        tags=tags,
    )
    return data, meta


def _grpo_estimator() -> GRPOAdvantageEstimator:
    return GRPOAdvantageEstimator(
        AdvEstimatorConfig(normalize_rewards=True, use_leave_one_out_baseline=True),
        ClippedPGLossConfig(),
    )


def _controller(
    plane: _DataPlane,
    *,
    num_generations_per_prompt: int,
    estimator=None,
    seq_logprob_error_threshold: float | None = 2.0,
    overlong_filtering: bool = False,
    masked_sample_rewards_in_baseline: bool = False,
    segment_rows_enabled: bool = True,
    is_ppo: bool = False,
):
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator if estimator is not None else _grpo_estimator()
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._is_ppo = is_ppo
    grpo = GRPOConfig(
        num_generations_per_prompt=num_generations_per_prompt,
        seq_logprob_error_threshold=seq_logprob_error_threshold,
        overlong_filtering=overlong_filtering,
        masked_sample_rewards_in_baseline=masked_sample_rewards_in_baseline,
    )
    ctrl._master_config = SimpleNamespace(
        grpo=grpo,
        token_capture=TokenCaptureConfig(
            enabled=True,
            segment_rows=SegmentRowsConfig(
                enabled=segment_rows_enabled,
                max_per_rollout=6 if segment_rows_enabled else 1,
            ),
        ),
    )
    ctrl._algo_cfg = grpo
    ctrl._message_level_advantage_penalties_enabled = False
    ctrl._step_log_dict = {
        "rewards": [],
        "sample_masks": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "num_mask_sample_filtered": [],
        "seq_logprob_error_metrics": [],
        "canonical_masks": [],
    }
    ctrl._rollout_stats_acc = new_rollout_stats_accumulator()
    ctrl._masking_stats_acc = new_masking_stats_accumulator()
    ctrl._segment_stats_acc = new_segment_stats_accumulator()
    return ctrl


def _run(rows: list[_Row], **controller_kwargs):
    data, meta = _batch(rows)
    plane = _DataPlane(data)
    ctrl = _controller(plane, **controller_kwargs)
    result_meta, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert plane.written is not None
    return plane, ctrl, result_meta, valid


def _written(plane: _DataPlane, key: str) -> torch.Tensor:
    assert plane.written is not None
    return torch.as_tensor(plane.written[key])


def _direct(rows: list[_Row], valid_mask: torch.Tensor) -> torch.Tensor:
    """The estimator on the canonical rows alone (one scalar per row)."""
    canonical = [row for row in rows if row.trace == 0]
    data, _ = _batch(canonical)
    adv = _grpo_estimator().compute_advantage(
        prompt_ids=data["prompt_ids_for_adv"],
        rewards=data["total_reward"],
        mask=data["token_mask"],
        valid_mask=valid_mask,
    )
    return adv[:, 0]


def _scalar(adv: torch.Tensor) -> torch.Tensor:
    """GRPO advantages are constant along the sequence; read one value per row."""
    assert torch.equal(adv, adv[:, :1].expand_as(adv))
    return adv[:, 0]


# ── baseline / broadcast ───────────────────────────────────────────────────


def _three_rollouts_with_extras() -> list[_Row]:
    return [
        _Row("a", 0, reward=1.0),
        _Row("a", 1, reward=0.0),
        _Row("a", 2, reward=0.0),
        _Row("a", 0, 1, "compaction_segment", reward=1.0, tokens=3),
        _Row("a", 0, 2, "compaction_summary", reward=1.0, tokens=2),
        _Row("a", 1, 1, "subagent", reward=0.0, tokens=5),
    ]


def test_extra_rows_do_not_change_the_baseline_and_inherit_the_rollout_advantage():
    rows = _three_rollouts_with_extras()
    plane, ctrl, result_meta, valid = _run(rows, num_generations_per_prompt=3)
    assert valid and "advantages" in (result_meta.fields or [])
    adv = _scalar(_written(plane, "advantages"))

    expected = _direct(rows, torch.ones(3))
    torch.testing.assert_close(adv[:3], expected)
    # Extra rows copy their rollout's advantage exactly.
    assert torch.equal(adv[3], adv[0]) and torch.equal(adv[4], adv[0])
    assert torch.equal(adv[5], adv[1])
    # Reward 1 against siblings 0/0 is positive; the extras did not dilute it.
    assert adv[0] > 0 and adv[1] < 0 and adv[2] < 0
    # No mask changed, so sample_mask was not written back.
    assert "sample_mask" not in plane.written.keys()
    # Step logs count rows, canonical_masks marks the canonical ones.
    assert ctrl._step_log_dict["canonical_masks"][0].tolist() == [1, 1, 1, 0, 0, 0]
    assert ctrl._step_log_dict["sample_masks"][0].tolist() == [1.0] * 6
    assert ctrl._segment_stats_acc["rollout_key"][0].tolist() == [0, 1, 2, 0, 0, 1]


def test_canonical_advantages_are_bit_identical_with_and_without_extras():
    rows = _three_rollouts_with_extras()
    with_extras, _, _, _ = _run(rows, num_generations_per_prompt=3)
    without, _, _, _ = _run([r for r in rows if r.trace == 0], num_generations_per_prompt=3)
    assert torch.equal(
        _written(with_extras, "advantages")[:3], _written(without, "advantages")
    )


def test_single_row_rollouts_are_bit_identical_with_segment_rows_on_and_off():
    rows = [
        _Row("a", 0, reward=1.0),
        _Row("a", 1, reward=0.0, truncated=True),
        _Row("b", 0, reward=0.5, mask_sample=True),
        _Row("b", 1, reward=0.0, gate_error=GATE_ERROR),
    ]
    on, _, _, _ = _run(rows, num_generations_per_prompt=2, overlong_filtering=True)
    off, _, _, _ = _run(
        rows,
        num_generations_per_prompt=2,
        overlong_filtering=True,
        segment_rows_enabled=False,
    )
    assert torch.equal(_written(on, "advantages"), _written(off, "advantages"))
    assert torch.equal(_written(on, "sample_mask"), _written(off, "sample_mask"))
    assert _written(on, "sample_mask").tolist() == [1.0, 0.0, 0.0, 0.0]


# ── per-row masks ──────────────────────────────────────────────────────────


def test_overlong_filtering_masks_only_the_truncated_row():
    rows = [
        _Row("a", 0, reward=1.0),
        _Row("a", 1, reward=0.0),
        _Row("a", 2, reward=0.0),
        _Row("a", 0, 1, "compaction_segment", reward=1.0, truncated=True),
    ]
    plane, _, _, valid = _run(rows, num_generations_per_prompt=3, overlong_filtering=True)
    assert valid
    # Opposite of an owner-level amin: the canonical row keeps training.
    assert _written(plane, "sample_mask").tolist() == [1.0, 1.0, 1.0, 0.0]
    adv = _scalar(_written(plane, "advantages"))
    torch.testing.assert_close(adv[:3], _direct(rows, torch.ones(3)))
    assert torch.equal(adv[3], adv[0])

    # The mirror image: the canonical row is truncated, the segment is not.
    # The rollout trains through the segment and votes through it (amax).
    rows[0].truncated, rows[3].truncated = True, False
    plane, _, _, valid = _run(rows, num_generations_per_prompt=3, overlong_filtering=True)
    assert valid
    assert _written(plane, "sample_mask").tolist() == [0.0, 1.0, 1.0, 1.0]
    adv = _scalar(_written(plane, "advantages"))
    torch.testing.assert_close(adv[:3], _direct(rows, torch.ones(3)))
    assert torch.equal(adv[3], adv[0])


def test_env_mask_sample_removes_every_row_of_the_rollout():
    """The finalizer copies mask_sample to all rows; the stage masks them all.

    Without ``masked_sample_rewards_in_baseline`` the rollout then does not
    vote (no row is baseline-valid); with it the rollout keeps voting through
    its reinstated rows while still training nothing.
    """
    rows = [
        _Row("a", 0, reward=1.0, mask_sample=True),
        _Row("a", 1, reward=0.0),
        _Row("a", 2, reward=0.0),
        _Row("a", 3, reward=1.0),
        _Row("a", 0, 1, "compaction_segment", reward=1.0, mask_sample=True),
    ]
    plane, ctrl, _, valid = _run(rows, num_generations_per_prompt=4)
    assert valid
    assert _written(plane, "sample_mask").tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    adv = _scalar(_written(plane, "advantages"))
    torch.testing.assert_close(adv[1:4], _direct(rows, torch.tensor([0.0, 1, 1, 1]))[1:])
    assert ctrl._step_log_dict["num_mask_sample_filtered"] == [2]

    plane, _, _, _ = _run(
        rows, num_generations_per_prompt=4, masked_sample_rewards_in_baseline=True
    )
    assert _written(plane, "sample_mask").tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    adv = _scalar(_written(plane, "advantages"))
    torch.testing.assert_close(adv[:4], _direct(rows, torch.ones(4)))


def test_gated_canonical_with_passing_extra_votes_through_the_extra():
    """refute-multirow-grpo F2 / pr-rl-4124 A-1 at stage level.

    Rewards [1, 1, 0, 0]; rollout 0's canonical row exceeds the seq-logprob
    threshold, its segment row does not. The segment trains, so rollout 0
    votes: its LOO baseline is mean(1, 0, 0) = 1/3 and the advantage handed
    to the segment is (1 - 1/3) / 0.57735 = 1.1547 -- not the 0.7071 the
    estimator assigns to a non-voting row (baseline sum(valid)/(num_valid-1)
    = 0.5), and the siblings see rollout 0's reward (1.1547 / -1.1547 rather
    than 1.0 / -0.7071).
    """
    rows = [
        _Row("a", 0, reward=1.0, gate_error=GATE_ERROR),
        _Row("a", 1, reward=1.0),
        _Row("a", 2, reward=0.0),
        _Row("a", 3, reward=0.0),
        _Row("a", 0, 1, "compaction_segment", reward=1.0),
    ]
    plane, ctrl, _, valid = _run(rows, num_generations_per_prompt=4)
    assert valid
    assert _written(plane, "sample_mask").tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    adv = _scalar(_written(plane, "advantages"))
    expected = torch.tensor([1.154698, 1.154698, -1.154699, -1.154699])
    torch.testing.assert_close(adv[:4], expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(adv[:4], _direct(rows, torch.ones(4)))
    assert adv[4].item() == pytest.approx(1.154698, abs=1e-5)
    # What voting through the canonical row alone would have produced.
    canonical_only = _direct(rows, torch.tensor([0.0, 1, 1, 1]))
    assert canonical_only[0].item() == pytest.approx(0.707107, abs=1e-5)
    assert not torch.allclose(adv[:4], canonical_only)
    gate = ctrl._step_log_dict["seq_logprob_error_metrics"][0]
    assert gate["num_masked_seqs_by_logprob_error"] == 1


def test_rollout_with_every_row_gated_does_not_vote_but_the_chunk_still_trains():
    rows = [
        _Row("a", 0, reward=1.0, gate_error=GATE_ERROR),
        _Row("a", 1, reward=1.0),
        _Row("a", 2, reward=0.0),
        _Row("a", 3, reward=0.0),
        _Row("a", 0, 1, "compaction_segment", reward=1.0, gate_error=GATE_ERROR),
    ]
    plane, _, _, valid = _run(rows, num_generations_per_prompt=4)
    assert valid
    assert _written(plane, "sample_mask").tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    adv = _scalar(_written(plane, "advantages"))
    torch.testing.assert_close(adv[1:4], _direct(rows, torch.tensor([0.0, 1, 1, 1]))[1:])
    # Rollout 1 now sees only the two zeros: baseline 0, advantage 1.
    assert adv[1].item() == pytest.approx(1.0, abs=1e-5)


def test_chunk_with_no_trainable_row_skips_the_estimator():
    rows = [
        _Row("a", 0, reward=1.0, mask_sample=True),
        _Row("a", 1, reward=0.0, mask_sample=True),
        _Row("a", 0, 1, "compaction_segment", reward=1.0, mask_sample=True),
    ]
    plane, _, _, valid = _run(rows, num_generations_per_prompt=2)
    assert not valid
    assert torch.equal(_written(plane, "advantages"), torch.zeros(3, SEQ))


# ── contract violations ────────────────────────────────────────────────────


def test_orphan_extra_row_raises():
    rows = [
        _Row("a", 0, reward=1.0),
        _Row("a", 1, reward=0.0),
        _Row("a", 2, 1, "compaction_segment", reward=0.0),  # no a_g2 canonical row
    ]
    with pytest.raises(RuntimeError, match="no canonical row"):
        _run(rows, num_generations_per_prompt=2)


def test_duplicate_canonical_row_raises():
    rows = [
        _Row("a", 0, reward=1.0),
        _Row("a", 1, reward=0.0),
        # Published under an extra id but tagged as a second canonical row.
        _Row("a", 0, 0, reward=1.0, sample_id="a_g0_t1"),
    ]
    with pytest.raises(ValueError, match="more than one canonical row"):
        _run(rows, num_generations_per_prompt=2)


def test_non_grpo_estimators_are_rejected_with_extra_rows():
    rows = [
        _Row("a", 0, reward=1.0),
        _Row("a", 1, reward=0.0),
        _Row("a", 0, 1, "compaction_segment", reward=1.0),
    ]
    gdpo = GDPOAdvantageEstimator(AdvEstimatorConfig(name="gdpo"), ClippedPGLossConfig())
    with pytest.raises(NotImplementedError, match="GDPOAdvantageEstimator"):
        _run(rows, num_generations_per_prompt=2, estimator=gdpo)
    # PPO is accepted: it runs GAE per row (see the PPO tests below).
    # Without extra rows the estimator type is not restricted here: the
    # guard must not fire; whether GDPO then succeeds or complains about the
    # reward components this fixture lacks is not what this test pins.
    data, meta = _batch(rows[:2])
    ctrl = _controller(_DataPlane(data), num_generations_per_prompt=2, estimator=gdpo)
    try:
        asyncio.run(ctrl._advantage_stage(meta))
    except NotImplementedError:
        pytest.fail("the GRPO-only guard fired on a chunk without extra rows")
    except Exception:  # noqa: BLE001 - GDPO's own input validation, not under test
        pass


def test_chunk_completeness_failures():
    # Three rollouts for a group configured with N=4.
    rows = [
        _Row("a", 0, reward=1.0),
        _Row("a", 1, reward=0.0),
        _Row("a", 2, reward=0.0),
        _Row("a", 0, 1, "compaction_segment", reward=1.0),
    ]
    with pytest.raises(RuntimeError, match="not made of whole groups"):
        _run(rows, num_generations_per_prompt=4)
    # Right rollout count, but rollout 0 declares a row the chunk lacks.
    rows = [
        _Row("a", 0, reward=1.0, rows_in_rollout=3),
        _Row("a", 1, reward=0.0),
        _Row("a", 2, reward=0.0),
        _Row("a", 0, 1, "compaction_segment", reward=1.0, rows_in_rollout=3),
    ]
    with pytest.raises(RuntimeError, match="does not hold every row"):
        _run(rows, num_generations_per_prompt=3)
    # The check is gated on segment_rows.enabled: a legacy N-mismatch chunk
    # (no extras) still flows through the single-row path when it is off.
    plane, _, _, valid = _run(
        [_Row("a", 0, reward=1.0), _Row("a", 1, reward=0.0), _Row("a", 2, reward=0.0)],
        num_generations_per_prompt=4,
        segment_rows_enabled=False,
    )
    assert valid and _written(plane, "advantages").shape == (3, SEQ)


# ── PPO: per-row GAE, DP pad rows ──────────────────────────────────────────


def _gae_estimator() -> GeneralizedAdvantageEstimator:
    return GeneralizedAdvantageEstimator(
        GAEConfig(gae_lambda=0.95, gae_gamma=1.0), ClippedPGLossConfig()
    )


def _ppo_batch(
    rows: list[_Row], *, pad_sources: tuple[int, ...] = ()
) -> tuple[TensorDict, KVBatchMeta]:
    """The finalizer's rows plus critic values, then DP pad rows appended the
    way ``_pad_rows_to_dp_multiple`` writes them: copies of the source rows
    under ``{sid}_pad{k}`` with the source tags and ``sample_mask`` zeroed."""
    data, meta = _batch(rows)
    n = len(rows)
    data["values"] = torch.rand(n, SEQ, generator=torch.Generator().manual_seed(0))
    if not pad_sources:
        return data, meta
    index = torch.tensor(list(range(n)) + list(pad_sources))
    padded = data[index].clone()
    padded["sample_mask"][n:] = 0.0
    tags = list(meta.tags or [])
    meta = KVBatchMeta(
        partition_id=meta.partition_id,
        task_name=meta.task_name,
        sample_ids=list(meta.sample_ids)
        + [f"{meta.sample_ids[i]}_pad{k}" for k, i in enumerate(pad_sources)],
        fields=list(padded.keys()),
        tags=tags + [dict(tags[i]) for i in pad_sources],
    )
    return padded, meta


def _run_ppo(data: TensorDict, meta: KVBatchMeta, **controller_kwargs):
    plane = _DataPlane(data)
    ctrl = _controller(
        plane, estimator=_gae_estimator(), is_ppo=True, **controller_kwargs
    )
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert plane.written is not None
    return plane, ctrl, valid


def _direct_gae(data: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE on every row, as the echo multi-trace path runs it (all_rows)."""
    final = data["sample_mask"] * (~data["mask_sample"]).float()
    return _gae_estimator().compute_advantage(
        prompt_ids=data["prompt_ids_for_adv"],
        rewards=data["total_reward"],
        mask=data["token_mask"],
        values=data["values"],
        valid_mask=final,
        logprobs_policy=data["prev_logprobs"],
    )


def test_ppo_segment_rows_get_per_row_gae_not_the_rollout_broadcast():
    rows = _three_rollouts_with_extras()
    data, meta = _ppo_batch(rows)
    plane, _, valid = _run_ppo(data, meta, num_generations_per_prompt=3)
    assert valid
    advantages, returns = _direct_gae(data)
    torch.testing.assert_close(_written(plane, "advantages"), advantages)
    torch.testing.assert_close(_written(plane, "returns"), returns)
    # Each segment row has its own values, so it is not a copy of its
    # rollout's canonical row (the GRPO broadcast).
    written = _written(plane, "advantages")
    assert not torch.equal(written[3], written[0])
    assert not torch.equal(written[5], written[1])


def test_ppo_pad_rows_do_not_count_as_rollouts_or_canonical_rows():
    rows = _three_rollouts_with_extras()
    # Two copies of row 0 (ppo.multi_trace_pad_source=row0), tags included.
    data, meta = _ppo_batch(rows, pad_sources=(0, 0))
    # Whole groups and whole rollouts hold on the real rows; the pads (own
    # unparsable ids, copied rows_in_rollout tags) must not trip the check.
    plane, ctrl, valid = _run_ppo(data, meta, num_generations_per_prompt=3)
    assert valid
    assert _written(plane, "advantages").shape == (len(rows) + 2, SEQ)
    # GAE still whitens over every row's tokens, pads included (all_rows).
    advantages, _ = _direct_gae(data)
    torch.testing.assert_close(_written(plane, "advantages"), advantages)
    # PPO files its canonical flags apart: ``reward`` stays the legacy per-row
    # mean and ``reward_per_rollout`` uses these.
    assert "canonical_masks" not in ctrl._step_log_dict or not ctrl._step_log_dict[
        "canonical_masks"
    ]
    assert ctrl._step_log_dict["segment_canonical_masks"][0].tolist() == [
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    assert ctrl._step_log_dict["num_mask_sample_filtered"] == [0]
    segments = reduce_segment_stats(ctrl._segment_stats_acc)
    assert segments["segments/rows"] == len(rows)
    assert segments["segments/rollouts"] == 3


def test_pad_rows_outside_ppo_fail_loudly():
    rows = [_Row("a", 0, reward=1.0), _Row("a", 1, reward=0.0)]
    data, meta = _ppo_batch(rows, pad_sources=(0,))
    plane = _DataPlane(data)
    ctrl = _controller(plane, num_generations_per_prompt=2)
    with pytest.raises(RuntimeError, match="DP pad rows"):
        asyncio.run(ctrl._advantage_stage(meta))


def test_ppo_reward_metric_stays_per_row_with_a_per_rollout_companion():
    from nemo_rl.algorithms.single_controller_utils.utils import (
        reduce_advantage_pump_metrics,
    )

    # Rollout 0 (reward 1) has a subagent row; rollout 1 (reward 0) has none.
    rewards = [torch.tensor([1.0, 0.0, 1.0])]
    sample_masks = [torch.ones(3)]
    out = reduce_advantage_pump_metrics(
        rewards,
        [torch.zeros(3)],
        [4, 4, 4],
        sample_masks=sample_masks,
        segment_canonical_masks=[torch.tensor([1.0, 1.0, 0.0])],
    )
    # Legacy echo semantics: every trained row counts.
    assert out["reward"] == pytest.approx(2 / 3)
    assert out["reward_per_rollout"] == pytest.approx(0.5)

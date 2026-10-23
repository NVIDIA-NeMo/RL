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
"""Per-turn scopes of positive_length_reweight (reweight_positive_text_per_turn).

Rows are built turn by turn: a one-token prompt, then per assistant turn THINKING tokens, CONTENT
tokens, TOOL-CALL tokens (ACTION) and one end-of-turn token (ACTION), with one tool-output token
(not trained) between turns. The base group follows the spec: G = 6, rewards [1, 1, 1, 0, 0, 0],
default GRPO normalization, so every positive row has A+ = 1.09544 and every negative A- = -A+.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
    PositiveLengthReweightConfig,
    PositiveLengthReweightStats,
    per_turn_length_reweight_config,
    reweight_positive_advantages_by_length,
    reweight_positive_text_per_turn,
)
from nemo_rl.algorithms.single_controller_utils.utils import (
    apply_message_level_advantage_penalties,
)

A_POS = 1.0954451
NEG = -A_POS


@dataclass
class Turn:
    thinking: int = 0
    content: int = 0
    tool: int = 0
    content_after: int = 0
    eos: bool = True


@dataclass
class Row:
    turns: list[Turn]
    # Filled by build(): per-position labels.
    labels: list[str] = field(default_factory=list)


def _row_columns(turns: list[Turn]):
    """Per-position (token_mask, turn_id, thinking, action, label) for one row."""
    mask, turn_id, thinking, action, labels = [0.0], [-1], [False], [False], ["prompt"]
    for t, turn in enumerate(turns):
        if t > 0:  # tool output between turns, not trained
            mask.append(0.0)
            turn_id.append(-1)
            thinking.append(False)
            action.append(False)
            labels.append("tool_output")
        for kind, count in (
            ("thinking", turn.thinking),
            ("content", turn.content),
            ("tool", turn.tool),
            ("content", turn.content_after),
            ("eos", 1 if turn.eos else 0),
        ):
            for _ in range(count):
                mask.append(1.0)
                turn_id.append(t)
                thinking.append(kind == "thinking")
                action.append(kind in ("tool", "eos"))
                labels.append(kind if kind != "content" else f"content@{t}")
    return mask, turn_id, thinking, action, labels


def build(
    rows: list[list[Turn]], rewards: list[float], prompt_ids: list[int] | None = None
):
    """Batch tensors plus the estimator's (un-reweighted) token-level advantages."""
    cols = [_row_columns(r) for r in rows]
    width = max(len(c[0]) for c in cols)
    n = len(rows)
    mask = torch.zeros((n, width))
    turn_id = torch.full((n, width), -1, dtype=torch.int32)
    thinking = torch.zeros((n, width), dtype=torch.bool)
    action = torch.zeros((n, width), dtype=torch.bool)
    labels = []
    for i, (m, tid, th, ac, lab) in enumerate(cols):
        k = len(m)
        mask[i, :k] = torch.tensor(m)
        turn_id[i, :k] = torch.tensor(tid, dtype=torch.int32)
        thinking[i, :k] = torch.tensor(th)
        action[i, :k] = torch.tensor(ac)
        labels.append(lab + ["pad"] * (width - k))
    # [B, 1]: the GRPO baseline groups rows of the 2-D prompt tensor.
    prompt = torch.tensor(prompt_ids if prompt_ids is not None else [0] * n).unsqueeze(
        -1
    )
    reward = torch.tensor(rewards, dtype=torch.float32)
    estimator = GRPOAdvantageEstimator(AdvEstimatorConfig(), None)
    adv = estimator.compute_advantage(prompt_ids=prompt, rewards=reward, mask=mask)
    return dict(
        advantages=adv,
        prompt_ids=prompt,
        rewards=reward,
        mask=mask,
        assistant_turn_id=turn_id,
        thinking_mask=thinking,
        action_mask=action,
        valid=torch.ones(n, dtype=torch.bool),
        labels=labels,
    )


def cfg(**kwargs) -> PositiveLengthReweightConfig:
    base = dict(enabled=True, scope="prose_per_turn")
    base.update(kwargs)
    return PositiveLengthReweightConfig(**base)


def run(batch, config):
    args = {k: v for k, v in batch.items() if k != "labels"}
    return reweight_positive_text_per_turn(**args, config=config)


def values_at(out, batch, row, kind) -> torch.Tensor:
    labels = batch["labels"][row]
    idx = [
        j
        for j, lab in enumerate(labels)
        if lab == kind or (kind == "content" and lab.startswith("content"))
    ]
    return out[row, idx]


def assert_actions_untouched(out, batch):
    """Every ACTION token (tool calls, end of turn) keeps the unweighted advantage, bit for bit."""
    act = batch["action_mask"] & batch["mask"].bool()
    assert torch.equal(out[act], batch["advantages"][act])


def assert_multiplier(out, batch, row, kind, mult, atol=1e-5):
    vals = values_at(out, batch, row, kind)
    assert vals.numel() > 0
    torch.testing.assert_close(
        vals, torch.full_like(vals, A_POS * mult), atol=atol, rtol=0
    )


def spec_group(
    c_turns: list[Turn] | None = None, b_turns: list[Turn] | None = None, a_turns=None
):
    """A, B, C correct (turns [4, 8, 4], thinking [800, 1600, 2400], tool tokens [400, 800, 400]); 3 wrong."""
    a = a_turns or [Turn(thinking=200, tool=100) for _ in range(4)]
    b = b_turns or [Turn(thinking=200, tool=100) for _ in range(8)]
    c = c_turns or [Turn(thinking=600, tool=100) for _ in range(4)]
    wrong = [
        [Turn(thinking=50, tool=10), Turn(thinking=70, content=5, tool=20)],
        [Turn(thinking=900, tool=30)],
        [Turn(thinking=10, tool=5) for _ in range(6)],
    ]
    return build([a, b, c, *wrong], [1, 1, 1, 0, 0, 0])


# ── spec tests ───────────────────────────────────────────────────────────


def test_base_group_advantages_match_the_spec():
    batch = spec_group()
    trained = batch["mask"].bool()
    for row in range(3):
        assert torch.allclose(
            batch["advantages"][row][trained[row]], torch.tensor(A_POS), atol=1e-6
        )
    for row in range(3, 6):
        assert torch.allclose(
            batch["advantages"][row][trained[row]], torch.tensor(NEG), atol=1e-6
        )


@pytest.mark.parametrize("scope", ["thinking_per_turn", "prose_per_turn"])
def test_1_multipliers_and_untouched_tokens(scope):
    batch = spec_group()
    out, stats = run(batch, cfg(scope=scope))
    for row, mult in zip(range(3), (1.5, 1.5, 0.5)):
        assert_multiplier(out, batch, row, "thinking", mult)
    torch.testing.assert_close(
        values_at(out, batch, 0, "thinking")[0],
        torch.tensor(1.64316),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        values_at(out, batch, 2, "thinking")[0],
        torch.tensor(0.54772),
        atol=1e-5,
        rtol=0,
    )
    assert_actions_untouched(out, batch)
    for row in range(3, 6):
        assert torch.equal(out[row], batch["advantages"][row])
    assert stats.groups_reweighted == 1 and stats.groups_with_valid_rows == 1
    assert stats.weight_min == pytest.approx(0.5) and stats.weight_max == pytest.approx(
        1.5
    )


def test_2_turn_count_does_not_change_the_weight():
    # B has 8 turns and 1600 thinking tokens: same per-turn verbosity as A, same multiplier.
    batch = spec_group()
    out, _ = run(batch, cfg())
    assert torch.equal(
        values_at(out, batch, 0, "thinking"), values_at(out, batch, 1, "thinking")[:800]
    )
    # B with 4 turns and 1600 thinking tokens is twice as verbose per turn.
    batch = spec_group(b_turns=[Turn(thinking=400, tool=200) for _ in range(4)])
    out, _ = run(batch, cfg())
    for row, mult in zip(range(3), (2.0, 1.0, 2.0 / 3.0)):
        assert_multiplier(out, batch, row, "thinking", mult)
    assert_actions_untouched(out, batch)


@pytest.mark.parametrize(
    "counted,mults,first",
    [
        ("with_spans", (1.5, 1.5, 0.5), 1.64316),
        ("all", (5 / 3, 4 / 3, 5 / 9), 1.82574),
    ],
)
def test_3_turns_counted(counted, mults, first):
    # A gets a fifth trained turn with no measured tokens: a 50-token tool call.
    a_turns = [Turn(thinking=200, tool=100) for _ in range(4)] + [Turn(tool=50)]
    batch = spec_group(a_turns=a_turns)
    out, _ = run(batch, cfg(turns_counted=counted))
    for row, mult in zip(range(3), mults):
        assert_multiplier(out, batch, row, "thinking", mult)
    torch.testing.assert_close(
        values_at(out, batch, 0, "thinking")[0], torch.tensor(first), atol=1e-5, rtol=0
    )
    if counted == "all":
        torch.testing.assert_close(
            values_at(out, batch, 1, "thinking")[0],
            torch.tensor(1.46059),
            atol=1e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            values_at(out, batch, 2, "thinking")[0],
            torch.tensor(0.60858),
            atol=1e-5,
            rtol=0,
        )
    assert_actions_untouched(out, batch)


def test_3b_turns_counted_none_measures_the_total():
    # V = P: A 800, B 1600, C 2400 thinking tokens; u = (1, 1/2, 1/3), c = 4800 / 2400 = 2.
    batch = spec_group()
    out, stats = run(batch, cfg(turns_counted="none"))
    for row, mult in zip(range(3), (2.0, 1.0, 2.0 / 3.0)):
        assert_multiplier(out, batch, row, "thinking", mult)
    assert_actions_untouched(out, batch)
    for row in range(3, 6):
        assert torch.equal(out[row], batch["advantages"][row])
    assert stats.weight_min == pytest.approx(2.0 / 3.0)
    assert stats.weight_max == pytest.approx(2.0)


@pytest.mark.parametrize("counted", ["with_spans", "none"])
def test_3c_one_short_turn_vs_ten_turns(counted):
    # 100 thinking tokens in 1 turn vs 1000 in 10 turns (100 per turn), both correct: per turn
    # they are equally verbose; in total the second wrote 10x more and, under "none", weighs less.
    short = [Turn(thinking=100, tool=10)]
    long = [Turn(thinking=100, tool=10) for _ in range(10)]
    wrong = [[Turn(thinking=300, tool=10)], [Turn(thinking=50, tool=10)]]
    batch = build([short, long, *wrong], [1, 1, 0, 0])
    out, _ = run(batch, cfg(turns_counted=counted))
    w = [
        (
            values_at(out, batch, r, "thinking")
            / values_at(batch["advantages"], batch, r, "thinking")
        )
        for r in (0, 1)
    ]
    for r in (0, 1):
        assert torch.allclose(w[r], w[r][0].expand_as(w[r]), atol=1e-6)
    if counted == "with_spans":
        assert float(w[0][0]) == pytest.approx(1.0) and float(w[1][0]) == pytest.approx(
            1.0
        )
    else:
        # u = (1, max(100 / 1000, 1/3)) = (1, 1/3); c = 1100 / (100 + 1000 / 3).
        c = 1100.0 / (100.0 + 1000.0 / 3.0)
        assert float(w[0][0]) == pytest.approx(c, rel=1e-5)
        assert float(w[1][0]) == pytest.approx(c / 3.0, rel=1e-5)
        # The group's positive credit on thinking tokens is unchanged.
        before = sum(
            float(values_at(batch["advantages"], batch, r, "thinking").sum())
            for r in (0, 1)
        )
        after = sum(float(values_at(out, batch, r, "thinking").sum()) for r in (0, 1))
        assert after == pytest.approx(before, rel=1e-5)
    assert_actions_untouched(out, batch)


def test_4_thinking_to_content_leak():
    # C moves 1600 of its 2400 thinking tokens into visible content (400 per turn).
    c_turns = [Turn(thinking=200, content=400, tool=100) for _ in range(4)]
    batch = spec_group(c_turns=c_turns)
    out, _ = run(batch, cfg(scope="prose_per_turn"))
    for row, mult in zip(range(3), (1.5, 1.5, 0.5)):
        assert_multiplier(out, batch, row, "thinking", mult)
    assert_multiplier(out, batch, 2, "content", 0.5)
    # thinking_per_turn: C escapes, every correct row keeps weight 1.
    out, _ = run(batch, cfg(scope="thinking_per_turn"))
    for row in range(3):
        assert_multiplier(out, batch, row, "thinking", 1.0)
    assert torch.equal(
        values_at(out, batch, 2, "content"),
        values_at(batch["advantages"], batch, 2, "content"),
    )


@pytest.mark.parametrize(
    "exclude,mults", [(False, (17 / 12, 1.5, 0.51)), (True, (1.5, 1.5, 0.5))]
)
def test_5_exclude_final_answer(exclude, mults):
    def with_answer(turns):
        return turns[:-1] + [
            Turn(thinking=turns[-1].thinking, tool=turns[-1].tool, content_after=100)
        ]

    batch = spec_group(
        a_turns=with_answer([Turn(thinking=200, tool=100) for _ in range(4)]),
        b_turns=with_answer([Turn(thinking=200, tool=100) for _ in range(8)]),
        c_turns=with_answer([Turn(thinking=600, tool=100) for _ in range(4)]),
    )
    out, _ = run(batch, cfg(exclude_final_answer=exclude))
    for row, mult in zip(range(3), mults):
        assert_multiplier(out, batch, row, "thinking", mult)
    for row in range(3):
        last_turn = int(batch["assistant_turn_id"][row].max())
        vals = values_at(out, batch, row, f"content@{last_turn}")
        assert vals.numel() == 100
        if exclude:
            assert torch.equal(
                vals, torch.full_like(vals, batch["advantages"][row, 1].item())
            )
        else:
            torch.testing.assert_close(
                vals, torch.full_like(vals, A_POS * mults[row]), atol=1e-5, rtol=0
            )
    if not exclude:
        for row, expected in zip(range(3), (1.55187, 1.64316, 0.55867)):
            torch.testing.assert_close(
                values_at(out, batch, row, "thinking")[0],
                torch.tensor(expected),
                atol=1e-5,
                rtol=0,
            )
    assert_actions_untouched(out, batch)


@pytest.mark.parametrize("scope", ["thinking_per_turn", "prose_per_turn"])
@pytest.mark.parametrize("mode", ["inverse_length", "rank"])
def test_7_mass_is_conserved(scope, mode):
    c_turns = [
        Turn(thinking=300, content=200, tool=50, content_after=30) for _ in range(3)
    ]
    batch = spec_group(c_turns=c_turns)
    out, stats = run(batch, cfg(scope=scope, mode=mode))
    trained = batch["mask"].bool()[:, 1:]
    measured = (
        batch["thinking_mask"][:, 1:]
        if scope == "thinking_per_turn"
        else ((batch["assistant_turn_id"][:, 1:] >= 0) & ~batch["action_mask"][:, 1:])
    )
    measured = measured & trained
    before, after = batch["advantages"][:, 1:].double(), out[:, 1:].double()
    pos = (before > 0) & measured
    torch.testing.assert_close(after[pos].sum(), before[pos].sum(), rtol=1e-6, atol=0)
    torch.testing.assert_close(
        (after.abs() * trained).sum(), (before.abs() * trained).sum(), rtol=1e-6, atol=0
    )
    assert sum(stats.mass_after) == pytest.approx(sum(stats.mass_before), rel=1e-6)
    m = stats.to_metrics()
    assert m["adv_len_reweight/share_max_abs_diff"] < 1e-6


def test_8_positives_without_measured_tokens_and_small_groups():
    # A third positive has only tool calls: P = 0, excluded; A and C are reweighted alone.
    batch = spec_group(b_turns=[Turn(tool=100) for _ in range(3)])
    out, stats = run(batch, cfg())
    assert torch.equal(out[1], batch["advantages"][1])
    # A: V = 200, C: V = 600 -> raw [1, 1/3], c = 3200 / (800 + 800) = 2.
    assert_multiplier(out, batch, 0, "thinking", 2.0)
    assert_multiplier(out, batch, 2, "thinking", 2.0 / 3.0)
    # A group with a single positive is left unchanged.
    batch = build(
        [
            [Turn(thinking=100, tool=10)],
            [Turn(thinking=10, tool=10)],
            [Turn(thinking=10, tool=10)],
        ],
        [1, 0, 0],
    )
    out, stats = run(batch, cfg())
    assert torch.equal(out, batch["advantages"])
    assert stats.groups_reweighted == 0 and stats.groups_with_valid_rows == 1


def test_9_penalties_keep_precedence():
    batch = spec_group()
    out, _ = run(batch, cfg())
    flagged = torch.zeros_like(batch["action_mask"])
    flagged[0, 1:50] = True  # part of A's first turn: an invalid tool call step
    malformed = torch.zeros_like(flagged)
    malformed[2, 1:10] = True
    final = apply_message_level_advantage_penalties(
        out,
        invalid_tool_call_mask=flagged & batch["mask"].bool(),
        malformed_thinking_mask=malformed & batch["mask"].bool(),
        invalid_tool_call_advantage=-3.0,
        malformed_thinking_advantage=-2.0,
    )
    assert torch.all(final[0, 1:50] == -3.0)
    assert torch.all(final[2, 1:10] == -2.0)
    untouched = ~(flagged | malformed)
    assert torch.equal(final[untouched], out[untouched])


@pytest.mark.parametrize("mode", ["inverse_length", "rank"])
def test_10_trajectory_scope_is_unchanged(mode):
    """scope=trajectory runs the original estimator path; a per-turn scope keeps it out."""
    batch = spec_group()
    traj = AdvEstimatorConfig(positive_length_reweight=dict(enabled=True, mode=mode))
    est = GRPOAdvantageEstimator(traj, None)
    got = est.compute_advantage(
        prompt_ids=batch["prompt_ids"], rewards=batch["rewards"], mask=batch["mask"]
    )
    plain = GRPOAdvantageEstimator(AdvEstimatorConfig(), None).compute_advantage(
        prompt_ids=batch["prompt_ids"], rewards=batch["rewards"], mask=batch["mask"]
    )
    expected, _ = reweight_positive_advantages_by_length(
        plain[:, :1].clone(),
        prompt_ids=batch["prompt_ids"],
        rewards=batch["rewards"],
        lengths=batch["mask"][:, 1:].sum(-1),
        valid=torch.ones(6, dtype=torch.bool),
        config=traj.positive_length_reweight,
    )
    assert torch.equal(got, expected.expand(batch["mask"].shape))
    assert "adv_len_reweight/turns_mean_correct" not in est.last_metrics
    per_turn = AdvEstimatorConfig(
        positive_length_reweight=dict(enabled=True, mode=mode, scope="prose_per_turn")
    )
    est = GRPOAdvantageEstimator(per_turn, None)
    got = est.compute_advantage(
        prompt_ids=batch["prompt_ids"], rewards=batch["rewards"], mask=batch["mask"]
    )
    assert torch.equal(got, plain)
    assert est.last_length_reweight_stats is None and est.last_metrics == {}
    with pytest.raises(ValueError, match="per-turn scope"):
        reweight_positive_text_per_turn(
            **{k: v for k, v in batch.items() if k != "labels"},
            config=traj.positive_length_reweight,
        )


def _two_groups():
    g1 = spec_group()
    g2 = build(
        [
            [Turn(thinking=30, tool=5) for _ in range(2)],
            [Turn(thinking=90, content=10, tool=5)],
            [Turn(thinking=5, tool=5)],
            [Turn(thinking=40, tool=1) for _ in range(3)],
        ],
        [1, 1, 0, 1],
        prompt_ids=[7, 7, 7, 7],
    )
    return g1, g2


def _concat(a, b):
    width = max(a["mask"].shape[1], b["mask"].shape[1])

    def pad(t, value):
        if t.dtype == torch.bool:
            return torch.nn.functional.pad(
                t.to(torch.uint8), (0, width - t.shape[1]), value=value
            ).bool()
        return torch.nn.functional.pad(t, (0, width - t.shape[1]), value=value)

    out = {}
    for key in (
        "advantages",
        "mask",
        "assistant_turn_id",
        "thinking_mask",
        "action_mask",
    ):
        value = -1 if key == "assistant_turn_id" else 0
        out[key] = torch.cat([pad(a[key], value), pad(b[key], value)])
    for key in ("prompt_ids", "rewards", "valid"):
        out[key] = torch.cat([a[key], b[key]])
    out["labels"] = a["labels"] + b["labels"]
    return out


@pytest.mark.parametrize("mode", ["inverse_length", "rank"])
def test_11_shard_and_permutation_invariance(mode):
    g1, g2 = _two_groups()
    both = _concat(g1, g2)
    out_both, stats_both = run(both, cfg(mode=mode))
    out1, stats1 = run(g1, cfg(mode=mode))
    out2, stats2 = run(g2, cfg(mode=mode))
    w1 = g1["mask"].shape[1]
    w2 = g2["mask"].shape[1]
    assert torch.equal(out_both[:6, :w1], out1)
    assert torch.equal(out_both[6:, :w2], out2)
    merged = PositiveLengthReweightStats.empty()
    merged.merge(stats1)
    merged.merge(stats2)
    assert merged.to_metrics() == pytest.approx(stats_both.to_metrics())
    perm = torch.tensor([5, 9, 0, 3, 7, 1, 8, 2, 6, 4])
    permuted = {
        k: (v[perm] if isinstance(v, torch.Tensor) else [v[i] for i in perm.tolist()])
        for k, v in both.items()
    }
    out_perm, _ = run(permuted, cfg(mode=mode))
    # Same values up to float64 summation order inside a group.
    torch.testing.assert_close(out_perm, out_both[perm], atol=1e-6, rtol=0)


# ── robustness ───────────────────────────────────────────────────────────


def test_invalid_rows_are_neither_reweighted_nor_counted():
    batch = spec_group()
    batch["valid"] = torch.tensor([True, False, True, True, True, True])
    out, _ = run(batch, cfg())
    assert torch.equal(out[1], batch["advantages"][1])
    # A and C only: V = [200, 600] -> [2, 2/3].
    assert_multiplier(out, batch, 0, "thinking", 2.0)
    assert_multiplier(out, batch, 2, "thinking", 2.0 / 3.0)


def test_untrained_tokens_are_never_measured_or_scaled():
    batch = spec_group()
    # Mask out half of C's thinking (e.g. a masked step); those tokens must stay untouched.
    batch["mask"][2, 1:1001] = 0.0
    out, _ = run(batch, cfg())
    assert torch.equal(out[2, 1:1001], batch["advantages"][2, 1:1001])


def test_max_ratio_none_and_alpha():
    batch = spec_group()
    out, _ = run(batch, cfg(max_ratio=None))
    # V = [200, 200, 600], raw [1, 1, 1/3] (no floor needed), same as with the cap of 3.
    for row, mult in zip(range(3), (1.5, 1.5, 0.5)):
        assert_multiplier(out, batch, row, "thinking", mult)
    out, _ = run(batch, cfg(alpha=0.0))
    for row in range(3):
        assert_multiplier(out, batch, row, "thinking", 1.0)


def test_rank_mode_uses_per_turn_verbosity():
    batch = spec_group()
    out, _ = run(batch, cfg(mode="rank", rank_spread=0.5))
    # V = [200, 200, 600]: average ranks [0.5, 0.5, 2] -> raw [1.25, 1.25, 0.5];
    # c = 4800 / (1000 + 2000 + 1200) = 8/7.
    for row, mult in zip(range(3), (1.25 * 8 / 7, 1.25 * 8 / 7, 0.5 * 8 / 7)):
        assert_multiplier(out, batch, row, "thinking", mult)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dtype_and_device_are_kept(dtype):
    batch = spec_group()
    batch["advantages"] = batch["advantages"].to(dtype)
    out, _ = run(batch, cfg())
    assert out.dtype == dtype and out.device == batch["advantages"].device
    assert out.data_ptr() != batch["advantages"].data_ptr()


def test_non_uniform_row_advantages_raise():
    batch = spec_group()
    batch["advantages"] = batch["advantages"].clone()
    batch["advantages"][0, 5] = 9.0
    with pytest.raises(ValueError, match="one advantage per row"):
        run(batch, cfg())


def test_shape_and_overlap_errors():
    batch = spec_group()
    bad = dict(batch)
    bad["thinking_mask"] = batch["thinking_mask"][:, :-1]
    with pytest.raises(ValueError, match="thinking_mask shape"):
        run(bad, cfg())
    bad = dict(batch)
    bad["action_mask"] = batch["action_mask"] | batch["thinking_mask"]
    with pytest.raises(ValueError, match="overlap"):
        run(bad, cfg())


def test_empty_batch():
    empty = dict(
        advantages=torch.zeros((0, 4)),
        prompt_ids=torch.zeros((0, 1), dtype=torch.long),
        rewards=torch.zeros(0),
        mask=torch.zeros((0, 4)),
        assistant_turn_id=torch.zeros((0, 4), dtype=torch.int32),
        thinking_mask=torch.zeros((0, 4), dtype=torch.bool),
        action_mask=torch.zeros((0, 4), dtype=torch.bool),
        valid=torch.zeros(0, dtype=torch.bool),
        labels=[],
    )
    out, stats = run(empty, cfg())
    assert out.shape == (0, 4) and stats.groups_with_valid_rows == 0


def test_metrics():
    batch = spec_group()
    _, stats = run(batch, cfg())
    m = stats.to_metrics()
    assert m["adv_len_reweight/frac_groups_reweighted"] == 1.0
    assert m["adv_len_reweight/turns_mean_correct"] == pytest.approx((4 + 8 + 4) / 3)
    assert m["adv_len_reweight/turns_mean_incorrect"] == pytest.approx((2 + 1 + 6) / 3)
    assert m["adv_len_reweight/thinking_per_turn_mean_correct"] == pytest.approx(
        4800 / 16
    )
    assert m["adv_len_reweight/content_per_turn_mean_correct"] == 0.0
    # Correct rows' action tokens per turn: (100 tool + 1 end of turn) each.
    assert m["adv_len_reweight/action_tokens_per_turn_mean_correct"] == pytest.approx(
        101.0
    )
    assert m["adv_len_reweight/thinking_share_of_prose_mean_correct"] == pytest.approx(
        1.0
    )
    # net_turn_credit = (A+ * 16 - A+ * 9) / (A+ * 25) in the 0.25-0.75 bucket.
    assert m["adv_len_reweight/net_turn_credit"] == pytest.approx(7 / 25, rel=1e-5)
    assert m["adv_len_reweight/net_turn_credit_bucket_0.25_0.75"] == pytest.approx(
        7 / 25, rel=1e-5
    )
    assert m["adv_len_reweight/weight_mean"] == pytest.approx(7 / 6)


# ── config ───────────────────────────────────────────────────────────────


def test_config_defaults_and_validation():
    c = PositiveLengthReweightConfig()
    assert (c.scope, c.exclude_final_answer, c.turns_counted) == (
        "trajectory",
        False,
        "with_spans",
    )
    assert (
        not c.per_turn
        and not PositiveLengthReweightConfig(scope="prose_per_turn").per_turn
    )
    assert PositiveLengthReweightConfig(
        enabled=True, scope="thinking_per_turn"
    ).per_turn
    with pytest.raises(ValueError, match="exclude_final_answer"):
        PositiveLengthReweightConfig(
            enabled=True, scope="thinking_per_turn", exclude_final_answer=True
        )
    with pytest.raises(ValueError, match="exclude_final_answer"):
        PositiveLengthReweightConfig(enabled=True, exclude_final_answer=True)
    with pytest.raises(ValueError):
        PositiveLengthReweightConfig(enabled=True, scope="per_turn")
    assert (
        PositiveLengthReweightConfig(enabled=True, turns_counted="none").turns_counted
        == "none"
    )
    with pytest.raises(ValueError):
        PositiveLengthReweightConfig(enabled=True, turns_counted="some")
    with pytest.raises(ValueError):
        PositiveLengthReweightConfig(enabled=True, scopee="prose_per_turn")


def test_per_turn_config_accessor():
    from types import SimpleNamespace

    on = AdvEstimatorConfig(
        positive_length_reweight=dict(enabled=True, scope="prose_per_turn")
    )
    assert (
        per_turn_length_reweight_config(SimpleNamespace(adv_estimator=on))
        is on.positive_length_reweight
    )
    off = AdvEstimatorConfig(
        positive_length_reweight=dict(enabled=False, scope="prose_per_turn")
    )
    assert per_turn_length_reweight_config(SimpleNamespace(adv_estimator=off)) is None
    traj = AdvEstimatorConfig(positive_length_reweight=dict(enabled=True))
    assert per_turn_length_reweight_config(SimpleNamespace(adv_estimator=traj)) is None
    # A PPO block has a GAE estimator without positive_length_reweight.
    assert (
        per_turn_length_reweight_config(
            SimpleNamespace(adv_estimator=SimpleNamespace(name="gae"))
        )
        is None
    )
    assert per_turn_length_reweight_config(object()) is None


def test_legacy_trainers_reject_per_turn_scopes():
    from types import SimpleNamespace

    from nemo_rl.algorithms.grpo import _raise_if_per_turn_length_reweight

    per_turn = AdvEstimatorConfig(
        positive_length_reweight=dict(enabled=True, scope="thinking_per_turn")
    )
    with pytest.raises(NotImplementedError, match="SingleController token-capture"):
        _raise_if_per_turn_length_reweight(
            SimpleNamespace(grpo=SimpleNamespace(adv_estimator=per_turn))
        )
    traj = AdvEstimatorConfig(positive_length_reweight=dict(enabled=True))
    _raise_if_per_turn_length_reweight(
        SimpleNamespace(grpo=SimpleNamespace(adv_estimator=traj))
    )


# ── rollouts spanning several rows (SingleController token-capture segment rows) ──────────


def _segment_group(*, final_content: int = 0):
    """The spec group with B = an early segment row (3 long turns) + its canonical row (5 short
    turns); returns (merged batch, split batch, row_to_rollout, final_answer_rows).

    In the split batch, rows 0-5 are the canonical rows A, B, C and the three wrong rollouts, and
    row 6 is B's segment row; its turn ids restart at 0 like a real subagent / compaction row.
    Every row carries its rollout's merged advantage, as after the SingleController broadcast.
    """
    a = [Turn(thinking=200, tool=100) for _ in range(4)]
    b_segment = [Turn(thinking=600, tool=100) for _ in range(2)] + [
        Turn(thinking=600, content=final_content, tool=100)
    ]
    b_canonical = [Turn(thinking=100, tool=100) for _ in range(4)] + [
        Turn(thinking=100, content=final_content, tool=0)
    ]
    c = [Turn(thinking=600, tool=100) for _ in range(4)]
    wrong = [
        [Turn(thinking=50, tool=10), Turn(thinking=70, content=5, tool=20)],
        [Turn(thinking=900, tool=30)],
        [Turn(thinking=10, tool=5) for _ in range(6)],
    ]
    merged = build([a, b_segment + b_canonical, c, *wrong], [1, 1, 1, 0, 0, 0])
    split = build([a, b_canonical, c, *wrong, b_segment], [1, 1, 1, 0, 0, 0, 1])
    row_to_rollout = torch.tensor([0, 1, 2, 3, 4, 5, 1])
    split["advantages"] = (
        merged["advantages"][row_to_rollout, :1].expand(split["mask"].shape).clone()
    )
    final_answer_rows = torch.tensor([True] * 6 + [False])
    return merged, split, row_to_rollout, final_answer_rows


def _run_rows(batch, config, **extra):
    args = {k: v for k, v in batch.items() if k != "labels"}
    return reweight_positive_text_per_turn(**args, config=config, **extra)


def test_segment_rows_are_weighted_as_one_rollout():
    merged, split, row_to_rollout, final_rows = _segment_group()
    out_m, stats_m = run(merged, cfg())
    out_s, stats_s = _run_rows(
        split, cfg(), row_to_rollout=row_to_rollout, final_answer_rows=final_rows
    )
    # B: (3 x 600 + 5 x 100) / 8 = 287.5 thinking tokens per turn, measured over both rows.
    b_merged = values_at(out_m, merged, 1, "thinking")
    b_split = torch.cat(
        [values_at(out_s, split, 6, "thinking"), values_at(out_s, split, 1, "thinking")]
    )
    assert torch.equal(b_split, b_merged)
    assert not torch.allclose(b_merged, torch.full_like(b_merged, A_POS))
    for row in (0, 2, 3, 4, 5):  # the two batches pad to different widths
        n = sum(lab != "pad" for lab in split["labels"][row])
        assert torch.equal(out_s[row, :n], out_m[row, :n])
    assert_actions_untouched(out_s, split)
    assert stats_s.to_metrics() == pytest.approx(stats_m.to_metrics())


def test_segment_rows_without_the_rollout_map_are_weighted_apart():
    _, split, _, _ = _segment_group()
    out, _ = _run_rows(split, cfg())
    # Read as two responses, B's long-turn segment (600 / turn) gets less per token than its
    # short-turn canonical row (100 / turn): the same rollout pulled two ways.
    seg = values_at(out, split, 6, "thinking")[0]
    can = values_at(out, split, 1, "thinking")[0]
    assert seg < can


def test_exclude_final_answer_only_in_the_canonical_row():
    _, split, row_to_rollout, final_rows = _segment_group(final_content=50)
    out, _ = _run_rows(
        split,
        cfg(exclude_final_answer=True),
        row_to_rollout=row_to_rollout,
        final_answer_rows=final_rows,
    )
    # The canonical row's last-turn content is the answer: unscaled.
    canonical_answer = values_at(out, split, 1, "content@4")
    assert canonical_answer.numel() == 50
    assert torch.equal(
        canonical_answer, values_at(split["advantages"], split, 1, "content@4")
    )
    # The segment row's last-turn content (a subagent's reply) is prose: scaled with B.
    segment_reply = values_at(out, split, 6, "content@2")
    segment_thinking = values_at(out, split, 6, "thinking")
    assert torch.equal(segment_reply, segment_thinking[: segment_reply.numel()])
    assert not torch.allclose(segment_reply, torch.full_like(segment_reply, A_POS))


def test_rollout_rows_must_share_the_advantage_and_map_every_rollout():
    _, split, row_to_rollout, final_rows = _segment_group()
    bad = dict(split)
    bad["advantages"] = split["advantages"].clone()
    bad["advantages"][6] = 0.5  # B's segment row disagrees with its canonical row
    with pytest.raises(ValueError, match="per rollout"):
        _run_rows(bad, cfg(), row_to_rollout=row_to_rollout)
    with pytest.raises(ValueError, match="row_to_rollout"):
        _run_rows(split, cfg(), row_to_rollout=torch.tensor([0, 1, 2, 3, 4, 5, -1]))
    with pytest.raises(ValueError, match="every rollout index"):
        _run_rows(split, cfg(), row_to_rollout=torch.tensor([0, 1, 2, 3, 4, 5, 7]))

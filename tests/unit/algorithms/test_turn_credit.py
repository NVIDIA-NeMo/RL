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
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.algorithms.single_controller_utils.turn_credit import (
    GOOD,
    HARMFUL,
    RolloutTurnLabels,
    TurnCreditConfig,
    TurnCreditStats,
    call_of,
    call_starts_of,
    shape_turn_credit,
    shuffled_labels,
    token_runs,
    tokens_match,
    turn_credit_config,
)

JUDGE = {"endpoint_file": "/tmp/turn_judge.env"}


def cfg(**kw):
    # The tests below exercise the shaping; the run-by-run token identity check has its own test.
    return TurnCreditConfig(
        enabled=True, judge=JUDGE, **{"require_token_match": False, **kw}
    )


def row_layout(turn_lengths, gap=2, prompt=3, pad=0):
    """token_mask of one row: a prompt, then generated runs separated by `gap` environment tokens, right padding."""
    mask = [0] * prompt
    runs = []
    for i, n in enumerate(turn_lengths):
        runs.append((len(mask), len(mask) + n))
        mask += [1] * n
        if i + 1 < len(turn_lengths):
            mask += [0] * gap
    mask += [0] * pad
    return torch.tensor(mask, dtype=torch.float32), runs


def batch(rows, advs):
    width = max(len(m) for m, _ in rows)
    token_mask = torch.zeros(len(rows), width)
    for i, (m, _) in enumerate(rows):
        token_mask[i, : len(m)] = m
    advantages = (
        torch.tensor(advs, dtype=torch.float32).unsqueeze(-1).expand(len(rows), width)
    )
    return advantages, token_mask


def labels(n_calls, pairs, gid="g", gen=0, aligned=None, trailing=0, step_tokens=None):
    return RolloutTurnLabels(
        group_id=gid,
        generation_index=gen,
        n_calls=n_calls,
        labels=tuple(pairs),
        aligned=aligned,
        trailing_empty_calls=trailing,
        step_tokens=None if step_tokens is None else tuple(step_tokens),
    )


def test_token_runs_and_call_mapping():
    m, runs = row_layout([4, 1, 3], pad=5)
    assert token_runs(m) == runs
    assert token_runs(torch.zeros(6)) == []
    assert token_runs(torch.ones(4)) == [(0, 4)]
    out = [
        {"type": "reasoning"},
        {"type": "function_call"},
        {"type": "function_call_output"},
        {"type": "reasoning"},
        {"type": "message"},
    ]
    starts = call_starts_of(out)
    assert starts == [0, 3]
    assert [call_of(starts, i) for i in range(5)] == [0, 0, 0, 1, 1]
    # A rollout whose first item is not reasoning still starts a call at item 0.
    assert call_starts_of([{"type": "message"}, {"type": "reasoning"}]) == [0, 1]


@pytest.mark.parametrize(
    "a,turn_labels,lengths,expected",
    [
        # PHASE3_DESIGN examples (lambda 0.5, 6 equal turns).
        (-0.5, [(2, HARMFUL)], [10] * 6, {2: -0.70833, "other": -0.45833}),
        (0.5, [(2, GOOD)], [10] * 6, {2: 0.70833, "other": 0.45833}),
        (
            -0.5,
            [(1, GOOD), (4, HARMFUL)],
            [10] * 6,
            {1: -0.25, 4: -0.75, "other": -0.5},
        ),
        # A harmful turn holding half of the tokens.
        (-0.5, [(0, HARMFUL)], [50, 10, 10, 10, 10, 10], {0: -0.625, "other": -0.375}),
    ],
)
def test_worked_examples(a, turn_labels, lengths, expected):
    m, runs = row_layout(lengths)
    adv, tm = batch([(m, runs)], [a])
    out, stats, records = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels={0: labels(len(lengths), turn_labels)},
        config=cfg(),
    )
    labeled = dict(turn_labels)
    for k, (s, e) in enumerate(runs):
        want = expected.get(k, expected["other"]) if k in labeled else expected["other"]
        torch.testing.assert_close(
            out[0, s:e], torch.full((e - s,), want), atol=1e-4, rtol=0
        )
    assert stats.shaped_rows == 1 and stats.sign_flips == 0
    assert stats.zero_sum_max_err < 1e-6
    assert records[0]["status"] == "shaped" and records[0][
        "lambda_eff"
    ] == pytest.approx(0.5)


def _invariants(out, adv, tm, rows):
    for r in rows:
        w = tm[r] > 0
        a = float(adv[r][w][0])
        moved = out[r][w].double() - a
        assert abs(float(moved.sum())) <= 1e-5 * abs(a) * int(w.sum())
        assert bool((torch.sign(out[r][w]) == (1.0 if a > 0 else -1.0)).all())


def test_zero_sum_and_no_sign_flip_random():
    gen = torch.Generator().manual_seed(0)
    for trial in range(200):
        n_calls = int(torch.randint(1, 30, (1,), generator=gen))
        lengths = torch.randint(1, 400, (n_calls,), generator=gen).tolist()
        m, runs = row_layout(lengths)
        a = float(torch.randn(1, generator=gen)) * 2
        k = int(torch.randint(1, n_calls + 1, (1,), generator=gen))
        calls = torch.randperm(n_calls, generator=gen)[:k].tolist()
        lab = [
            (c, GOOD if float(torch.rand(1, generator=gen)) < 0.5 else HARMFUL)
            for c in calls
        ]
        adv, tm = batch([(m, runs)], [a])
        kappa = 0.8
        out, stats, _ = shape_turn_credit(
            adv,
            train_mask=tm,
            token_mask=tm,
            row_labels={0: labels(n_calls, lab)},
            config=cfg(lam=1.5, kappa=kappa),
        )
        if stats.shaped_rows:
            _invariants(out, adv, tm, [0])
            assert stats.max_shift_over_abs_adv <= kappa + 1e-6
        else:
            assert stats.noops.get("uniform_labels") == 1


def test_lambda_cap_when_labels_cover_most_tokens():
    # One good token vs a 99-token harmful turn: jbar = -0.98, max |j - jbar| = 1.98 -> lambda_eff = 0.8 / 1.98.
    m, runs = row_layout([1, 99])
    adv, tm = batch([(m, runs)], [-0.5])
    out, stats, records = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels={0: labels(2, [(0, GOOD), (1, HARMFUL)])},
        config=cfg(),
    )
    assert records[0]["lambda_eff"] == pytest.approx(0.8 / 1.98)
    assert stats.lambda_capped_rows == 1
    assert float(out[0, runs[0][0]]) == pytest.approx(-0.5 + 0.8 * 0.5, abs=1e-6)
    _invariants(out, adv, tm, [0])


def test_noops_and_drops_leave_rows_untouched():
    lengths = [5, 5, 5]
    m, runs = row_layout(lengths)
    rows = [(m, runs)] * 7
    adv, tm = batch(rows, [0.5, 0.0, 0.5, 0.5, 0.5, -0.5, 0.5])
    train = tm.clone()
    train[6] = 0  # masked row (ctx-limited, seq-logprob error, ...)
    row_labels = {
        0: labels(3, []),  # no labels
        1: labels(3, [(0, GOOD)]),  # A = 0
        2: labels(4, [(0, GOOD)]),  # call count differs from the row's runs
        3: labels(
            3, [(0, GOOD), (1, GOOD), (2, GOOD)]
        ),  # every call the same label: j - jbar = 0
        4: labels(3, [(5, GOOD)]),  # out of range
        5: labels(3, [(1, HARMFUL), (1, GOOD)]),  # two labels on one call
        6: labels(3, [(0, GOOD)]),
    }
    out, stats, records = shape_turn_credit(
        adv, train_mask=train, token_mask=tm, row_labels=row_labels, config=cfg()
    )
    assert out is adv  # nothing shaped: the input object comes back
    assert stats.shaped_rows == 0
    assert stats.drops == {
        "n_runs_mismatch": 1,
        "bad_label": 1,
        "duplicate_call_label": 1,
    }
    assert stats.noops == {
        "no_labels": 1,
        "zero_advantage": 1,
        "uniform_labels": 1,
        "untrained_row": 1,
    }
    assert len(records) == 6


def test_input_not_mutated_and_unlabeled_rows_bitwise_equal():
    m, runs = row_layout([3, 4, 5])
    adv, tm = batch([(m, runs)] * 3, [0.7, -0.3, 1.1])
    assert adv.stride(-1) == 0  # the estimator's expand
    before = adv.clone()
    out, stats, _ = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels={1: labels(3, [(2, HARMFUL)])},
        config=cfg(),
    )
    assert torch.equal(adv, before)
    assert out is not adv and out.dtype == adv.dtype and out.is_contiguous()
    assert torch.equal(out[0], adv[0]) and torch.equal(out[2], adv[2])
    assert not torch.equal(out[1], adv[1])
    # Untrained positions (prompt, environment tokens, padding) keep A.
    assert torch.equal(out[1][tm[1] == 0], adv[1][tm[1] == 0])


def test_shadow_mode_returns_input_with_stats():
    m, runs = row_layout([3, 4, 5])
    adv, tm = batch([(m, runs)], [0.7])
    out, stats, records = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels={0: labels(3, [(0, GOOD)])},
        config=cfg(apply=False),
    )
    assert out is adv
    assert stats.shaped_rows == 1 and not stats.applied
    assert stats.to_metrics()["turn_credit/applied"] == 0.0
    assert records[0]["status"] == "shaped"


def test_shuffle_is_deterministic_and_keeps_the_multiset():
    lab = ((1, GOOD), (4, HARMFUL), (7, HARMFUL))
    a = shuffled_labels(lab, 20, seed=3, group_id="grp", generation_index=5)
    b = shuffled_labels(lab, 20, seed=3, group_id="grp", generation_index=5)
    c = shuffled_labels(lab, 20, seed=4, group_id="grp", generation_index=5)
    assert a == b and a != c
    assert Counter(v for _, v in a) == Counter(v for _, v in lab)
    assert len({k for k, _ in a}) == 3 and all(0 <= k < 20 for k, _ in a)
    # Uniform over calls.
    hits = Counter()
    for g in range(4000):
        for k, _ in shuffled_labels(
            ((0, GOOD),), 5, seed=0, group_id=str(g), generation_index=0
        ):
            hits[k] += 1
    assert min(hits.values()) > 700


def test_shuffled_arm_moves_labels():
    m, runs = row_layout([5] * 10)
    adv, tm = batch([(m, runs)], [-0.5])
    out, stats, records = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels={0: labels(10, [(0, HARMFUL)], gid="grp", gen=2)},
        config=cfg(shuffle_within_rollout=True, shuffle_seed=1, shuffle_mode="uniform"),
    )
    ((k, _),) = shuffled_labels(
        ((0, HARMFUL),), 10, seed=1, group_id="grp", generation_index=2
    )
    s, e = runs[k]
    assert float(out[0, s]) < -0.5 and records[0]["applied_labels"] == [[k, HARMFUL]]
    _invariants(out, adv, tm, [0])


def test_lam_harm_scales_harmful_labels():
    m, runs = row_layout([10] * 4)
    adv, tm = batch([(m, runs)], [-1.0])
    out, _, _ = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels={0: labels(4, [(0, HARMFUL), (1, GOOD)])},
        config=cfg(lam=0.5, lam_harm=0.25),
    )
    # j = -0.5, +1, 0, 0 -> jbar = 0.125; harmful shift = 0.5 * (-0.625), good = 0.5 * 0.875.
    assert float(out[0, runs[0][0]]) == pytest.approx(-1.0 - 0.3125, abs=1e-6)
    assert float(out[0, runs[1][0]]) == pytest.approx(-1.0 + 0.4375, abs=1e-6)
    _invariants(out, adv, tm, [0])


def test_stats_merge_and_metrics():
    m, runs = row_layout([10] * 6)
    adv, tm = batch([(m, runs)] * 2, [-0.5, 0.5])
    _, s1, _ = shape_turn_credit(
        adv[:1],
        train_mask=tm[:1],
        token_mask=tm[:1],
        row_labels={0: labels(6, [(2, HARMFUL)])},
        config=cfg(),
    )
    _, s2, _ = shape_turn_credit(
        adv[1:],
        train_mask=tm[1:],
        token_mask=tm[1:],
        row_labels={0: labels(6, [(2, GOOD)])},
        config=cfg(),
    )
    total = TurnCreditStats()
    total.merge(s1)
    total.merge(s2)
    met = total.to_metrics()
    assert met["turn_credit/shaped_rows"] == 2.0
    assert (
        met["turn_credit/good_calls"] == 1.0 and met["turn_credit/harmful_calls"] == 1.0
    )
    assert met["turn_credit/labeled_token_share"] == pytest.approx(20 / 120)
    assert met["turn_credit/adv_mean/harmful_neg"] == pytest.approx(-0.70833, abs=1e-4)
    assert met["turn_credit/adv_mean/good_pos"] == pytest.approx(0.70833, abs=1e-4)
    assert met["turn_credit/mass_moved_share"] == pytest.approx(
        2 * (10 * 0.20833 + 50 * 0.041667) / 60, abs=1e-4
    )
    assert TurnCreditStats().to_metrics() == {}


def test_config_parsing_and_validation():
    assert turn_credit_config(SimpleNamespace()) is None
    assert turn_credit_config(SimpleNamespace(turn_credit={"enabled": False})) is None
    parsed = turn_credit_config(
        SimpleNamespace(turn_credit={"enabled": True, "judge": JUDGE, "lam": 0.75})
    )
    assert (
        parsed is not None
        and parsed.lam == 0.75
        and parsed.kappa == 0.8
        and parsed.apply
    )
    with pytest.raises(ValidationError):
        TurnCreditConfig(enabled=True, judge=JUDGE, kappa=1.0)
    with pytest.raises(ValidationError):
        TurnCreditConfig(enabled=True)
    with pytest.raises(ValidationError):
        TurnCreditConfig(enabled=True, judge=JUDGE, typo=1)
    with pytest.raises(ValidationError):
        TurnCreditConfig(enabled=True, judge={**JUDGE, "budget_s": 600})


def test_size_matched_shuffle_keeps_sizes_and_avoids_real_positions():
    tokens = [100, 4000, 120, 3500, 90, 5000, 110, 80, 4200, 95]
    lab = ((1, GOOD), (5, HARMFUL))
    for g in range(200):
        out = shuffled_labels(
            lab, 10, seed=0, group_id=str(g), generation_index=0, call_tokens=tokens
        )
        calls = dict(out)
        assert sorted(calls.values()) == sorted(v for _, v in lab)
        assert not set(calls) & {
            1,
            5,
        }  # other large calls exist, so never a real position
        for c in calls:
            assert tokens[c] >= 3000  # within 1.5x of 4000 / 5000
    # Deterministic per rollout.
    assert shuffled_labels(
        lab, 10, seed=0, group_id="x", generation_index=3, call_tokens=tokens
    ) == shuffled_labels(
        lab, 10, seed=0, group_id="x", generation_index=3, call_tokens=tokens
    )
    # No call in range: the nearest in log size (here the only other large call).
    out = shuffled_labels(
        ((0, GOOD),),
        3,
        seed=0,
        group_id="y",
        generation_index=0,
        call_tokens=[9000, 10, 2000],
    )
    assert out == ((2, GOOD),)


def test_size_matched_shuffled_arm_moves_similar_mass():
    lengths = [40, 4000, 50, 3800, 60, 45, 4100, 55]
    m, runs = row_layout(lengths)
    adv, tm = batch([(m, runs)], [-0.5])
    real_labels = {0: labels(8, [(1, HARMFUL)], gid="grp", gen=1)}
    _, real, _ = shape_turn_credit(
        adv, train_mask=tm, token_mask=tm, row_labels=real_labels, config=cfg()
    )
    _, shuf, rec = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels=real_labels,
        config=cfg(shuffle_within_rollout=True),
    )
    assert rec[0]["applied_labels"][0][0] in (3, 6)
    assert shuf.moved_mass == pytest.approx(real.moved_mass, rel=0.1)
    _, uni, _ = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels=real_labels,
        config=cfg(shuffle_within_rollout=True, shuffle_mode="uniform", shuffle_seed=2),
    )
    assert uni.shaped_rows == 1


def _shape_one(lengths, lab, turns, ok, a=-0.5):
    m, runs = row_layout(lengths)
    adv, tm = batch([(m, runs)], [a])
    out, stats, records = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels={0: lab},
        config=cfg(),
        row_turns={0: turns},
        trailing_ok={0: ok},
    )
    return out, stats, records, runs, adv


def test_trailing_empty_call_run_is_accepted_only_with_every_proof():
    # 3 judge calls; the row has a 4th, 3-token run (an empty model reply the service proved at the end).
    lengths = [10, 10, 10, 3]
    good = labels(3, [(1, HARMFUL)], aligned=True, trailing=1)
    out, stats, records, runs, adv = _shape_one(lengths, good, turns=4, ok=True)
    assert stats.shaped_rows == 1 and stats.trailing_extra_accepted == 1
    s1, e1 = runs[1]
    s3, e3 = runs[3]
    s0, _ = runs[0]
    # The label sits on run 1 (call 1); the extra trailing run is unlabeled like run 0.
    assert float(out[0, s1]) < -0.5 < float(out[0, s0])
    assert float(out[0, s3]) == pytest.approx(float(out[0, s0]))
    _invariants(out, adv, batch([row_layout(lengths)], [-0.5])[1], [0])
    assert (
        records[0]["trailing_extra_runs"] == 1
        and records[0]["run_lengths_tail"][-1] == 3
    )

    rejected = [
        (
            good,
            4,
            False,
            "drop:n_runs_mismatch",
        ),  # the advantage stage's token check failed
        (good, None, True, "drop:n_runs_mismatch"),  # no turns tag
        (good, 3, True, "drop:turns_tag_mismatch"),  # turns tag disagrees with the runs
        (
            labels(3, [(1, HARMFUL)], aligned=None, trailing=1),
            4,
            True,
            "drop:n_runs_mismatch",
        ),  # older service
        (
            labels(3, [(1, HARMFUL)], aligned=False, trailing=1),
            4,
            True,
            "drop:alignment_not_proven",
        ),
        (
            labels(3, [(1, HARMFUL)], aligned=True, trailing=0),
            4,
            True,
            "drop:n_runs_mismatch",
        ),  # none proven
    ]
    for lab, turns, ok, status in rejected:
        out, stats, records, _, adv = _shape_one(lengths, lab, turns, ok)
        assert (
            out is adv and stats.shaped_rows == 0 and records[0]["status"] == status
        ), (lab, turns, ok)
    # Two extra runs with one proven empty call; fewer runs than calls.
    out, stats, records, _, adv = _shape_one([10, 10, 10, 3, 3], good, turns=5, ok=True)
    assert records[0]["status"] == "drop:n_runs_mismatch"
    out, stats, records, _, adv = _shape_one([10, 10], good, turns=2, ok=True)
    assert records[0]["status"] == "drop:n_runs_mismatch"


def test_turns_tag_must_match_runs_when_present():
    lengths = [10, 10, 10]
    lab = labels(3, [(1, HARMFUL)])
    out, stats, records, _, adv = _shape_one(lengths, lab, turns=4, ok=False)
    assert records[0]["status"] == "drop:turns_tag_mismatch" and out is adv
    out, stats, records, _, adv = _shape_one(
        lengths, lab, turns=None, ok=False
    )  # unknown: counts decide
    assert records[0]["status"] == "shaped"
    out, stats, records, _, adv = _shape_one(lengths, lab, turns=3, ok=False)
    assert records[0]["status"] == "shaped" and stats.trailing_extra_accepted == 0


@pytest.mark.parametrize("mode", ["all", "passing", "failing"])
def test_shape_rollouts_by_advantage_sign(mode):
    m, runs = row_layout([10] * 4)
    adv, tm = batch([(m, runs)] * 2, [0.5, -0.5])
    labs = {0: labels(4, [(1, GOOD)], gen=0), 1: labels(4, [(2, HARMFUL)], gen=1)}
    out, stats, records = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels=labs,
        config=cfg(shape_rollouts=mode),
    )
    shaped = {r["generation_index"]: r["status"] for r in records}
    expect = {
        "all": ("shaped", "shaped"),
        "passing": ("shaped", "noop:not_passing"),
        "failing": ("noop:not_failing", "shaped"),
    }[mode]
    assert (shaped[0], shaped[1]) == expect
    for row, status in ((0, shaped[0]), (1, shaped[1])):
        same = bool(torch.equal(out[row], adv[row]))
        assert same == (status != "shaped")
    _invariants(out, adv, tm, [0, 1])


def test_token_identity_check():
    """Shaped only when every run has the token count of the same call in the export the judge read (LOG 094)."""
    lengths = [5, 3, 7]
    m, runs = row_layout(lengths)
    m2, runs2 = row_layout(lengths + [1])  # one trailing empty call (1 token)

    def shape(lab, row=(m, runs), **kw):
        adv, tm = batch([row], [-0.5])
        trailing_ok = {0: True} if len(row[1]) > lab.n_calls else None
        out, stats, records = shape_turn_credit(
            adv,
            train_mask=tm,
            token_mask=tm,
            row_labels={0: lab},
            config=cfg(require_token_match=True, **kw),
            row_turns={0: len(row[1])},
            trailing_ok=trailing_ok,
        )
        return out, stats, records[0]["status"] if "status" in records[0] else "shaped"

    harm = [(1, HARMFUL)]
    assert shape(labels(3, harm, aligned=True, step_tokens=[5, 3, 7]))[2] == "shaped"
    # The export may list trailing steps that left no run in the row.
    assert shape(labels(3, harm, aligned=True, step_tokens=[5, 3, 7, 0]))[2] == "shaped"
    # Same call count, another trajectory: a re-dispatched row training the other session's tokens.
    out, stats, status = shape(labels(3, harm, aligned=True, step_tokens=[5, 4, 7]))
    assert status == "drop:token_mismatch" and stats.drops == {"token_mismatch": 1}
    adv, tm = batch([(m, runs)], [-0.5])
    assert torch.equal(out, adv)
    # More runs than the export has steps; no token counts from the service.
    assert (
        shape(labels(3, harm, aligned=True, step_tokens=[5, 3]))[2]
        == "drop:token_mismatch"
    )
    assert shape(labels(3, harm, aligned=True))[2] == "drop:no_token_fingerprint"
    # A trailing empty call's run is checked too.
    lab = labels(3, harm, aligned=True, trailing=1, step_tokens=[5, 3, 7, 1])
    assert shape(lab, row=(m2, runs2))[2] == "shaped"
    lab = labels(3, harm, aligned=True, trailing=1, step_tokens=[5, 3, 7, 2])
    assert shape(lab, row=(m2, runs2))[2] == "drop:token_mismatch"
    # Off: the old behaviour.
    adv, tm = batch([(m, runs)], [-0.5])
    _, stats, _ = shape_turn_credit(
        adv,
        train_mask=tm,
        token_mask=tm,
        row_labels={0: labels(3, harm, aligned=True)},
        config=cfg(require_token_match=False),
    )
    assert stats.shaped_rows == 1
    assert TurnCreditConfig(enabled=True, judge=JUDGE).require_token_match is True


def test_tokens_match_rule():
    """The validated identity rule (LOG 095): exact per run, an unrecorded final step skipped, nothing extra."""
    assert tokens_match([5, 3, 7], (5, 3, 7))
    assert tokens_match(
        [5, 3, 7], (5, 3, 7, 0, None)
    )  # trailing steps that produced no run
    assert tokens_match(
        [5, 3, 7], (5, 3, None)
    )  # the final step never finished: its usage was not recorded
    assert not tokens_match(
        [5, 3, 7], (5, 4, 7)
    )  # another trajectory with the same call count
    assert not tokens_match(
        [5, 3, 7], (5, None, 7)
    )  # an unrecorded step can only be the last one
    assert not tokens_match(
        [5, 3, 7], (5, 3, 7, 4)
    )  # the export generated more than the row holds
    assert not tokens_match([5, 3, 7], (5, 3))  # more runs than steps
    assert not tokens_match(
        [5, 9], (5, None)
    )  # fewer than MIN_MATCHED_RUNS exact matches
    assert not tokens_match([], ())

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
"""Unit tests for turn-level credit assignment (CPU only, no Ray/Megatron)."""

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    TurnLevelGeneralizedAdvantageEstimator,
)
from nemo_rl.algorithms.turn_level import (
    build_turn_rewards,
    build_turn_spans,
    build_turn_value_batch,
    gather_turn_values,
    scatter_turns_to_anchors,
    scatter_turns_to_tokens,
    turn_gae,
    turn_level_metrics,
    validate_turn_spans,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


# ---------------------------------------------------------------- helpers
def _msg(role, n, start_id=0):
    return {"role": role, "token_ids": torch.arange(start_id, start_id + n)}


def _agentic_log(turn_lens, obs_len=3, prompt_len=5):
    """[prompt, (assistant, observation)*] — the NeMo-Gym alternating layout."""
    log = [_msg("user", prompt_len)]
    for i, n in enumerate(turn_lens):
        log.append(_msg("assistant", n))
        if i < len(turn_lens) - 1:
            log.append(_msg("user", obs_len))
    return log


def _flat_len(log):
    return sum(len(m["token_ids"]) for m in log)


def _token_mask_from(log, seq_len):
    mask = torch.zeros(seq_len, dtype=torch.long)
    pos = 0
    for m in log:
        n = len(m["token_ids"])
        if m["role"] == "assistant":
            mask[pos : pos + n] = 1
        pos += n
    return mask


class _FakeLossCfg:
    use_kl_in_reward = False
    reference_policy_kl_penalty = 0.0
    reference_policy_kl_type = "low_var_kl"


def _estimator(**overrides):
    cfg = {
        "turn_gae_gamma": 1.0,
        "turn_gae_lambda_value": 1.0,
        "turn_gae_lambda_policy": 1.0,
        "normalize_advantages": False,
    }
    cfg.update(overrides)
    return TurnLevelGeneralizedAdvantageEstimator(cfg, _FakeLossCfg())


# ---------------------------------------------------------------- spans
def test_spans_locate_every_assistant_message():
    log = _agentic_log([4, 2, 3], obs_len=3, prompt_len=5)
    seq_len = _flat_len(log) + 7  # padding
    spans = build_turn_spans([log], seq_len)

    # prompt 5 | asst 4 @5 | obs 3 | asst 2 @12 | obs 3 | asst 3 @17
    assert spans.num_turns.tolist() == [3]
    assert spans.anchor_pos[0, :3].tolist() == [5, 12, 17]
    assert spans.turn_ntokens[0, :3].tolist() == [4, 2, 3]
    assert spans.turn_valid[0, :3].all()
    assert spans.anchor_mask[0].sum() == 3
    assert spans.anchor_mask[0, [5, 12, 17]].tolist() == [1, 1, 1]

    ti = spans.turn_index[0]
    assert ti[:5].tolist() == [-1] * 5
    assert ti[5:9].tolist() == [0] * 4
    assert ti[9:12].tolist() == [-1] * 3
    assert ti[12:14].tolist() == [1, 1]
    assert ti[17:20].tolist() == [2, 2, 2]
    assert ti[20:].tolist() == [-1] * (seq_len - 20)


def test_anchors_are_always_response_tokens():
    logs = [_agentic_log([4, 2, 3]), _agentic_log([7])]
    seq_len = max(_flat_len(x) for x in logs)
    spans = build_turn_spans(logs, seq_len)
    token_mask = torch.stack([_token_mask_from(x, seq_len) for x in logs])
    # the property the whole design rests on: anchors ⊆ response mask
    assert bool((spans.anchor_mask.bool() & ~token_mask.bool()).sum() == 0)
    validate_turn_spans(spans, token_mask, torch.ones(2))


def test_ragged_batch_is_left_aligned_and_padded():
    logs = [_agentic_log([2, 2, 2, 2]), _agentic_log([5])]
    seq_len = max(_flat_len(x) for x in logs)
    spans = build_turn_spans(logs, seq_len)
    assert spans.anchor_pos.shape[1] == 4
    assert spans.num_turns.tolist() == [4, 1]
    assert spans.turn_valid[1].tolist() == [True, False, False, False]
    assert spans.turn_ntokens[1].tolist() == [5, 0, 0, 0]


def test_consecutive_assistant_messages_are_separate_turns():
    log = [_msg("user", 3), _msg("assistant", 2), _msg("assistant", 4)]
    spans = build_turn_spans([log], _flat_len(log))
    assert spans.num_turns.tolist() == [2]
    assert spans.anchor_pos[0, :2].tolist() == [3, 5]


def test_empty_assistant_message_is_not_a_turn():
    log = [_msg("user", 3), _msg("assistant", 0), _msg("assistant", 4)]
    spans = build_turn_spans([log], _flat_len(log))
    assert spans.num_turns.tolist() == [1]
    assert spans.anchor_pos[0, 0].item() == 3


def test_assistant_at_index_zero_raises():
    # The value head is right-shifted, so position 0 carries no value.
    with pytest.raises(ValueError, match="token 0"):
        build_turn_spans([[_msg("assistant", 3)]], 3)


def test_seq_len_overflow_raises():
    log = _agentic_log([4, 4])
    with pytest.raises(ValueError, match="flattens to"):
        build_turn_spans([log], _flat_len(log) - 1)


def test_sample_without_turns_is_reported_not_fatal(capsys):
    """An empty trajectory contributes no tokens, so it must not kill the step."""
    log = [_msg("user", 4)]
    spans = build_turn_spans([log], 4)
    empty_mask = torch.zeros(1, 4, dtype=torch.long)
    validate_turn_spans(spans, empty_mask, torch.ones(1))
    assert "no assistant message" in capsys.readouterr().out
    # a sample that carries no gradient is not even worth reporting
    validate_turn_spans(spans, empty_mask, torch.zeros(1))
    assert "no assistant message" not in capsys.readouterr().out


def test_validate_rejects_response_tokens_outside_any_turn():
    """A response token with no turn would carry a policy gradient with no advantage."""
    log = _agentic_log([3])
    seq_len = _flat_len(log)
    spans = build_turn_spans([log], seq_len)
    bad_mask = torch.ones(1, seq_len, dtype=torch.long)  # claims the prompt is trainable
    with pytest.raises(ValueError, match="belong to no turn"):
        validate_turn_spans(spans, bad_mask, torch.ones(1))


# ---------------------------------------------------------------- GAE math
def test_turn_gae_matches_hand_computation():
    v = torch.tensor([[0.2, 0.5, 0.4]])
    r = torch.tensor([[0.0, 0.0, 1.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    gamma, lam = 0.9, 0.5

    # backwards by hand
    d2 = 1.0 + gamma * 0.0 - 0.4
    a2 = d2
    d1 = 0.0 + gamma * 0.4 - 0.5
    a1 = d1 + gamma * lam * a2
    d0 = 0.0 + gamma * 0.5 - 0.2
    a0 = d0 + gamma * lam * a1

    adv, ret = turn_gae(v, r, valid, gamma, lam)
    assert torch.allclose(adv, torch.tensor([[a0, a1, a2]]), atol=1e-6)
    assert torch.allclose(ret, adv + v, atol=1e-6)


def test_lambda_one_gamma_one_gives_monte_carlo_returns():
    """The property that makes turn-level stage-B comparable to the token-level run.

    With gamma=lambda=1 and a terminal-only reward, G_k == R for every turn, so
    the critic regresses on the same Monte-Carlo target it does today — only at
    ~200 anchors instead of ~45k tokens.
    """
    torch.manual_seed(0)
    v = torch.randn(4, 6)
    valid = torch.ones(4, 6, dtype=torch.bool)
    valid[1, 4:] = False
    valid[2, 2:] = False
    R = torch.tensor([1.0, 0.0, 1.0, 0.5])
    r = torch.zeros(4, 6)
    last = valid.long().sum(1) - 1
    r[torch.arange(4), last] = R

    _, ret = turn_gae(v * valid, r, valid, 1.0, 1.0)
    for i in range(4):
        n = int(valid[i].sum())
        assert torch.allclose(ret[i, :n], R[i].expand(n), atol=1e-5)


def test_invalid_turns_do_not_leak_into_valid_ones():
    v = torch.tensor([[0.3, 0.7, 99.0]])
    r = torch.tensor([[0.0, 1.0, 0.0]])
    valid = torch.tensor([[True, True, False]])
    adv, _ = turn_gae(v, r, valid, 1.0, 1.0)
    # turn 1 is terminal: A = R - V = 1 - 0.7; turn 0: A = 1 - 0.3
    assert torch.allclose(adv[0, :2], torch.tensor([0.7, 0.3]), atol=1e-6)


# ---------------------------------------------------------------- scatter
def test_gather_and_scatter_round_trip():
    log = _agentic_log([4, 2, 3])
    seq_len = _flat_len(log)
    spans = build_turn_spans([log], seq_len)
    values = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)

    tv = gather_turn_values(values, spans)
    assert tv[0, :3].tolist() == [5.0, 12.0, 17.0]

    per_turn = torch.tensor([[10.0, 20.0, 30.0]])
    tok = scatter_turns_to_tokens(per_turn, spans, seq_len)
    assert tok[0, 5:9].tolist() == [10.0] * 4
    assert tok[0, 9:12].tolist() == [0.0] * 3  # observation gets nothing
    assert tok[0, 12:14].tolist() == [20.0] * 2
    assert tok[0, 17:20].tolist() == [30.0] * 3

    anc = scatter_turns_to_anchors(per_turn, spans, seq_len)
    assert anc.sum().item() == 60.0
    assert anc[0, [5, 12, 17]].tolist() == [10.0, 20.0, 30.0]


def test_turn_rewards_land_on_the_last_turn():
    logs = [_agentic_log([2, 2, 2]), _agentic_log([3])]
    seq_len = max(_flat_len(x) for x in logs)
    spans = build_turn_spans(logs, seq_len)
    tr = build_turn_rewards(torch.tensor([1.0, 0.25]), spans)
    assert tr[0].tolist()[:3] == [0.0, 0.0, 1.0]
    assert tr[1, 0].item() == 0.25
    assert tr[1, 1:].sum().item() == 0.0


def test_token_penalty_is_summed_into_its_turn():
    log = _agentic_log([4, 2])
    seq_len = _flat_len(log)
    spans = build_turn_spans([log], seq_len)
    penalty = torch.zeros(1, seq_len)
    penalty[0, 5:9] = -0.1  # turn 0 (4 tokens)
    penalty[0, 12:14] = -0.5  # turn 1 (2 tokens)
    tr = build_turn_rewards(torch.tensor([1.0]), spans, penalty)
    assert tr[0, 0].item() == pytest.approx(-0.4, abs=1e-6)
    assert tr[0, 1].item() == pytest.approx(1.0 - 1.0, abs=1e-6)


# ---------------------------------------------------------------- estimator
def test_estimator_advantage_is_constant_within_a_turn():
    logs = [_agentic_log([4, 2, 3])]
    seq_len = _flat_len(logs[0])
    spans = build_turn_spans(logs, seq_len)
    token_mask = torch.stack([_token_mask_from(x, seq_len) for x in logs])
    torch.manual_seed(1)
    values = torch.randn(1, seq_len)

    est = _estimator(turn_gae_lambda_policy=0.9)
    adv, ret = est.compute_advantage(
        prompt_ids=None,
        rewards=torch.tensor([1.0]),
        mask=token_mask,
        values=values,
        turn_spans=spans,
        sample_mask=torch.ones(1),
    )
    assert len(set(adv[0, 5:9].tolist())) == 1
    assert len(set(adv[0, 12:14].tolist())) == 1
    assert adv[0, 9:12].abs().sum().item() == 0.0  # observations carry none
    # anchor supervision: returns live only at anchors
    assert ret[0].nonzero().flatten().tolist() == [5, 12, 17]


def test_estimator_lambda_one_reproduces_the_monte_carlo_baseline():
    logs = [_agentic_log([4, 2, 3])]
    seq_len = _flat_len(logs[0])
    spans = build_turn_spans(logs, seq_len)
    token_mask = torch.stack([_token_mask_from(x, seq_len) for x in logs])
    values = torch.rand(1, seq_len)
    R = 1.0

    est = _estimator()
    adv, ret = est.compute_advantage(
        prompt_ids=None,
        rewards=torch.tensor([R]),
        mask=token_mask,
        values=values,
        turn_spans=spans,
        sample_mask=torch.ones(1),
    )
    # A_k = R - V(s_k) at gamma=lambda=1
    for anchor, lo, hi in ((5, 5, 9), (12, 12, 14), (17, 17, 20)):
        expected = R - values[0, anchor]
        assert torch.allclose(adv[0, lo:hi], expected.expand(hi - lo), atol=1e-5)
    assert torch.allclose(ret[0, [5, 12, 17]], torch.full((3,), R), atol=1e-5)


def test_estimator_returns_are_covered_by_the_critic_mask():
    """Every turn return must land where the critic batch is actually supervised.

    The estimator's `returns` layout and `build_turn_value_batch`'s token_mask
    are set in two different places; if they drift apart, targets outside the
    mask are silently dropped and the critic trains on less than it looks like.
    """
    logs = [_agentic_log([4, 2, 3]), _agentic_log([5])]
    seq_len = max(_flat_len(x) for x in logs)
    spans = build_turn_spans(logs, seq_len)
    td = _train_data(logs, seq_len)

    est = _estimator(turn_gae_lambda_policy=0.9)
    _, ret = est.compute_advantage(
        prompt_ids=None,
        rewards=torch.tensor([1.0, 0.0]),
        mask=td["token_mask"],
        values=td["values"],
        turn_spans=spans,
        sample_mask=td["sample_mask"],
    )
    td["returns"] = ret
    batch = build_turn_value_batch(td, spans)
    covered = batch["token_mask"].bool()
    assert bool(((ret != 0) & ~covered).sum() == 0)
    # and the mask has exactly one position per real turn
    assert int(covered.sum()) == int(spans.num_turns.sum())


def test_estimator_normalizes_over_response_tokens_only():
    logs = [_agentic_log([4, 2, 3])]
    seq_len = _flat_len(logs[0])
    spans = build_turn_spans(logs, seq_len)
    token_mask = torch.stack([_token_mask_from(x, seq_len) for x in logs])
    est = _estimator(normalize_advantages=True, turn_gae_lambda_policy=0.5)
    adv, _ = est.compute_advantage(
        prompt_ids=None,
        rewards=torch.tensor([1.0]),
        mask=token_mask,
        values=torch.rand(1, seq_len),
        turn_spans=spans,
        sample_mask=torch.ones(1),
    )
    resp = adv[token_mask.bool()]
    assert resp.mean().abs().item() < 1e-5
    assert abs(resp.std(unbiased=False).item() - 1.0) < 1e-3
    assert adv[~token_mask.bool()].abs().sum().item() == 0.0


def test_missing_lambda_fails_loud():
    with pytest.raises(ValueError, match="must be set explicitly"):
        TurnLevelGeneralizedAdvantageEstimator(
            {
                "turn_gae_gamma": 1.0,
                "turn_gae_lambda_value": None,
                "turn_gae_lambda_policy": 1.0,
                "normalize_advantages": True,
            },
            _FakeLossCfg(),
        )


def test_missing_turn_spans_fails_loud():
    est = _estimator()
    with pytest.raises(ValueError, match="requires turn_spans"):
        est.compute_advantage(
            prompt_ids=None,
            rewards=torch.tensor([1.0]),
            mask=torch.ones(1, 4, dtype=torch.long),
            values=torch.zeros(1, 4),
        )


# ---------------------------------------------------------------- critic batch
def _train_data(logs, seq_len):
    token_mask = torch.stack([_token_mask_from(x, seq_len) for x in logs])
    return BatchedDataDict(
        {
            "input_ids": torch.zeros(len(logs), seq_len, dtype=torch.long),
            "input_lengths": torch.tensor([_flat_len(x) for x in logs]),
            "token_mask": token_mask,
            "sample_mask": torch.ones(len(logs)),
            "values": torch.rand(len(logs), seq_len),
            "returns": torch.zeros(len(logs), seq_len),
            # keys the policy needs but the critic must not inherit
            "advantages": torch.zeros(len(logs), seq_len),
            "prev_logprobs": torch.zeros(len(logs), seq_len),
        }
    )


def test_critic_batch_swaps_in_the_anchor_mask():
    logs = [_agentic_log([4, 2, 3])]
    seq_len = _flat_len(logs[0])
    spans = build_turn_spans(logs, seq_len)
    td = _train_data(logs, seq_len)

    batch = build_turn_value_batch(td, spans)
    # This swap is the whole mechanism: process_global_batch derives
    # global_valid_toks from token_mask, so the value loss becomes a per-turn
    # mean with no change to MseValueLossFn or the value workers.
    assert batch["token_mask"].sum() == 3
    assert torch.equal(batch["token_mask"], spans.anchor_mask)
    assert batch["token_mask"].dtype == td["token_mask"].dtype
    # the policy-only tensors stay out of the critic's batch
    assert "advantages" not in batch and "prev_logprobs" not in batch
    for k in ("input_ids", "input_lengths", "sample_mask", "returns", "values"):
        assert k in batch
    # train_data itself is untouched (the policy still needs the full mask)
    assert td["token_mask"].sum() == 9


def test_critic_batch_requires_returns():
    logs = [_agentic_log([2, 2])]
    seq_len = _flat_len(logs[0])
    spans = build_turn_spans(logs, seq_len)
    td = _train_data(logs, seq_len)
    del td["returns"]
    with pytest.raises(ValueError, match="returns"):
        build_turn_value_batch(td, spans)


# ---------------------------------------------------------------- metrics
def test_turn_metrics_are_reported():
    logs = [_agentic_log([2, 2, 2]), _agentic_log([3])]
    seq_len = max(_flat_len(x) for x in logs)
    spans = build_turn_spans(logs, seq_len)
    tv = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.0, 0.0]])
    ta = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]) - tv
    m = turn_level_metrics(tv, ta, spans, torch.ones(2))
    assert m["turn/total_turns"] == 4.0
    assert m["turn/num_turns_mean"] == pytest.approx(2.0)
    assert m["turn/num_turns_max"] == 3.0
    assert "advantage/turn_abs_mean_prenorm" in m
    # per-position EV / terminal AUC deliberately live in the existing
    # _positional_value_metrics / terminal_value_reward_auc, not here
    assert not any(k.startswith("critic/turn_ev") for k in m)


def test_turn_metrics_respect_the_sample_mask():
    logs = [_agentic_log([2, 2, 2]), _agentic_log([3])]
    spans = build_turn_spans(logs, max(_flat_len(x) for x in logs))
    tv = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.0, 0.0]])
    m = turn_level_metrics(tv, torch.zeros_like(tv), spans, torch.tensor([1.0, 0.0]))
    assert m["turn/total_turns"] == 3.0

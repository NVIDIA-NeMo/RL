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
"""SingleController collector for the legacy async-PPO diagnostics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.legacy_ppo_diagnostics import pooled_explained_var
from nemo_rl.algorithms.single_controller_utils.legacy_diagnostics import (
    NULL_EFFICIENCY_CLOCK,
    LegacyEfficiencyClock,
    LegacyPPODiagnostics,
    add_legacy_timing_aliases,
    fetch_generation_logger_metrics,
    group_key,
    is_pad_row,
    legacy_efficiency_summary,
    legacy_setup_timing_metrics,
)
from nemo_rl.experience.legacy_rollout_metrics import ROLLOUT_DEBUG_TAG
from nemo_rl.utils.timer import Timer

# Column set of a legacy rollout_debug_step*.jsonl line (golden run exp_005).
LEGACY_JSONL_KEYS = {
    "rollout_info",
    "trace_metadata",
    "rollout_local_idx",
    "trace_in_rollout_idx",
    "is_empty_rollout",
    "trace_group_id",
    "trace_rollout_id",
    "reward",
    "sample_loss_mask",
    "pre_seq_error_sample_loss_mask",
    "seq_mult_prob_error",
    "masked_by_seq_logprob_error",
    "advantage",
    "advantage_min",
    "advantage_max",
    "value_first_token",
    "num_generated_tokens",
    "total_tokens",
    "trainer_weight_version",
    "idx",
}


def _tag(local, trace, kind, *, empty=False, prompt_tokens=2, version=3, extra=None):
    row = {
        "rollout_info": {"instance_id": f"inst-{local}", "mask_sample": empty},
        "trace_metadata": {"kind": kind, "turns": 1, "prompt_tokens": prompt_tokens},
        "rollout_local_idx": local,
        "trace_in_rollout_idx": trace,
        "is_empty_rollout": empty,
    }
    return {
        "weight_version": version,
        "prompt_idx": 11,
        ROLLOUT_DEBUG_TAG: json.dumps(row),
        **(extra or {}),
    }


class _Estimator:
    last_metrics = {"adv_raw/std": 0.7, "adv_raw/mean": 0.1}


def _batch():
    """5 real rows of 2 prompt groups (gpp=2) + 1 DP pad row; S = 6.

    grpA: rollout 0 = main trace + subagent trace; rollout 1 = one trace.
    grpB: rollout 0 = one trace; rollout 1 = the masked empty placeholder.
    """
    sample_ids = ["grpA_g0", "grpA_g1", "grpA_g2", "grpB_g0", "grpB_g1", "grpA_g0_pad0"]
    tags = [
        _tag(0, 0, "uncompacted", version=3),
        _tag(0, 1, "subagent", version=3, prompt_tokens=4),
        _tag(1, 0, "uncompacted", version=3),
        _tag(0, 0, "uncompacted", version=1),
        _tag(1, 0, "empty", empty=True, version=1, prompt_tokens=1),
        _tag(0, 0, "uncompacted", version=3),  # pad copy of grpA_g0
    ]
    token_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    mask_sample = torch.tensor([False, False, False, False, True, False])
    final_sample_mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    values = torch.arange(36, dtype=torch.float32).reshape(6, 6) / 40.0
    returns = rewards.unsqueeze(-1) * token_mask
    advantages = (returns - values) * token_mask
    gen_lp = torch.zeros(6, 6)
    pol_lp = torch.full((6, 6), -0.1)
    seq_err = torch.tensor([1.1, 1.1, 1.1, 1.1, 0.0, 1.1])
    seq_tensors = {
        "pre_seq_error_sample_loss_mask": torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        "seq_mult_prob_error": seq_err,
        "masked_by_seq_logprob_error": torch.tensor(
            [False, False, True, False, False, False]
        ),
    }
    return dict(
        sample_ids=sample_ids,
        tags=tags,
        sequence_lengths=[10, 20, 30, 40, 1, 10],
        trainer_version=4,
        rewards=rewards,
        token_mask=token_mask,
        mask_sample=mask_sample,
        final_sample_mask=final_sample_mask,
        seq_error_tensors=seq_tensors,
        generation_logprobs=gen_lp,
        policy_logprobs=pol_lp,
        values=values,
        advantages=advantages,
        returns=returns,
        estimator_mask=token_mask,
        advantage_estimator=_Estimator(),
    )


def _diag(**overrides):
    kwargs = dict(
        num_generations_per_prompt=2,
        policy_training_start_step=0,
        log_rollout_debug=True,
        log_critic_diagnostics=True,
        log_post_update_critic_metrics=False,
    )
    kwargs.update(overrides)
    diag = LegacyPPODiagnostics(**kwargs)
    diag.strict = True
    return diag


def test_row_identity_helpers():
    assert is_pad_row("abc_g3_pad0") and not is_pad_row("abc_g3")
    assert group_key("abc-1_g3") == "abc-1"
    assert group_key("abc-1_g3_pad12") == "abc-1"


def test_from_algo_config_reads_ppo_flags():
    from nemo_rl.algorithms.ppo import PPOConfig

    cfg = PPOConfig(
        num_generations_per_prompt=4,
        policy_training_start_step=7,
        log_post_update_critic_metrics=True,
        log_rollout_dump=True,
        rollout_dump_period=5,
    )
    diag = LegacyPPODiagnostics.from_algo_config(cfg)
    assert diag.log_rollout_debug and diag.log_critic_diagnostics
    assert diag.post_update_enabled
    assert (diag.num_generations_per_prompt, diag.policy_training_start_step) == (4, 7)
    assert [diag.rollout_dump_due(s) for s in (4, 5, 10)] == [False, True, True]
    assert not LegacyPPODiagnostics.from_algo_config(PPOConfig()).rollout_dump_due(5)
    with pytest.raises(ValueError, match="rollout_dump_period"):
        _diag(log_rollout_dump=True, rollout_dump_period=0)


def test_on_advantage_stage_and_step_metrics():
    diag = _diag()
    batch = _batch()
    diag.on_advantage_stage(**batch)
    value_result = {"grad_norm_groups": {"moe": 3.0, "value_head": 4.0}}
    m = diag.step_metrics(value_result=value_result, buffer_size=7, sc_reward=0.9)
    # Legacy reward: per rollout (first trace), masked rollouts included.
    assert m["reward"] == pytest.approx(0.25)
    assert m["reward_trained_rows"] == 0.9
    resp = batch["advantages"][:5][batch["token_mask"][:5].bool()]
    assert m["advantages/mean"] == pytest.approx(resp.mean().item())
    assert m["advantages/min"] == pytest.approx(resp.min().item())

    # Multi-trace composition over the 5 unpadded rows.
    assert m["multi_trace/num_traces"] == 5
    assert m["multi_trace/num_rollouts"] == 4
    assert m["multi_trace/padding_rows"] == 1
    assert m["multi_trace/traces_per_rollout_max"] == 2
    assert m["multi_trace/masked_trace_fraction"] == pytest.approx(2 / 5)
    assert m["multi_trace/fully_masked_rollout_fraction"] == pytest.approx(2 / 4)
    assert m["multi_trace/trace_kind_count/subagent"] == 1.0
    assert m["multi_trace/empty_rollout_trace_fraction"] == pytest.approx(1 / 5)
    assert m["multi_trace/env_masked_trace_fraction"] == pytest.approx(1 / 5)
    assert m["multi_trace/seq_logprob_masked_trace_fraction"] == pytest.approx(1 / 5)
    # Trainable tokens: the loss's token_mask[:, 1:] * (sample_mask > 0).
    assert m["multi_trace/trainable_tokens/total"] == 3 + 2 + 4
    assert m["multi_trace/trainable_tokens/by_segment_kind/subagent"] == 2
    assert "logprob_error/seq_mult_prob_error/by_trace_in_rollout_idx/1/mean" in m
    assert m["logprob_error/seq_masked_fraction/by_segment_kind/uncompacted"] == (
        pytest.approx(1 / 3)
    )
    assert m["logprob_error/token_abs_err/by_position_quartile/q1"] == (
        pytest.approx(0.1)
    )
    # Legacy-only aggregates over the unpadded rows.
    assert m["total_num_tokens"] == 10 + 20 + 30 + 40 + 1
    assert m["mean_prompt_length"] == pytest.approx((2 + 4 + 2 + 2 + 1) / 5)
    # One generation version per prompt group: grpA v3, grpB v1, trainer v4.
    assert m["avg_trajectory_age"] == pytest.approx(2.0)
    assert m["max_trajectory_policy_age"] == 3.0
    # adv_raw passes straight through from the estimator.
    assert m["adv_raw/std"] == 0.7
    # Critic: pooled pre-update EV overrides the training-pass number.
    ev_abs, _ = pooled_explained_var(
        batch["values"],
        batch["returns"],
        batch["token_mask"],
        batch["final_sample_mask"],
    )
    assert m["critic/explained_var"] == pytest.approx(ev_abs)
    assert "critic/ev_res" in m
    assert {"critic/ev_early", "critic/ece_mid", "critic/bias_late"} <= set(m)
    assert {"residual/b_loo_mean", "residual/frac_groups_mixed"} <= set(m)
    assert m["critic/abs_returns_mean"] == pytest.approx(
        float(
            (
                batch["returns"]
                * batch["token_mask"]
                * batch["final_sample_mask"][:, None]
            ).sum()
            / (batch["token_mask"] * batch["final_sample_mask"][:, None]).sum()
        )
    )
    assert m["critic/gnorm/value_head"] == 4.0
    assert m["critic/gnorm_frac/moe"] == pytest.approx(9 / 25)
    assert m["buffer_size"] == 7
    # Consumed: the next step starts clean.
    assert diag.step_metrics(value_result=None) == {}


def test_post_update_critic_metrics():
    diag = _diag(log_post_update_critic_metrics=True)
    diag.post_value_result = {
        "grad_norm": torch.tensor([0.0]),
        "loss": torch.tensor([0.2, 0.4]),
        "all_mb_metrics": {
            "returns_mean": [0.5],
            "values_mean": [0.5],
            "returns_sq_mean": [0.5],
            "residual_sq_mean": [0.1],
        },
    }
    m = diag.step_metrics(value_result=None)
    assert m["critic/loss_post_update"] == pytest.approx(0.3)
    assert m["critic/explained_var_post_update"] == pytest.approx(1 - 0.1 / 0.25)
    assert diag.post_value_result is None


def test_critic_diagnostics_flag_off():
    diag = _diag(log_critic_diagnostics=False)
    diag.on_advantage_stage(**_batch())
    m = diag.step_metrics(value_result={"grad_norm_groups": {}})
    assert not any(k.startswith(("critic/ev", "residual/")) for k in m)
    assert "critic/explained_var" not in m
    assert "multi_trace/num_traces" in m  # cheap breakdowns stay on


def test_rollout_debug_jsonl_matches_legacy_schema(tmp_path):
    from nemo_rl.utils.logger import Logger

    diag = _diag()
    batch = _batch()
    diag.on_advantage_stage(**batch)
    fake_logger = SimpleNamespace(base_log_dir=str(tmp_path))
    fake_logger.log_batched_dict_as_jsonl = lambda rows, name: (
        Logger.log_batched_dict_as_jsonl(fake_logger, rows, name)
    )
    diag.write_rollout_debug(fake_logger, step=12, trainer_weight_version=12)
    lines = [
        json.loads(line)
        for line in (tmp_path / "rollout_debug_step12.jsonl").read_text().splitlines()
    ]
    assert len(lines) == 5  # pad row excluded
    for i, line in enumerate(lines):
        assert set(line) == LEGACY_JSONL_KEYS
        assert line["idx"] == i
        assert all(
            isinstance(v, list) and len(v) == 1 for k, v in line.items() if k != "idx"
        )
    col = {k: [line[k][0] for line in lines] for k in LEGACY_JSONL_KEYS - {"idx"}}
    assert col["trace_group_id"] == [0, 0, 0, 1, 1]
    assert col["trace_rollout_id"] == [0, 0, 1, 2, 3]
    assert col["trace_in_rollout_idx"] == [0, 1, 0, 0, 0]
    assert col["is_empty_rollout"] == [False, False, False, False, True]
    assert col["trainer_weight_version"] == [12] * 5
    assert col["total_tokens"] == [10, 20, 30, 40, 1]
    assert col["num_generated_tokens"] == [3, 2, 2, 4, 0]
    assert col["masked_by_seq_logprob_error"] == [False, False, True, False, False]
    assert col["sample_loss_mask"] == [1.0, 1.0, 0.0, 1.0, 0.0]
    # First generated token of row 0 is position 2.
    values, adv = batch["values"], batch["advantages"]
    assert col["value_first_token"][0] == pytest.approx(values[0, 2].item())
    assert col["advantage"][0] == pytest.approx(adv[0, 2].item())
    assert col["advantage_min"][0] == pytest.approx(adv[0, 2:5].min().item())
    assert col["advantage_max"][3] == pytest.approx(adv[3, 2:6].max().item())
    # Empty row: zeros rather than +-inf.
    assert (col["advantage_min"][4], col["value_first_token"][4]) == (0.0, 0.0)
    assert col["rollout_info"][1]["instance_id"] == "inst-0"
    assert col["trace_metadata"][1]["kind"] == "subagent"
    # Consumed by the write.
    assert diag.pop_rollout_debug(13) is None


def test_rollout_debug_rows_without_provenance():
    """Non-Gym rows (no tag) still get positional rollout ids."""
    diag = _diag()
    batch = _batch()
    batch["tags"] = [{"weight_version": 3}] * 6
    diag.on_advantage_stage(**batch)
    rows = diag.pop_rollout_debug(1)
    assert rows["trace_rollout_id"] == [0, 1, 2, 2, 3]
    assert rows["rollout_info"] == [{}] * 5


def test_failures_are_contained_outside_strict_mode(capsys):
    diag = _diag()
    diag.strict = False
    bad = _batch()
    bad["token_mask"] = torch.zeros(2, 2)  # wrong shape -> raises inside
    assert diag.on_advantage_stage(**bad) is None
    assert "legacy diagnostics on_advantage_stage failed" in capsys.readouterr().out
    assert diag.pop_rollout_debug(1) is None
    diag.strict = True
    with pytest.raises(Exception):
        diag.on_advantage_stage(**bad)


def test_rollout_dump_round_trip(tmp_path):
    class _Tok:
        def decode(self, ids, skip_special_tokens=False):
            return " ".join(str(i) for i in ids)

        def convert_ids_to_tokens(self, ids):
            return [f"t{i}" for i in ids]

    diag = _diag(log_rollout_dump=True, rollout_dump_period=5)
    batch = _batch()
    batch["input_ids"] = torch.arange(36).reshape(6, 6)
    batch["truncated"] = torch.zeros(6, dtype=torch.bool)
    diag.on_advantage_stage(**batch)
    path = diag.write_rollout_dump(
        str(tmp_path), step=5, tokenizer_factory=lambda: _Tok()
    )
    dump = torch.load(path, weights_only=False)
    assert dump["format_version"] == 2 and dump["step"] == 5
    assert dump["num_response_tokens"].tolist() == [3, 2, 2, 4, 0]
    n_tok = 11
    for key in ("token_ids", "values", "advantages", "returns", "prev_logprobs"):
        assert dump[key].shape[0] == n_tok, key
    assert dump["token_ids"][:3].tolist() == [2, 3, 4]
    assert dump["token_text"][:3] == ["t2", "t3", "t4"]
    assert dump["token_sample_index"].tolist()[:5] == [0, 0, 0, 1, 1]
    assert dump["token_response_position"].tolist()[:5] == [0, 1, 2, 0, 1]
    assert dump["prompt_group_index"].tolist() == [0, 0, 0, 1, 1]
    assert dump["generation_index"].tolist() == [0, 0, 1, 0, 1]
    assert dump["prompt_length"].tolist() == [2, 4, 2, 2, 1]
    assert dump["input_length"].tolist() == [10, 20, 30, 40, 1]
    assert dump["content"][0] == " ".join(str(i) for i in range(6))
    assert dump["adv_raw_std"] == 0.7
    assert dump["idx"] == [11] * 5
    assert "_row_input_ids" not in dump
    # Nothing packed on a non-dump step.
    assert (
        diag.write_rollout_dump(str(tmp_path), step=6, tokenizer_factory=_Tok) is None
    )


def test_efficiency_clock_starvation_and_summary():
    timer = Timer()
    clock = LegacyEfficiencyClock()
    clock.starved()
    clock.starved()  # the first poll opens the wait
    clock.fed(timer)
    clock.fed(timer)  # not starved: no entry
    clock.starved()
    clock.fed(timer)
    assert timer.reduce("init/total", "sum") >= 0.0
    assert len(timer._timers["init/total"]) == 1
    assert len(timer._timers["idle/buffer_starvation"]) == 1
    clock.record_pre_dispatch_wait(2.5)
    clock.record_pre_dispatch_wait(-1.0)  # clock skew never subtracts
    eff = clock.summary(timer, failed_trajectory_s=1.5)
    assert eff["efficiency/idle/generation_limit_pause_s"] == 2.5
    assert eff["efficiency/wasted/failed_trajectory_s"] == 1.5
    assert eff["efficiency/thread_seconds_total_s"] == 4.0
    assert eff["efficiency/idle/validation_s"] == 0.0
    assert len(eff) == 17


def test_first_batch_without_wait_marks_initial_fill_done():
    timer = Timer()
    clock = LegacyEfficiencyClock()
    clock.fed(timer)
    clock.starved()
    clock.fed(timer)
    assert "init/total" not in timer._timers
    assert len(timer._timers["idle/buffer_starvation"]) == 1


def test_legacy_efficiency_summary_math():
    eff = legacy_efficiency_summary(
        {
            "init/total": 10.0,
            "idle/buffer_starvation": 5.0,
            "idle/refit_bubble": 5.0,
            "idle/generation_limit_pause": 7.0,
        },
        100.0,
    )
    assert eff["efficiency/total_waste_s"] == 20.0
    assert eff["efficiency/productive_time_s"] == 80.0
    assert eff["efficiency/efficiency_pct"] == 80.0
    assert eff["efficiency/init/total_pct"] == 10.0
    assert eff["efficiency/idle/refit_bubble_pct"] == 5.0
    assert eff["efficiency/thread_seconds_total_s"] == 7.0
    assert eff["efficiency/total_wall_time_s"] == 100.0
    # Waste is clamped to the wall time.
    assert (
        legacy_efficiency_summary({"init/total": 500.0}, 100.0)[
            "efficiency/efficiency_pct"
        ]
        == 0.0
    )


def test_null_clock_is_inert():
    timer = Timer()
    NULL_EFFICIENCY_CLOCK.starved()
    NULL_EFFICIENCY_CLOCK.fed(timer)
    NULL_EFFICIENCY_CLOCK.record_pre_dispatch_wait(5.0)
    assert timer._timers == {} or not timer._timers
    assert NULL_EFFICIENCY_CLOCK.collector_seconds["idle/generation_limit_pause"] == 0


def test_timing_and_setup_aliases():
    tm = {
        "get_logprobs/shard_meta": 1.0,
        "get_logprobs/submit_futures": 2.0,
        "policy_training/shard_meta": 3.0,
        "policy_training/submit_microbatch_futures": 4.0,
        "value_training/shard_meta": 5.0,
        "value_training/submit_training_futures": 6.0,
    }
    out = add_legacy_timing_aliases(dict(tm))
    assert out["get_logprobs/shard_data"] == 1.0
    assert out["get_logprobs/submit_logprob_futures"] == 2.0
    assert out["policy_training/sharding_data"] == 3.0
    assert out["policy_training/submit_training_futures"] == 4.0
    assert out["value_training/sharding_data"] == 5.0
    assert out["value_training/submit_training_futures"] == 6.0
    assert out["get_logprobs/shard_meta"] == 1.0  # SC names stay
    setup = {"generation_init_time_s": 42.0, "policy_init_time_s": 3.0}
    assert legacy_setup_timing_metrics(setup, "vllm")["vllm_init_time_s"] == 42.0
    assert "vllm_init_time_s" not in legacy_setup_timing_metrics(setup, "megatron")


def test_fetch_generation_logger_metrics_reads_then_clears():
    class _Gen:
        def __init__(self):
            self.cleared = 0

        def get_logger_metrics(self):
            return {
                "inflight_batch_sizes": {0: [1, 2]},
                "num_pending_samples": {0: [0]},
            }

        def clear_logger_metrics(self):
            self.cleared += 1

    gen = _Gen()
    out = fetch_generation_logger_metrics(gen)
    assert out["generation_logger_metrics"]["inflight_batch_sizes"] == {0: [1, 2]}
    assert gen.cleared == 1

    class _NoLogger:  # GenerationInterface's defaults
        def get_logger_metrics(self):
            return {}

        def clear_logger_metrics(self):
            pass

    assert fetch_generation_logger_metrics(_NoLogger()) == {}

    class _Broken:
        def get_logger_metrics(self):
            raise RuntimeError("actor died")

    assert fetch_generation_logger_metrics(_Broken()) == {}


def test_timer_kwargs_only_for_callees_that_take_a_timer():
    from nemo_rl.algorithms.single_controller_utils.legacy_diagnostics import (
        timer_kwargs,
    )

    timer = Timer()

    def with_timer(meta, timer=None):
        return timer

    def with_kwargs(meta, **kwargs):
        return kwargs

    def without(meta):
        return meta

    assert timer_kwargs(with_timer, timer) == {"timer": timer}
    assert timer_kwargs(with_kwargs, timer) == {}
    assert timer_kwargs(without, timer) == {}
    assert timer_kwargs(len, timer) in ({}, {"timer": timer})  # builtins: no crash

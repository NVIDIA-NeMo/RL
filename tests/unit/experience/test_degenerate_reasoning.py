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
"""reward_penalties.penalize_degenerate_reasoning (nemo_rl/experience/degenerate_reasoning.py)."""

import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from nemo_rl.algorithms.grpo import RewardPenaltyConfig
from nemo_rl.data_plane.schema import MASK_SAMPLE
from nemo_rl.experience import degenerate_reasoning as dr
from nemo_rl.experience.mask_sample_rules import apply_mask_sample_rules, parse_mask_sample_rules
from nemo_rl.experience.rollouts import compute_reward_penalty_metrics, resolve_reward_penalty_config

CFG_ON = {"penalize_degenerate_reasoning": True}
SWE_AGENT = "swe_rebench_opencode_sandboxed_agent"
RULES = parse_mask_sample_rules(
    {
        "mask_sample_rules": [
            {"name": "eval_incomplete", "field": "evaluation_completed", "equals": False},
            {"name": "context_limited", "field": "opencode_context_limited", "equals": True},
        ]
    }
)


def _reasoning(text):
    return {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": text}]}


def _call():
    return {"type": "function_call", "name": "bash", "arguments": "{}", "call_id": "c"}


def _result(*texts, reward=1.0, ctx=False, evaluated=True):
    output = []
    for t in texts:
        output += [_reasoning(t), _call(), {"type": "function_call_output", "call_id": "c", "output": "ok"}]
    return {
        "response": {"output": output},
        "reward": reward,
        "opencode_context_limited": ctx,
        "evaluation_completed": evaluated,
    }


def test_reasoning_turns_split_on_tool_calls():
    output = [_reasoning("a"), _reasoning("b"), _call(), {"type": "function_call_output"}, _reasoning("c"),
              {"type": "message", "content": [{"type": "output_text", "text": "done"}]}]
    assert dr.reasoning_turns(output) == ["ab", "c"]
    assert dr.reasoning_turns(None) == []


def test_applies_to_gates():
    r = _result("a")
    assert not dr.applies_to({}, {"name": "x"}, r)
    assert dr.applies_to(CFG_ON, {"name": "x"}, r)
    allow = {**CFG_ON, "degenerate_reasoning_agents": ["swe_rebench_opencode_sandboxed_agent"]}
    assert dr.applies_to(allow, {"name": "swe_rebench_opencode_sandboxed_agent"}, r)
    assert not dr.applies_to(allow, {"name": "browsecomp_sandboxed_agent"}, r)
    assert not dr.applies_to(CFG_ON, {"name": "x"}, {**r, "is_empty_rollout": True})


def test_apply_verdict_flagged_and_unflagged():
    r = _result("a", reward=1.0)
    assert dr.apply_verdict(r, 0.97, True, CFG_ON) is True
    assert r["reward"] == 0.0 and r[dr.REWARD_BEFORE_KEY] == 1.0
    assert r[dr.FLAG_KEY] is True and r[dr.SCORE_KEY] == 0.97
    r = _result("a", reward=1.0)
    assert dr.apply_verdict(r, 0.1, False, CFG_ON) is False
    assert r["reward"] == 1.0 and r[dr.FLAG_KEY] is False and dr.REWARD_BEFORE_KEY not in r


def test_threshold_override():
    cfg = {**CFG_ON, "degenerate_reasoning_threshold": 0.5}
    r = _result("a")
    assert dr.apply_verdict(r, 0.6, False, cfg) is True and r["reward"] == 0.0
    r = _result("a")
    assert dr.apply_verdict(r, 0.4, True, cfg) is False and r["reward"] == 1.0
    r = _result("a")
    assert dr.apply_verdict(r, None, True, cfg) is False


def test_flagged_context_limited_rollout_is_not_masked_but_others_are():
    unmask = dr.unmask_rule_names(CFG_ON)
    flagged = _result("a", ctx=True)
    dr.apply_verdict(flagged, 0.99, True, CFG_ON)
    assert apply_mask_sample_rules(flagged, dr.rules_for_result(flagged, RULES, unmask)) == []
    assert not (flagged.get("instance_config") or {}).get(MASK_SAMPLE)
    clean = _result("a", ctx=True)
    dr.apply_verdict(clean, 0.1, False, CFG_ON)
    assert apply_mask_sample_rules(clean, dr.rules_for_result(clean, RULES, unmask)) == ["context_limited"]
    assert clean["instance_config"][MASK_SAMPLE] is True


def test_keeps_truncated_only_for_flagged():
    flagged = _result("a", ctx=True)
    dr.apply_verdict(flagged, 0.99, True, CFG_ON)
    clean = _result("a", ctx=True)
    dr.apply_verdict(clean, 0.1, False, CFG_ON)
    assert dr.keeps_truncated(flagged) is True
    assert dr.keeps_truncated(clean) is False
    assert dr.keeps_truncated(_result("a", ctx=True)) is False
    assert dr.keeps_truncated(None) is False


def test_infra_mask_rules_still_apply_to_flagged():
    unmask = dr.unmask_rule_names(CFG_ON)
    r = _result("a", ctx=True, evaluated=False)
    dr.apply_verdict(r, 0.99, True, CFG_ON)
    assert apply_mask_sample_rules(r, dr.rules_for_result(r, RULES, unmask)) == ["eval_incomplete"]


def test_check_result_scores_and_survives_detector_errors(monkeypatch):
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(dr, "_pool", lambda: pool)
    seen = {}

    def fake(turns):
        seen["turns"] = turns
        return 0.95, True

    monkeypatch.setattr(dr, "score_turns", fake)
    r = _result("x", "y", reward=1.0)
    asyncio.run(dr.check_result(r, {"name": "a"}, CFG_ON))
    assert seen["turns"] == ["x", "y"] and r["reward"] == 0.0 and r[dr.FLAG_KEY] is True

    def boom(turns):
        raise RuntimeError("bad")

    monkeypatch.setattr(dr, "score_turns", boom)
    r = _result("x", reward=1.0)
    asyncio.run(dr.check_result(r, {"name": "a"}, CFG_ON))
    assert r["reward"] == 1.0 and dr.FLAG_KEY not in r

    r = _result("x", reward=1.0)
    asyncio.run(dr.check_result(r, {"name": "a"}, {}))
    assert dr.FLAG_KEY not in r


def test_config_and_metric():
    cfg = RewardPenaltyConfig(penalize_degenerate_reasoning=True)
    assert cfg.degenerate_reasoning_unmask_rules == ["context_limited"]
    counts = {"duplicated_reasoning": 0, "empty_final_answer": 0, "unwanted_token": 0, "malformed_think_tag": 0,
              "degenerate_reasoning": 3}
    assert compute_reward_penalty_metrics(counts, 10, cfg) == {"degenerate_reasoning_rate": 0.3}


def test_resolved_config_keeps_the_penalty():
    """The single controller scores with resolve_reward_penalty_config's dict, so the penalty's keys must survive it."""
    cfg = RewardPenaltyConfig(
        penalize_degenerate_reasoning=True, degenerate_reasoning_threshold=0.8, degenerate_reasoning_agents=[SWE_AGENT]
    )
    resolved = resolve_reward_penalty_config(cfg, tokenizer=None)
    assert dr.enabled(resolved)
    assert dr.applies_to(resolved, {"name": SWE_AGENT}, {})
    assert not dr.applies_to(resolved, {"name": "browsecomp_sandboxed_agent"}, {})
    assert dr.unmask_rule_names(resolved) == frozenset({"context_limited"})
    assert resolved["degenerate_reasoning_threshold"] == 0.8
    counts = {"duplicated_reasoning": 0, "empty_final_answer": 0, "unwanted_token": 0, "malformed_think_tag": 0,
              "degenerate_reasoning": 0}
    assert compute_reward_penalty_metrics(counts, 10, resolved) == {"degenerate_reasoning_rate": 0.0}


def test_real_detector_flags_filler_not_prose():
    pytest.importorskip("nemo_rl.experience.loop_detector")
    filler = "OK hold on — hmm — hmm. " * 4000
    rng = random.Random(0)
    words = ("parser token lookahead buffer test fixture module config schema index cache request handler retry "
             "timeout socket path import cursor commit branch patch diff regression assert stack frame scope").split()
    prose = " ".join(" ".join(rng.choice(words) for _ in range(rng.randint(8, 20))) + "." for _ in range(1500))
    s_loop, f_loop = dr.score_turns(["Looking at the failing test first.", filler])
    s_prose, f_prose = dr.score_turns([prose, "Now run the tests."])
    assert f_loop is True and f_prose is False
    assert s_loop is not None and s_prose is not None and s_loop > s_prose

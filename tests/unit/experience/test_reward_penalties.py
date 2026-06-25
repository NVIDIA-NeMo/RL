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

import sys

import pytest
import torch

from nemo_rl.experience.rollouts import (
    apply_reward_penalties,
    resolve_reward_penalty_config,
)


# ---- Helpers to build minimal result dicts ----


def _make_result(
    reward=1.0,
    output_items=None,
    message_log=None,
):
    """Build a minimal result dict matching the structure from nemo_gym."""
    return {
        "full_result": {
            "reward": reward,
            "response": {"output": output_items or []},
        },
        "message_log": message_log or [],
    }


def _reasoning_item(text, generation_str=None):
    item = {"type": "reasoning", "summary": [{"text": text, "type": "summary_text"}]}
    if generation_str is not None:
        item["generation_str"] = generation_str
    return item


def _message_item(text, generation_str=None):
    item = {
        "type": "message",
        "content": [{"text": text, "type": "output_text"}],
        "role": "assistant",
    }
    if generation_str is not None:
        item["generation_str"] = generation_str
    return item


def _function_call_item(name="tool", generation_str=None):
    item = {"type": "function_call", "name": name, "arguments": "{}", "call_id": "c1"}
    if generation_str is not None:
        item["generation_str"] = generation_str
    return item


def _function_call_output_item(output="result"):
    return {"type": "function_call_output", "output": output, "call_id": "c1"}


def _msg(role, token_ids, **extra):
    msg = {"role": role, "token_ids": torch.tensor(token_ids)}
    msg.update(extra)
    return msg


class _FakeTokenizer:
    def __init__(self, eos_token_id=2, token_map=None):
        self.eos_token_id = eos_token_id
        self.token_map = token_map or {"<think>": [12], "</think>": [13]}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return self.token_map[text]


# =====================================================================
# Penalty 1: penalize_duplicated_reasoning
# =====================================================================


class TestPenalizeDuplicatedReasoning:
    CFG = {"penalize_duplicated_reasoning": True}

    def test_exact_match_penalized(self):
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("The answer is 42"),
                _message_item("The answer is 42"),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0
        assert counts["duplicated_reasoning"] == 1

    def test_different_text_not_penalized(self):
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("Let me think about this"),
                _message_item("The answer is 42"),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert counts["duplicated_reasoning"] == 0

    def test_whitespace_stripped_before_compare(self):
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("  hello world  "),
                _message_item("hello world"),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_empty_reasoning_not_penalized(self):
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item(""),
                _message_item(""),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_reasoning_followed_by_function_call_not_penalized(self):
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("I need to call a tool"),
                _function_call_item(),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_no_reasoning_item_not_penalized(self):
        result = _make_result(
            reward=1.0,
            output_items=[
                _message_item("The answer is 42"),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_disabled_by_default(self):
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("same"),
                _message_item("same"),
            ],
        )
        counts = apply_reward_penalties([result], {})
        assert result["full_result"]["reward"] == 1.0

    def test_multi_turn_first_pair_matches(self):
        """In multi-turn, if any reasoning/answer pair matches, penalize."""
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("duplicated"),
                _message_item("duplicated"),
                _function_call_item(),
                _function_call_output_item(),
                _reasoning_item("different thinking"),
                _message_item("final answer"),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0


# =====================================================================
# Penalty 2: penalize_empty_final_answer
# =====================================================================


class TestPenalizeEmptyFinalAnswer:
    CFG = {"penalize_empty_final_answer": True}

    def test_empty_content_penalized(self):
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("thinking"),
                _message_item(""),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0
        assert counts["empty_final_answer"] == 1

    def test_nonempty_content_not_penalized(self):
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("thinking"),
                _message_item("The answer is 42"),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_no_message_item_penalized(self):
        """Only reasoning + tool calls, no final message."""
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("thinking"),
                _function_call_item(),
                _function_call_output_item(),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_empty_output_items_penalized(self):
        result = _make_result(reward=1.0, output_items=[])
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_whitespace_only_penalized(self):
        result = _make_result(
            reward=1.0,
            output_items=[
                _message_item("   "),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_last_item_function_call_not_penalized(self):
        """Model ended mid-agentic-loop with a function_call — not an empty answer."""
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("thinking"),
                _function_call_item(),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert counts["empty_final_answer"] == 0

    def test_message_before_tool_call_uses_last_message(self):
        """Last content-bearing item is the function_call_output (no content field),
        but there's a message earlier. The reverse walk finds the message."""
        result = _make_result(
            reward=1.0,
            output_items=[
                _message_item("answer"),
                _function_call_item(),
                _function_call_output_item(),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        # function_call_output has no "content", function_call has no "content",
        # so reverse walk reaches the message item
        assert result["full_result"]["reward"] == 1.0


# =====================================================================
# Penalty 3: penalize_eos_token
# =====================================================================


class TestPenalizeEosToken:
    CFG = {"penalize_eos_token": True, "token_ids": {"eos": 2}}

    def test_eos_in_generation_penalized(self):
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 200]),
                _msg("assistant", [300, 2, 400]),  # token 2 = EOS
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0
        assert counts["eos_token"] == 1

    def test_no_eos_not_penalized(self):
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 200]),
                _msg("assistant", [300, 400, 500]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_terminal_eos_not_penalized(self):
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 200]),
                _msg("assistant", [300, 400, 2]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert counts["eos_token"] == 0

    def test_eos_in_user_not_penalized(self):
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 2, 200]),  # EOS in user prompt
                _msg("assistant", [300, 400]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_custom_eos_token_id(self):
        cfg = {"penalize_eos_token": True, "token_ids": {"eos": 99}}
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 99, 400]),
            ],
        )
        counts = apply_reward_penalties([result], cfg)
        assert result["full_result"]["reward"] == 0.0

    def test_multi_turn_terminal_eos_not_penalized(self):
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 2]),
                _msg("user", [500]),
                _msg("assistant", [600, 2]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert counts["eos_token"] == 0

    def test_multi_turn_internal_eos_penalized(self):
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 2]),
                _msg("user", [500]),
                _msg("assistant", [600, 2, 700]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0
        assert counts["eos_token"] == 1

    def test_empty_generation_not_penalized(self):
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", []),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_eos_token_id_must_be_explicit(self):
        with pytest.raises(ValueError, match="reward_penalties.token_ids.eos"):
            resolve_reward_penalty_config(
                {"penalize_eos_token": True}, _FakeTokenizer(eos_token_id=2)
            )

    def test_null_token_ids_requires_explicit_eos(self):
        with pytest.raises(ValueError, match="reward_penalties.token_ids.eos"):
            resolve_reward_penalty_config(
                {"penalize_eos_token": True, "token_ids": None},
                _FakeTokenizer(eos_token_id=2),
            )

    def test_missing_eos_direct_apply_raises(self):
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 2]),
            ],
        )
        with pytest.raises(ValueError, match="reward_penalties.token_ids.eos"):
            apply_reward_penalties([result], {"penalize_eos_token": True})


# =====================================================================
# Penalty 4: penalize_malformed_think_tag
# =====================================================================


class TestPenalizeMultiEndThink:
    CFG = {
        "penalize_malformed_think_tag": True,
        "token_ids": {"think_open": 12, "think_close": 13},
    }

    # --- 4a: Token ID checks ---

    def test_enable_thinking_true_valid(self):
        """enable_thinking=True: <think>(12) in user prompt, </think>(13) in gen."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12, 200]),  # 1x <think> in prompt
                _msg("assistant", [300, 13, 400]),  # 1x </think> in gen
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_enable_thinking_false_valid(self):
        """enable_thinking=False: <think>(12) and </think>(13) both in prompt."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12, 13, 200]),  # both in prompt
                _msg("assistant", [300, 400]),  # none in gen
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_missing_think_open_penalized(self):
        """No <think> token at all."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 200]),
                _msg("assistant", [300, 13, 400]),  # only </think>
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0
        assert counts["malformed_think_tag"] == 1

    def test_missing_think_close_penalized(self):
        """No </think> token at all."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12, 200]),
                _msg("assistant", [300, 400]),  # no </think>
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_double_think_close_penalized(self):
        """Two </think> tokens in generation."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12, 200]),
                _msg("assistant", [300, 13, 400, 13]),  # 2x </think>
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_double_think_open_penalized(self):
        """Two <think> tokens."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12, 12, 200]),  # 2x <think>
                _msg("assistant", [300, 13, 400]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_multi_turn_valid(self):
        """Two valid turns, each with exactly 1 <think> and 1 </think>."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400]),
                _msg("user", [500, 12]),
                _msg("assistant", [600, 13, 700]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_multi_turn_second_turn_invalid(self):
        """First turn ok, second turn has double </think>."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400]),
                _msg("user", [500, 12]),
                _msg("assistant", [600, 13, 13, 700]),  # 2x </think>
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_custom_token_ids(self):
        cfg = {
            "penalize_malformed_think_tag": True,
            "token_ids": {"think_open": 50, "think_close": 51},
        }
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 50]),
                _msg("assistant", [300, 51, 400]),
            ],
        )
        counts = apply_reward_penalties([result], cfg)
        assert result["full_result"]["reward"] == 1.0

    def test_think_token_ids_inferred_from_tokenizer(self):
        cfg = resolve_reward_penalty_config(
            {"penalize_malformed_think_tag": True}, _FakeTokenizer()
        )
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400]),
            ],
        )
        counts = apply_reward_penalties([result], cfg)
        assert result["full_result"]["reward"] == 1.0

    def test_custom_thinking_tags_used_for_inference_and_string_check(self):
        cfg = resolve_reward_penalty_config(
            {"penalize_malformed_think_tag": True},
            _FakeTokenizer(token_map={"<thinking>": [50], "</thinking>": [51]}),
            thinking_tags=["<thinking>", "</thinking>"],
        )
        valid_result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 50]),
                _msg("assistant", [300, 51, 400]),
            ],
        )
        counts = apply_reward_penalties([valid_result], cfg)
        assert valid_result["full_result"]["reward"] == 1.0
        assert counts["malformed_think_tag"] == 0

        leaked_result = _make_result(
            reward=1.0,
            output_items=[
                _message_item("answer", generation_str="leaked <thinking> hidden text")
            ],
            message_log=[
                _msg("user", [100, 50]),
                _msg("assistant", [300, 51, 400]),
            ],
        )
        counts = apply_reward_penalties([leaked_result], cfg)
        assert leaked_result["full_result"]["reward"] == 0.0
        assert counts["malformed_think_tag"] == 1

    def test_multitoken_think_tags_skip_token_count_fallback(self):
        cfg = resolve_reward_penalty_config(
            {"penalize_malformed_think_tag": True},
            _FakeTokenizer(token_map={"<think>": [12, 98], "</think>": [13, 99]}),
        )
        assert cfg is not None
        assert "token_ids" not in cfg
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 400]),
            ],
        )
        counts = apply_reward_penalties([result], cfg)
        assert result["full_result"]["reward"] == 1.0
        assert counts["malformed_think_tag"] == 0

    def test_multi_turn_history_think_tokens_valid(self):
        """Prompt has many think tokens from history; only the delta matters."""
        # Simulates a prompt with 5 <think> and 4 </think> from prior turns,
        # plus a trailing <think> for the current turn = 5 open, 4 close -> thinking_on
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg(
                    "user", [100, 12, 13, 12, 13, 12, 13, 12, 13, 12]
                ),  # 5 open, 4 close
                _msg("assistant", [300, 13, 400]),  # 0 open, 1 close
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_thinking_off_with_think_in_generation_penalized(self):
        """enable_thinking=False inferred (balanced), but model generates </think>."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12, 13]),  # 1 open, 1 close -> thinking_off
                _msg("assistant", [300, 13, 400]),  # unexpected </think>
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_unexpected_prompt_pattern_penalized(self):
        """More </think> than <think> in prompt — unexpected pattern."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 13, 13, 12]),  # 1 open, 2 close -> unexpected
                _msg("assistant", [300, 400]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_existing_has_malformed_thinking_flag_without_token_ids_penalized(self):
        cfg = {"penalize_malformed_think_tag": True}
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400], has_malformed_thinking=True),
            ],
        )
        counts = apply_reward_penalties([result], cfg)
        assert result["full_result"]["reward"] == 0.0
        assert counts["malformed_think_tag"] == 1

    # --- 4b: String checks ---

    def test_piecemeal_think_open_in_generation_penalized(self):
        """Model spells out <think> with regular tokens in generation_str."""
        result = _make_result(
            reward=1.0,
            output_items=[
                _message_item(
                    "answer", generation_str="some <think> text </think> answer"
                )
            ],
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400]),  # token IDs are fine
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_piecemeal_double_think_close_in_generation_penalized(self):
        """Model spells out </think> twice with regular tokens."""
        result = _make_result(
            reward=1.0,
            output_items=[
                _message_item("answer", generation_str="</think> text </think>")
            ],
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400]),  # token IDs are fine
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0

    def test_single_think_close_in_generation_str_ok(self):
        """One </think> in generation_str is normal for enable_thinking=True."""
        result = _make_result(
            reward=1.0,
            output_items=[
                _message_item("answer", generation_str="thinking </think> answer")
            ],
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0

    def test_no_generation_str_skipped(self):
        """Output items without generation_str are skipped (not penalized)."""
        result = _make_result(
            reward=1.0,
            output_items=[
                _function_call_output_item(),  # no generation_str
                _message_item("answer"),  # no generation_str
            ],
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0


# =====================================================================
# Cross-cutting: multiple penalties, config gating, batch behavior
# =====================================================================


class TestCrossCutting:
    def test_no_config_no_penalties(self):
        result = _make_result(reward=1.0)
        counts = apply_reward_penalties([result], None)
        assert result["full_result"]["reward"] == 1.0
        assert all(v == 0 for v in counts.values())

    def test_empty_config_no_penalties(self):
        result = _make_result(reward=1.0)
        counts = apply_reward_penalties([result], {})
        assert result["full_result"]["reward"] == 1.0

    def test_empty_results_no_crash(self):
        counts = apply_reward_penalties([], {"penalize_eos_token": True})
        assert all(v == 0 for v in counts.values())

    def test_multiple_penalties_stack(self):
        """A result that triggers both duplicated reasoning and EOS penalty."""
        cfg = {
            "penalize_duplicated_reasoning": True,
            "penalize_eos_token": True,
            "token_ids": {"eos": 2},
        }
        result = _make_result(
            reward=1.0,
            output_items=[
                _reasoning_item("same"),
                _message_item("same"),
            ],
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 2, 400]),  # EOS internal (not terminal)
            ],
        )
        counts = apply_reward_penalties([result], cfg)
        assert result["full_result"]["reward"] == 0.0
        assert counts["duplicated_reasoning"] == 1
        assert counts["eos_token"] == 1

    def test_batch_of_results_mixed(self):
        """Two results: first is fine, second has EOS."""
        cfg = {"penalize_eos_token": True, "token_ids": {"eos": 2}}
        r1 = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 400]),
            ],
        )
        r2 = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 2, 400]),  # EOS internal (not terminal)
            ],
        )
        counts = apply_reward_penalties([r1, r2], cfg)
        assert r1["full_result"]["reward"] == 1.0
        assert r2["full_result"]["reward"] == 0.0
        assert counts["eos_token"] == 1


if __name__ == "__main__":
    import traceback

    test_classes = [
        TestPenalizeDuplicatedReasoning,
        TestPenalizeEmptyFinalAnswer,
        TestPenalizeEosToken,
        TestPenalizeMultiEndThink,
        TestCrossCutting,
    ]

    passed = 0
    failed = 0
    for cls in test_classes:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(obj, name)()
                print(f"  PASS {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {cls.__name__}.{name}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)

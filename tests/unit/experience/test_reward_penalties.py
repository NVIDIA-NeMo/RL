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

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import (
    _calculate_gdpo_reward_feature_metrics,
    _extract_mask_sample_flags,
    _postprocess_single_nemo_gym_group,
    _record_gdpo_think_count_features,
    apply_reward_penalties,
    resolve_reward_penalty_config,
    should_mask_flagged_samples,
)
from nemo_rl.utils.timer import Timer

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
        self.pad_token_id = 0
        self.token_map = token_map or {"<think>": [12], "</think>": [13]}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return self.token_map[text]


class TestExtractMaskSampleFlags:
    def test_reads_mask_sample_from_instance_config(self):
        results = [
            {"full_result": {"instance_config": {"mask_sample": True}}},
            {"full_result": {"instance_config": {"mask_sample": False}}},
            {"full_result": {"instance_config": {}}},
            {"full_result": {}},
            {"full_result": {"instance_config": None}},
        ]

        mask_sample = _extract_mask_sample_flags(results)

        assert mask_sample.dtype == torch.bool
        assert torch.equal(
            mask_sample, torch.tensor([True, False, False, False, False])
        )


class TestShouldMaskFlaggedSamples:
    def test_reads_env_should_mask_flagged_samples(self):
        assert should_mask_flagged_samples({}) is True
        assert (
            should_mask_flagged_samples({"should_mask_flagged_samples": None}) is True
        )
        assert (
            should_mask_flagged_samples({"should_mask_flagged_samples": True}) is True
        )
        assert (
            should_mask_flagged_samples({"should_mask_flagged_samples": False}) is False
        )


class _FakeGeneration:
    cfg = {"max_total_sequence_length": 100}


def _gate_result(mask_sample):
    return {
        "full_result": {
            "reward": 1.0,
            "response": {"output": []},
            "instance_config": {"mask_sample": mask_sample},
        },
        "message_log": [
            {"role": "user", "token_ids": torch.tensor([3, 4])},
            {"role": "assistant", "token_ids": torch.tensor([1, 2])},
        ],
        "input_message_log": [{"role": "user", "token_ids": torch.tensor([3, 4])}],
    }


class TestMaskEnvFlaggedSamplesBatchedGate:
    """The batched NeMo-Gym postprocess gate at _postprocess_single_nemo_gym_group."""

    def _final_batch(self, mask_env_flagged_samples):
        results = [_gate_result(True), _gate_result(False)]
        nemo_gym_rows = [{"agent_ref": {"name": "agent"}} for _ in results]
        rollout_result = _postprocess_single_nemo_gym_group(
            nemo_gym_rows=nemo_gym_rows,
            results=results,
            timer=Timer(),
            timer_prefix="timing/test",
            policy_generation=_FakeGeneration(),
            input_batch=BatchedDataDict({"loss_multiplier": torch.ones(2)}),
            tokenizer=_FakeTokenizer(),
            log_full_result_tables=False,
            mask_env_flagged_samples=mask_env_flagged_samples,
        )
        return rollout_result.final_batch

    def test_gate_on_carries_mask_sample(self):
        final_batch = self._final_batch(True)
        assert torch.equal(final_batch["mask_sample"], torch.tensor([True, False]))

    def test_gate_off_omits_mask_sample(self):
        assert "mask_sample" not in self._final_batch(False)


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
# Penalty 3: penalize_unwanted_tokens
# =====================================================================


class TestPenalizeUnwantedTokens:
    CFG = {"penalize_unwanted_tokens": True, "token_ids": {"unwanted": [2]}}

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
        assert counts["unwanted_token"] == 1

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

    def test_terminal_eos_penalized(self):
        # A trailing EOS is penalized too — the whole sequence is checked.
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 200]),
                _msg("assistant", [300, 400, 2]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0
        assert counts["unwanted_token"] == 1

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
        cfg = {"penalize_unwanted_tokens": True, "token_ids": {"unwanted": [99]}}
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 99, 400]),
            ],
        )
        counts = apply_reward_penalties([result], cfg)
        assert result["full_result"]["reward"] == 0.0

    def test_multiple_unwanted_token_ids(self):
        cfg = {"penalize_unwanted_tokens": True, "token_ids": {"unwanted": [98, 99]}}
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 99, 400]),
            ],
        )
        counts = apply_reward_penalties([result], cfg)
        assert result["full_result"]["reward"] == 0.0
        assert counts["unwanted_token"] == 1

    def test_multi_turn_terminal_eos_penalized(self):
        # Trailing EOS in any assistant turn is penalized.
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
        assert result["full_result"]["reward"] == 0.0
        assert counts["unwanted_token"] == 1

    def test_multi_turn_internal_eos_penalized(self):
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 400]),
                _msg("user", [500]),
                _msg("assistant", [600, 2, 700]),
            ],
        )
        counts = apply_reward_penalties([result], self.CFG)
        assert result["full_result"]["reward"] == 0.0
        assert counts["unwanted_token"] == 1

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

    def test_unwanted_token_ids_must_be_explicit(self):
        with pytest.raises(ValueError, match="reward_penalties.token_ids.unwanted"):
            resolve_reward_penalty_config(
                {"penalize_unwanted_tokens": True}, _FakeTokenizer(eos_token_id=2)
            )

    def test_null_token_ids_requires_explicit_unwanted(self):
        with pytest.raises(ValueError, match="reward_penalties.token_ids.unwanted"):
            resolve_reward_penalty_config(
                {"penalize_unwanted_tokens": True, "token_ids": None},
                _FakeTokenizer(eos_token_id=2),
            )

    def test_empty_unwanted_list_requires_explicit_unwanted(self):
        with pytest.raises(ValueError, match="reward_penalties.token_ids.unwanted"):
            resolve_reward_penalty_config(
                {
                    "penalize_unwanted_tokens": True,
                    "token_ids": {"unwanted": []},
                },
                _FakeTokenizer(eos_token_id=2),
            )

    def test_missing_unwanted_direct_apply_raises(self):
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 2]),
            ],
        )
        with pytest.raises(ValueError, match="reward_penalties.token_ids.unwanted"):
            apply_reward_penalties([result], {"penalize_unwanted_tokens": True})


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
# GDPO feature: think_count_delta
# =====================================================================


class TestGdpoThinkCountFeature:
    CFG = {"token_ids": {"think_close": 13}}

    def test_one_close_think_has_zero_delta(self):
        result = _make_result(
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400]),
            ]
        )
        _record_gdpo_think_count_features([result], self.CFG)
        entry = result["full_result"]["gdpo_reward_features"]["think_count_delta"]
        assert entry["reward"] == 0.0
        assert entry["num_close_think_tags"] == 1

    def test_missing_close_think_has_negative_delta(self):
        result = _make_result(
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 400]),
            ]
        )
        _record_gdpo_think_count_features([result], self.CFG)
        entry = result["full_result"]["gdpo_reward_features"]["think_count_delta"]
        assert entry["reward"] == -1.0
        assert entry["num_close_think_tags"] == 0

    def test_double_close_think_has_negative_delta(self):
        result = _make_result(
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400, 13]),
            ]
        )
        _record_gdpo_think_count_features([result], self.CFG)
        entry = result["full_result"]["gdpo_reward_features"]["think_count_delta"]
        assert entry["reward"] == -1.0
        assert entry["num_close_think_tags"] == 2

    def test_generation_str_fallback_counts_close_thinks(self):
        result = _make_result(
            output_items=[
                _message_item("answer", generation_str="a </think> b </think>")
            ],
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 400]),
            ],
        )
        _record_gdpo_think_count_features([result], self.CFG)
        entry = result["full_result"]["gdpo_reward_features"]["think_count_delta"]
        assert entry["reward"] == -1.0
        assert entry["num_close_think_tags"] == 2

    def test_unconfigured_token_id_is_neutral_without_text(self):
        # No think_close token id configured, no tokenizer, no generation
        # text: counting is impossible, so the feature must stay neutral
        # (count=1 -> delta=0) instead of guessing a token id.
        result = _make_result(
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400, 13]),
            ]
        )
        _record_gdpo_think_count_features([result], {})
        entry = result["full_result"]["gdpo_reward_features"]["think_count_delta"]
        assert entry["reward"] == 0.0
        assert entry["count_source"] == "unavailable"

    def test_unconfigured_token_id_uses_generation_str(self):
        # Without a configured token id the generation string count is used
        # alone (token ids of other tokenizers must not leak into the count).
        result = _make_result(
            output_items=[
                _message_item("answer", generation_str="a </think> b")
            ],
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400, 13]),
            ],
        )
        _record_gdpo_think_count_features([result], {})
        entry = result["full_result"]["gdpo_reward_features"]["think_count_delta"]
        assert entry["reward"] == 0.0
        assert entry["num_close_think_tags"] == 1

    def test_generic_gdpo_feature_metrics(self):
        r1 = _make_result()
        r1["full_result"]["gdpo_reward_features"] = {
            "length_adjusted_reward": {"reward": 1.0},
            "think_count_delta": {"reward": 0.0},
        }
        r2 = _make_result()
        r2["full_result"]["gdpo_reward_features"] = {
            "length_adjusted_reward": {"reward": -0.5},
            "think_count_delta": {"reward": -2.0},
        }
        r3 = _make_result()
        r3["full_result"]["gdpo_reward_features"] = {
            "length_adjusted_reward": {"reward": 0.5},
        }

        metrics = _calculate_gdpo_reward_feature_metrics([r1, r2, r3])

        assert metrics["gdpo_length_adjusted_reward/min"] == -0.5
        assert metrics["gdpo_length_adjusted_reward/max"] == 1.0
        assert metrics["gdpo_length_adjusted_reward/mean"] == (1.0 - 0.5 + 0.5) / 3
        assert metrics["gdpo_think_count_delta/min"] == -2.0
        assert metrics["gdpo_think_count_delta/max"] == 0.0
        assert metrics["gdpo_think_count_delta/mean"] == -1.0


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
        counts = apply_reward_penalties([], {"penalize_unwanted_tokens": True})
        assert all(v == 0 for v in counts.values())

    def test_multiple_penalties_stack(self):
        """A result that triggers both duplicated reasoning and unwanted-token penalty."""
        cfg = {
            "penalize_duplicated_reasoning": True,
            "penalize_unwanted_tokens": True,
            "token_ids": {"unwanted": [2]},
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
        assert counts["unwanted_token"] == 1

    def test_batch_of_results_mixed(self):
        """Two results: first is fine, second has EOS."""
        cfg = {"penalize_unwanted_tokens": True, "token_ids": {"unwanted": [2]}}
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
        assert counts["unwanted_token"] == 1


if __name__ == "__main__":
    import traceback

    test_classes = [
        TestPenalizeDuplicatedReasoning,
        TestPenalizeEmptyFinalAnswer,
        TestPenalizeUnwantedTokens,
        TestPenalizeMultiEndThink,
        TestGdpoThinkCountFeature,
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

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

"""Unit tests for SDPO teacher-input construction helpers.

Focuses on the env-feedback teacher path added for the LCBv6 recipe — in
particular the helper that extracts the environment's textual feedback from
a rollout message log.
"""

from __future__ import annotations

import torch

from nemo_rl.algorithms.sdpo import (
    _extract_environment_output,
    align_teacher_topk,
    build_sdpo_teacher_data,
)


def test_extract_environment_output_picks_last_env_message():
    msg_log = [
        {"role": "user", "content": "solve this"},
        {"role": "assistant", "content": "```python\nprint(0)\n```"},
        {"role": "environment", "content": "Test 1/3 wrong answer\nEXPECTED: 42"},
    ]
    assert _extract_environment_output(msg_log) == (
        "Test 1/3 wrong answer\nEXPECTED: 42"
    )


def test_extract_environment_output_returns_none_when_missing():
    msg_log = [
        {"role": "user", "content": "solve this"},
        {"role": "assistant", "content": "```python\nprint(0)\n```"},
    ]
    assert _extract_environment_output(msg_log) is None


def test_extract_environment_output_returns_none_when_env_empty():
    msg_log = [
        {"role": "user", "content": "solve this"},
        {"role": "assistant", "content": "..."},
        {"role": "environment", "content": ""},
    ]
    assert _extract_environment_output(msg_log) is None


def test_extract_environment_output_picks_most_recent_env_turn():
    msg_log = [
        {"role": "user", "content": "solve this"},
        {"role": "assistant", "content": "v1"},
        {"role": "environment", "content": "first error"},
        {"role": "assistant", "content": "v2"},
        {"role": "environment", "content": "second error"},
    ]
    assert _extract_environment_output(msg_log) == "second error"


class _StubTokenizer:
    """Minimal tokenizer for build_sdpo_teacher_data unit tests.

    `apply_chat_template` returns deterministic token ids derived from the
    concatenated message contents; it also captures the last rendered
    teacher messages so tests can assert which template was used.
    """

    pad_token_id = 0

    def __init__(self) -> None:
        self.last_rendered: list[dict[str, str]] | None = None

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        return_dict=False,
        truncation=False,
        max_length=None,
        padding=False,
    ):
        self.last_rendered = [dict(m) for m in messages]
        text = "\n".join(m.get("content", "") for m in messages)
        ids = torch.tensor([ord(c) % 50000 + 1 for c in text], dtype=torch.long)
        if max_length is not None and truncation:
            ids = ids[:max_length]
        if return_dict:
            return {"input_ids": ids.unsqueeze(0)}
        return ids.unsqueeze(0)


def _make_msg_log(prompt: str, response: str, env_feedback: str | None = None):
    response_tokens = torch.tensor(
        [ord(c) % 50000 + 1 for c in response], dtype=torch.long
    )
    user_tokens = torch.tensor(
        [ord(c) % 50000 + 1 for c in prompt], dtype=torch.long
    )
    log = [
        {"role": "user", "content": prompt, "token_ids": user_tokens},
        {"role": "assistant", "content": response, "token_ids": response_tokens},
    ]
    if env_feedback is not None:
        env_tokens = torch.tensor(
            [ord(c) % 50000 + 1 for c in env_feedback], dtype=torch.long
        )
        log.append(
            {"role": "environment", "content": env_feedback, "token_ids": env_tokens}
        )
    return log


def test_combined_mode_signals_both_successes_and_failures():
    """Paper Table 2: combined mode distills on every rollout in a group.

    Group of 4: 2 successes + 2 failures. With at least one peer success,
    all 4 should get a teacher signal.
    """
    msg_logs = [
        _make_msg_log("solve this", "good code 1"),  # success
        _make_msg_log("solve this", "wrong code 1", env_feedback="wrong answer: X"),
        _make_msg_log("solve this", "good code 2"),  # success
        _make_msg_log("solve this", "wrong code 2", env_feedback="runtime error: Y"),
    ]
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

    tokenizer = _StubTokenizer()
    _, sdpo_mask = build_sdpo_teacher_data(
        message_logs=msg_logs,
        rewards=rewards,
        num_generations=4,
        tokenizer=tokenizer,
        success_reward_threshold=1.0,
        feedback_source="combined",
        dont_reprompt_on_self_success=True,
    )

    assert sdpo_mask.tolist() == [True, True, True, True]


def test_combined_mode_failed_sample_renders_peer_and_env():
    """Failed sample with a peer success and env feedback uses combined template."""
    msg_logs = [
        _make_msg_log("solve this", "PEER_GOOD"),  # success peer
        _make_msg_log("solve this", "BAD", env_feedback="ENV_ERROR_DETAIL"),  # failed
    ]
    rewards = torch.tensor([1.0, 0.0])

    tokenizer = _StubTokenizer()
    build_sdpo_teacher_data(
        message_logs=msg_logs,
        rewards=rewards,
        num_generations=2,
        tokenizer=tokenizer,
        success_reward_threshold=1.0,
        feedback_source="combined",
        dont_reprompt_on_self_success=True,
    )
    # The last call rendered the teacher for sample index 1 (the failed one).
    rendered = tokenizer.last_rendered[0]["content"]
    assert "PEER_GOOD" in rendered
    assert "ENV_ERROR_DETAIL" in rendered
    assert "Correct solution" in rendered
    assert "unsuccessful earlier attempt" in rendered


def test_combined_mode_falls_back_to_env_only_when_no_peer():
    """No successful peer + env feedback present -> falls back to env-only."""
    msg_logs = [
        _make_msg_log("solve this", "BAD1", env_feedback="ERR1"),
        _make_msg_log("solve this", "BAD2", env_feedback="ERR2"),
    ]
    rewards = torch.tensor([0.0, 0.0])

    tokenizer = _StubTokenizer()
    _, sdpo_mask = build_sdpo_teacher_data(
        message_logs=msg_logs,
        rewards=rewards,
        num_generations=2,
        tokenizer=tokenizer,
        success_reward_threshold=1.0,
        feedback_source="combined",
    )

    # Both failed samples should still get an env-only teacher signal.
    assert sdpo_mask.tolist() == [True, True]
    rendered = tokenizer.last_rendered[0]["content"]
    assert "ERR2" in rendered
    assert "Correct solution" not in rendered  # no peer demo available


def test_env_feedback_mode_skips_successes():
    """Regression: env_feedback mode only signals on failures."""
    msg_logs = [
        _make_msg_log("solve this", "GOOD"),  # success
        _make_msg_log("solve this", "BAD", env_feedback="ERR"),  # failure
    ]
    rewards = torch.tensor([1.0, 0.0])

    tokenizer = _StubTokenizer()
    _, sdpo_mask = build_sdpo_teacher_data(
        message_logs=msg_logs,
        rewards=rewards,
        num_generations=2,
        tokenizer=tokenizer,
        success_reward_threshold=1.0,
        feedback_source="env_feedback",
    )
    assert sdpo_mask.tolist() == [False, True]


# ── align_teacher_topk: off-by-one regression ────────────────────────────────


def test_align_teacher_topk_writes_at_predicting_logit_positions():
    """Teacher signal must land at the logit positions whose predictions are
    response tokens — NOT at the response token positions themselves.

    Causal LM: logits at position p predict token at p+1. So the prediction of
    the j-th response token (at position resp_start+j) is produced by logits
    at position resp_start+j-1. The loss masks via ``token_mask[:, 1:]`` which
    activates positions [resp_start-1, resp_end-2]; teacher signal must align
    there.

    Sequences:
        student_token_mask: [0, 0, 0, 1, 1, 1, 0]    response @ [3,4,5]
        teacher_token_mask: [0, 0, 0, 0, 0, 1, 1, 1, 0]   response @ [5,6,7]
    Expected:
        aligned_logits[2] = teacher_topk_logits[4]   (predicts response[0])
        aligned_logits[3] = teacher_topk_logits[5]   (predicts response[1])
        aligned_logits[4] = teacher_topk_logits[6]   (predicts response[2])
        aligned_logits[5] = 0   (would predict past response — wasted)
    """
    K = 4
    student_seq_len = 7
    teacher_seq_len = 9

    teacher_topk_logits = torch.arange(
        teacher_seq_len * K, dtype=torch.float32
    ).reshape(1, teacher_seq_len, K) + 1.0  # non-zero, distinguishable per position
    teacher_topk_indices = teacher_topk_logits.long()

    student_token_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 0]], dtype=torch.long)
    teacher_token_mask = torch.tensor(
        [[0, 0, 0, 0, 0, 1, 1, 1, 0]], dtype=torch.long
    )

    aligned_logits, aligned_indices = align_teacher_topk(
        teacher_topk_logits=teacher_topk_logits,
        teacher_topk_indices=teacher_topk_indices,
        teacher_token_mask=teacher_token_mask,
        student_seq_len=student_seq_len,
        student_token_mask=student_token_mask,
    )

    # Pre-response positions (last prompt token): teacher signal lands here.
    assert torch.equal(aligned_logits[0, 2], teacher_topk_logits[0, 4]), (
        "First response token's prediction (student pos 2) must align to "
        "teacher's last-prompt logits (teacher pos 4)"
    )
    assert torch.equal(aligned_logits[0, 3], teacher_topk_logits[0, 5])
    assert torch.equal(aligned_logits[0, 4], teacher_topk_logits[0, 6])

    # Position 5 in student is the last response token — its prediction (of
    # what comes AFTER response) is not used in the loss (masked out by
    # shifted_mask), so we don't write there.
    assert torch.all(aligned_logits[0, 5] == 0), (
        "Last response token position should not be filled (predicts past response)"
    )
    # Prompt positions before the prediction position also stay zero.
    assert torch.all(aligned_logits[0, 0] == 0)
    assert torch.all(aligned_logits[0, 1] == 0)
    # Indices should match the same alignment.
    assert torch.equal(aligned_indices[0, 2], teacher_topk_indices[0, 4])
    assert torch.equal(aligned_indices[0, 3], teacher_topk_indices[0, 5])
    assert torch.equal(aligned_indices[0, 4], teacher_topk_indices[0, 6])


def test_align_teacher_topk_first_response_token_signal_nonzero():
    """Regression for the original bug: aligned_logits at the FIRST response
    token's prediction position must NOT be zero.

    This is the position the loss includes via shifted_mask=1 but the previous
    implementation left as zero (because it wrote at student_resp_pos rather
    than student_resp_pos - 1)."""
    K = 4
    student_seq_len = 6
    teacher_seq_len = 8

    teacher_topk_logits = torch.randn(1, teacher_seq_len, K)
    teacher_topk_indices = torch.randint(0, 100, (1, teacher_seq_len, K))

    student_token_mask = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.long)
    teacher_token_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0]], dtype=torch.long)

    aligned_logits, _ = align_teacher_topk(
        teacher_topk_logits=teacher_topk_logits,
        teacher_topk_indices=teacher_topk_indices,
        teacher_token_mask=teacher_token_mask,
        student_seq_len=student_seq_len,
        student_token_mask=student_token_mask,
    )

    # student_resp_pos[0] = 2; predicting position is 2-1 = 1.
    # Before fix: aligned_logits[0, 1] = 0 (BUG).
    # After fix:  aligned_logits[0, 1] = teacher_topk_logits[0, 3] (≠ 0 in general).
    assert not torch.all(aligned_logits[0, 1] == 0), (
        "First response token's prediction position must be filled with teacher signal"
    )

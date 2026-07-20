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
"""Unit tests for the privileged (answer-conditioned) critic helpers."""

import pytest
import torch

from nemo_rl.algorithms.privileged_critic import (
    build_privileged_value_inputs,
    remap_by_response_mask,
)


# --------------------------------------------------------------------------- #
# remap_by_response_mask: the exact scatter/gather between the augmented and
# original layouts. This is the one piece of index logic that must be correct.
# --------------------------------------------------------------------------- #
def test_remap_moves_response_values_and_zeros_elsewhere():
    B, S_dst, S_src = 2, 7, 9
    dst_mask = torch.zeros(B, S_dst)
    dst_mask[0, 3:6] = 1  # response len 3
    dst_mask[1, 4:6] = 1  # response len 2
    # augmented layout: response pushed right by the (longer) answer-augmented prompt
    src_mask = torch.zeros(B, S_src)
    src_mask[0, 6:9] = 1  # response len 3
    src_mask[1, 7:9] = 1  # response len 2
    src = torch.zeros(B, S_src)
    src[0, 6:9] = torch.tensor([1.0, 2.0, 3.0])
    src[1, 7:9] = torch.tensor([4.0, 5.0])

    dst = remap_by_response_mask(src, src_mask, dst_mask)

    assert dst.shape == (B, S_dst)
    assert torch.equal(dst[0, 3:6], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(dst[1, 4:6], torch.tensor([4.0, 5.0]))
    # everything outside the response mask is exactly zero (GAE masks these anyway)
    assert dst[dst_mask == 0].abs().sum() == 0


def test_remap_is_invertible_roundtrip():
    B, S_dst, S_src = 3, 6, 8
    dst_mask = torch.zeros(B, S_dst)
    src_mask = torch.zeros(B, S_src)
    for b, (a, n) in enumerate([(2, 3), (1, 2), (3, 2)]):
        dst_mask[b, a : a + n] = 1
        src_mask[b, S_src - n : S_src] = 1  # response is the tail in the aug layout
    src = torch.randn(B, S_src) * src_mask

    orig = remap_by_response_mask(src, src_mask, dst_mask)
    back = remap_by_response_mask(orig, dst_mask, src_mask)
    assert torch.equal(back * src_mask, src * src_mask)


def test_remap_rejects_mismatched_response_counts():
    # A per-row count mismatch means the response tokens were NOT preserved verbatim.
    dst_mask = torch.zeros(1, 5)
    dst_mask[0, 2:5] = 1  # count 3
    src_mask = torch.zeros(1, 5)
    src_mask[0, 3:5] = 1  # count 2
    with pytest.raises(AssertionError, match="response-token counts differ"):
        remap_by_response_mask(torch.zeros(1, 5), src_mask, dst_mask)


# --------------------------------------------------------------------------- #
# build_privileged_value_inputs: the careful message-level prompt construction.
# The invariant that matters most: the RESPONSE tokens are byte-identical to the
# generated tokens, and the answer only lengthens the PROMPT region.
# --------------------------------------------------------------------------- #
class _FakeTokenizer:
    """Mirrors the HF pattern the code uses: apply_chat_template(tokenize=False)
    renders to a STRING, then __call__ tokenizes it. Prompt ids land in [200,249]
    so they never collide with the test response/env ids (50..91)."""

    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 50 + 200 for c in text]

    def decode(self, ids):
        return "".join(chr((int(i) - 200) % 50 + 65) for i in ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    ):
        s = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            s += "<assistant>"
        return s

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([self.encode(text)], dtype=torch.long)}


def _make_repeated_batch(response_ids, gold):
    msgs = [
        {"role": "user", "content": "What is 2+2?", "token_ids": torch.tensor([50, 51, 52])},
        {"role": "assistant", "content": "resp", "token_ids": torch.tensor(response_ids)},
    ]
    return {"message_log": [msgs], "extra_env_info": [{"ground_truth": gold}]}


def test_build_preserves_response_tokens_verbatim():
    tok = _FakeTokenizer()
    response_ids = [77, 78, 79, 80]
    batch = _make_repeated_batch(response_ids, gold="42")
    pcfg = {"enabled": True, "placement": "user_suffix", "max_answer_tokens": 256}

    critic = build_privileged_value_inputs(batch, tok, pcfg)

    ids = critic["input_ids"][0]
    mask = critic["token_mask"][0].bool()
    # response region == the verbatim generated tokens, in order
    assert ids[mask].tolist() == response_ids
    # response count matches
    assert int(mask.sum()) == len(response_ids)
    # answer lives strictly in the prompt (mask==0) region
    assert mask[: -len(response_ids)].sum() == 0


def test_build_drops_trailing_environment_turn():
    # The math rollout's message_log is [user, assistant, environment-feedback].
    # The trailing environment turn is masked and comes AFTER the response, so it must
    # be dropped from the critic input (not rejected, not included).
    tok = _FakeTokenizer()
    response_ids = [77, 78, 79]
    msgs = [
        {"role": "user", "content": "What is 2+2?", "token_ids": torch.tensor([50, 51, 52])},
        {"role": "assistant", "content": "resp", "token_ids": torch.tensor(response_ids)},
        {"role": "environment", "content": "feedback", "token_ids": torch.tensor([90, 91])},
    ]
    batch = {"message_log": [msgs], "extra_env_info": [{"ground_truth": "42"}]}
    critic = build_privileged_value_inputs(batch, tok, {"enabled": True})

    ids = critic["input_ids"][0]
    mask = critic["token_mask"][0].bool()
    assert ids[mask].tolist() == response_ids        # response verbatim
    assert int(mask.sum()) == len(response_ids)       # count matches train_data's mask
    assert 90 not in ids.tolist() and 91 not in ids.tolist()  # env turn dropped entirely


def test_build_answer_lengthens_only_the_prompt():
    tok = _FakeTokenizer()
    response_ids = [77, 78, 79, 80]
    pcfg = {"enabled": True, "placement": "user_suffix", "max_answer_tokens": 256}

    with_answer = build_privileged_value_inputs(
        _make_repeated_batch(response_ids, gold="the answer is 42 with reasoning"),
        tok,
        pcfg,
    )
    no_answer = build_privileged_value_inputs(
        _make_repeated_batch(response_ids, gold=""), tok, pcfg
    )
    # a longer reference answer only grows the prompt; response count is unchanged
    assert with_answer["input_lengths"][0] > no_answer["input_lengths"][0]
    assert int(with_answer["token_mask"][0].sum()) == len(response_ids)
    assert int(no_answer["token_mask"][0].sum()) == len(response_ids)


def test_build_truncates_long_answer():
    tok = _FakeTokenizer()
    response_ids = [77, 78]
    long_gold = "x" * 500
    short = build_privileged_value_inputs(
        _make_repeated_batch(response_ids, long_gold),
        tok,
        {"enabled": True, "max_answer_tokens": 8},
    )
    untrunc = build_privileged_value_inputs(
        _make_repeated_batch(response_ids, long_gold),
        tok,
        {"enabled": True, "max_answer_tokens": 500},
    )
    assert short["input_lengths"][0] < untrunc["input_lengths"][0]

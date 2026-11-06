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

import pytest
from speechlm_gspo.reward import asr_reward, normalize_asr_tokens


@pytest.mark.parametrize(
    "reference,hypothesis,reward",
    [
        ("", "", 1),
        ("", "a b c", -2),
        ("spoken words", "", 0),
        ("a b", "a b c", 0.5),
        ("नमस्ते", "नमस्ते", 1),
        ("a", "a b c d", -2),
        ("<spk:2>", "spk 2", 1),
    ],
)
def test_wer_contract(reference, hypothesis, reward):
    assert asr_reward(reference, hypothesis, {}) == reward


def test_unicode_marks_and_no_nfkc():
    assert normalize_asr_tokens("नमस्ते दुनिया") == ["नमस्ते", "दुनिया"]
    assert normalize_asr_tokens("Ａ") != normalize_asr_tokens("A")


MANDARIN_CER = {
    "language": "cmn",
    "task": "ordinary_asr",
    "reward_metric": "cer",
    "cer_whitespace_mode": "han_adjacent",
}


def test_mandarin_routing():
    with pytest.raises(ValueError):
        asr_reward("你好", "你好", {**MANDARIN_CER, "reward_metric": "wer"})
    with pytest.raises(ValueError):
        asr_reward("你好", "你好", {**MANDARIN_CER, "task": "sot"})


def test_mandarin_cer():
    pytest.importorskip("nemo_skills.evaluation.evaluator.audio")
    assert asr_reward("今 天 use AI", "今天use AI", MANDARIN_CER) == 1
    with pytest.raises(ValueError):
        asr_reward("", "你好", MANDARIN_CER)

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


@pytest.mark.parametrize(
    "reference,hypothesis,reward",
    [
        ("今 天 use AI", "今天use AI", 1),
        ("我 love you", "我loveyou", 8 / 9),
        ("你好，世界！", "你好 世界", 1),
        ("我有 twenty one 个", "我有21个", 1),
        ("colour 和 color", "color和color", 1),
        ("１２３个", "123个", 1),
        ("\U00020000 字", "\U00020000字", 1),
        ("你好", "", 0),
    ],
)
def test_mandarin_cer(reference, hypothesis, reward):
    pytest.importorskip("whisper_normalizer")
    assert asr_reward(reference, hypothesis, MANDARIN_CER) == pytest.approx(reward)


def test_mandarin_cer_rejects_empty_reference():
    pytest.importorskip("whisper_normalizer")
    with pytest.raises(ValueError, match="nonempty"):
        asr_reward("，", "你好", MANDARIN_CER)


def test_ascend_mer():
    pytest.importorskip("nemo.collections.speechlm2.parts.metrics.mer")
    metadata = {"reward_metric": "mer"}
    assert asr_reward("你好 world", "你好 world", metadata) == 1
    assert asr_reward("你好 world", "你好 word", metadata) == pytest.approx(2 / 3)
    assert asr_reward("你好 world", "world", metadata) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="nonempty"):
        asr_reward("", "你好", metadata)

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

"""Fractional transcript rewards, separate from benchmark normalization."""

import functools
import re
import unicodedata
from typing import Any


def normalize_asr_tokens(text: str) -> list[str]:
    """Casefold and split punctuation while preserving Indic combining marks."""
    cleaned = "".join(
        c
        if c.isalnum() or c in "'_" or unicodedata.category(c) in ("Mn", "Mc")
        else " "
        for c in text.casefold()
    )
    return cleaned.split()


def asr_reward(reference: str, hypothesis: str, metadata: dict[str, Any]) -> float:
    """Return unbounded ``1-error_rate`` with metadata-driven Mandarin CER."""
    language = str(metadata.get("language") or metadata.get("lang") or "").lower()
    metric = metadata.get("reward_metric") or "wer"
    if metric == "mer":
        # Match Speech's ASCEND evaluation: Chinese characters / English words.
        from nemo.collections.speechlm2.parts.metrics.mer import mer_tokens

        ref, hyp = mer_tokens(reference).split(), mer_tokens(hypothesis).split()
        if not ref:
            raise ValueError("MER requires a nonempty normalized reference")
        error = _edit_distance(ref, hyp) / len(ref)
    elif (
        language in ("zh", "cmn", "cmn_hans_cn", "zh-cn", "zh_cn")
        and metadata.get("task") == "ordinary_asr"
    ):
        if metric != "cer" or metadata.get("cer_whitespace_mode") != "han_adjacent":
            raise ValueError(
                "Mandarin ordinary_asr requires cer with han_adjacent whitespace"
            )
        error = _mandarin_cer(reference, hypothesis)
    else:
        if metric != "wer":
            raise ValueError(
                f"Unsupported reward metric {metric!r} for {language!r}/{metadata.get('task')!r}"
            )
        ref, hyp = normalize_asr_tokens(reference), normalize_asr_tokens(hypothesis)
        error = _edit_distance(ref, hyp) / len(ref) if ref else len(hyp)
    return 1.0 - error


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    row = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, 1):
        diagonal, row[0] = row[0], i
        for j, hyp_token in enumerate(hyp, 1):
            diagonal, row[j] = (
                row[j],
                min(row[j] + 1, row[j - 1] + 1, diagonal + (ref_token != hyp_token)),
            )
    return row[-1]


# Han ideographs (CJK unified, extensions and compatibility blocks) and U+3007.
_HAN = "\u3007" + "".join(
    f"{chr(start)}-{chr(end)}"
    for start, end in (
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x2EBF0, 0x2EE5F),
        (0x2F800, 0x2FA1F),
        (0x30000, 0x3134F),
        (0x31350, 0x323AF),
    )
)
_HAN_ADJACENT_SPACE = re.compile(f"(?<=[{_HAN}]) | (?=[{_HAN}])")


def _mandarin_cer(reference: str, hypothesis: str) -> float:
    """Character error rate that ignores whitespace next to Han characters.

    Text is lowercased and Whisper-normalized first; spaces between Latin
    words remain characters.
    """
    ref, hyp = _cer_text(reference), _cer_text(hypothesis)
    if not ref:
        raise ValueError("Mandarin CER requires a nonempty normalized reference")
    return _edit_distance(list(ref), list(hyp)) / len(ref)


def _cer_text(text: str) -> str:
    text = re.sub(r"\s+", " ", _english_normalizer()(text.lower())).strip()
    return _HAN_ADJACENT_SPACE.sub("", text)


@functools.cache
def _english_normalizer():
    # Installed only in the policy environment, which runs the ASR reward.
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()

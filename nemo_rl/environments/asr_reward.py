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

import math
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
    if (
        language in ("zh", "cmn", "cmn_hans_cn", "zh-cn", "zh_cn")
        and metadata.get("task") == "ordinary_asr"
    ):
        if metric != "cer" or metadata.get("cer_whitespace_mode") != "han_adjacent":
            raise ValueError(
                "Mandarin ordinary_asr requires cer with han_adjacent whitespace"
            )
        # Skills is optional for recipes that do not use Mandarin CER.
        from nemo_skills.evaluation.evaluator.audio import evaluate_cer

        error = evaluate_cer(
            reference,
            hypothesis,
            normalization_mode="standard",
            normalize_compound=False,
            whitespace_mode="han_adjacent",
            lang="cmn",
            remove_diacritics=True,
        )["cer"]
        if error is None or not math.isfinite(error):
            raise ValueError(
                "Mandarin CER requires a nonempty normalized reference and finite error"
            )
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

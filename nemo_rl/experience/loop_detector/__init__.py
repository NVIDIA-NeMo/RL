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
"""Degenerate-looping detector for thinking turns (OpenCode SWE / TB training rollouts). Stdlib only.

    from nemo_rl.experience.loop_detector import score_turn, score_rollout
    p = score_turn(text)                                   # P(loop) for one thinking turn, None if < 5,000 chars
    score, flag = score_rollout(list_of_turn_texts)        # detector A, penalty (high-precision) threshold
    score, flag = score_rollout(turns, mode="monitor")     # lower threshold for monitoring
    score, flag = score_rollout(turns, detector="B")       # independent cross-check

A turn is the reasoning text the model writes before one tool call. Turns shorter than 5,000 characters are not scored.
Rollout score = the maximum turn probability over its scored turns (None if no turn is scored); a rollout is flagged
when that maximum reaches the rollout threshold of the chosen mode, i.e. when any turn is flagged.

Detector A: logistic regression on 5 features (features.py, detector A family):
  - a_sym_rep8 / a_sym_k5: 8-gram repetition and high-multiplicity (5+) repetition of masked prose;
  - a_raw_k3: 8-grams of raw text seen 3+ times;
  - a_filler: filler-word rate;
  - a_log_len: turn length.
Detector B: logistic regression on compression and lexical-diversity features (no n-gram or filler counts).
Model files: model_A.json, model_B.json. Thresholds: thresholds.json. Fitted on labelled turns from the 4-env Super
hybrid runs; at the penalty threshold (0.8) test precision is ~0.97 and recall ~0.78 per rollout. It misses
deliberation loops that do not repeat text (re-deciding with varied wording).
"""

import json
import math
import os

from nemo_rl.experience.loop_detector.features import features_turn

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    with open(os.path.join(_DIR, name)) as f:
        return json.load(f)


_M = {k: _load(f"model_{k}.json") for k in ("A", "B")}
THRESHOLDS = _load("thresholds.json")
MIN_CHARS = 5000


def _prob(m, f):
    z = m["b"] + sum(w * (f[c] - mu) / sd for c, w, mu, sd in zip(m["feats"], m["w"], m["mu"], m["sd"]))
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))


def score_turn(text, detector="A"):
    """P(loop) for one thinking turn, or None when the turn has fewer than 5,000 characters."""
    if not text or len(text) < MIN_CHARS:
        return None
    return _prob(_M[detector], features_turn(text, detector))


def turn_flag(text, detector="A", mode="penalty"):
    p = score_turn(text, detector)
    return p, p is not None and p >= THRESHOLDS[detector]["turn"][mode]


def score_rollout(turns, detector="A", mode="penalty"):
    """-> (max turn P(loop) or None, flag). mode: 'penalty' (high precision) or 'monitor'."""
    ps = [p for p in (score_turn(t, detector) for t in turns) if p is not None]
    if not ps:
        return None, False
    s = max(ps)
    return s, s >= THRESHOLDS[detector]["rollout"][mode]

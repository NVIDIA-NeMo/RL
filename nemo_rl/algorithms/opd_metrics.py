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
"""[CUSTOM] Per-teacher / per-agent MOPD diagnostics.

Shared by OPDAdvantageEstimator (online, logged every step) and scripts/mopd_metrics.py (offline, from the
train_data_step*.jsonl dumps), so both report identical keys. With A_t = log pi_T(y_t) - log pi_S(y_t) on the sampled
token: k1 = -A and k3 = exp(A) - 1 - A are reverse-KL(S||T) estimators; -mean(log pi_gen) estimates student entropy.
budget_share = the group's share of sum|A| over loss tokens (= tok_share * abs_adv, normalised across teachers): with a
token-mean loss this is the share of the update each teacher actually drives (Open-MOPD, arXiv:2608.19098, B_d).
"""

import os
from typing import Optional, Sequence

import torch

PREFIX = "on_policy_distillation"
POS_EDGES = (0, 1024, 4096, 16384, 65536)
POS_NAMES = ("0-1k", "1k-4k", "4k-16k", "16k-64k", "64k+")
SMALL_ADV, NEG_ADV, BIG_ADV = 0.01, -1.0, 5.0
CORRECT_REWARD = 0.5  # verifier reward >= this counts as a correct rollout
MAX_QUANTILE_TOKENS = 1_000_000


def teacher_labels(teacher_model_by_agent_name: dict[str, str]) -> dict[str, str]:
    """Agent -> short teacher label: the first path component after the teachers' common root.

    e.g. .../domyn-edge-teachers/{math/s200, if-fc/step_340, branded} -> math, if-fc, branded.
    """
    paths = {
        a: os.path.normpath(p).split(os.sep)
        for a, p in teacher_model_by_agent_name.items()
    }
    if len({tuple(p) for p in paths.values()}) < 2:
        return {a: p[-1] for a, p in paths.items()}
    common = os.path.commonprefix(list(paths.values()))
    return {
        a: (p[len(common)] if len(p) > len(common) else p[-1]) for a, p in paths.items()
    }


def compute_opd_metrics(
    *,
    teacher_lp: torch.Tensor,
    student_lp: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    teachers: Sequence[str],
    rewards: Optional[torch.Tensor] = None,
    truncated: Optional[torch.Tensor] = None,
    gen_lp: Optional[torch.Tensor] = None,
    agents: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    """Metrics per teacher label (and "all"); `agents` adds a lighter per-agent set.

    Args:
        teacher_lp, student_lp, gen_lp: [B, S] log-probs of the sampled tokens (student = prev_logprobs,
            gen = inference engine).
        token_mask: [B, S] generated-token mask; sample_mask: [B] loss multiplier (0 = overlong-filtered).
        teachers / agents: B labels. rewards, truncated: [B] (raw verifier reward, hit max tokens).

    KL / advantage / entropy metrics use loss tokens (token_mask * sample_mask); length and truncation use all samples.
    """
    token_mask = token_mask.bool()
    sample_mask = sample_mask.reshape(-1).to(token_mask.device)
    loss_mask = token_mask & (sample_mask > 0).unsqueeze(1)
    rows, cols = loss_mask.nonzero(as_tuple=True)
    pos = (token_mask.cumsum(dim=1) - 1)[rows, cols]
    adv = (teacher_lp[rows, cols] - student_lp[rows, cols]).double()
    tok = {
        "adv": adv,
        "row": rows,
        "bucket": torch.bucketize(
            pos, torch.tensor(POS_EDGES[1:], device=pos.device), right=True
        ),
        "teacher_lp": teacher_lp[rows, cols].double(),
        "gen_lp": gen_lp[rows, cols].double() if gen_lp is not None else None,
    }
    gen_len = token_mask.sum(dim=1).double()
    correct = (
        rewards.reshape(-1).to(gen_len.device) >= CORRECT_REWARD
        if rewards is not None
        else None
    )
    trunc = (
        truncated.reshape(-1).to(gen_len.device).bool()
        if truncated is not None
        else None
    )
    n_tok_total = max(len(adv), 1)
    abs_total = float(adv.abs().sum()) or 1.0

    def row_select(names: Sequence[str], value: str) -> torch.Tensor:
        return torch.tensor([n == value for n in names], device=gen_len.device)

    out: dict[str, float] = {}
    groups = [("all", torch.ones_like(gen_len, dtype=torch.bool))]
    groups += [(t, row_select(teachers, t)) for t in sorted(set(teachers))]
    for name, sel_rows in groups:
        m = _group_metrics(
            tok, sel_rows, gen_len, correct, trunc, n_tok_total, abs_total, full=True
        )
        out.update({f"{PREFIX}/{name}/{k}": v for k, v in m.items()})
    for agent in sorted(set(agents or [])):
        m = _group_metrics(
            tok,
            row_select(agents, agent),
            gen_len,
            correct,
            trunc,
            n_tok_total,
            abs_total,
            full=False,
        )
        out.update({f"{PREFIX}/agent/{agent}/{k}": v for k, v in m.items()})
    return out


def _group_metrics(
    tok, sel_rows, gen_len, correct, trunc, n_tok_total, abs_total, full: bool
) -> dict[str, float]:
    m = {"n_samples": float(sel_rows.sum()), "gen_len_mean": _mean(gen_len[sel_rows])}
    if trunc is not None:
        m["trunc_rate"] = _mean(trunc[sel_rows].double())
    sel = sel_rows[tok["row"]]
    adv = tok["adv"][sel]
    if len(adv) == 0:
        return m
    m["k1_tok"] = float(-adv.mean())
    if not full:
        return m

    row = tok["row"][sel]
    n_rows = len(gen_len)
    tok_per_row = torch.zeros(
        n_rows, dtype=torch.float64, device=adv.device
    ).index_add_(0, row, torch.ones_like(adv))
    k1_per_row = torch.zeros_like(tok_per_row).index_add_(0, row, -adv)
    has_tok = tok_per_row > 0
    m.update(
        k1_seq=float((k1_per_row[has_tok] / tok_per_row[has_tok]).mean()),
        k3_tok=float(torch.expm1(adv).sub(adv).mean()),
        abs_adv=float(adv.abs().mean()),
        frac_tok_small=float((adv.abs() < SMALL_ADV).double().mean()),
        frac_tok_neg=float((adv < NEG_ADV).double().mean()),
        frac_tok_big=float((adv.abs() > BIG_ADV).double().mean()),
        tok_share=len(adv) / n_tok_total,
        budget_share=float(adv.abs().sum()) / abs_total,
        teacher_xent=float(-tok["teacher_lp"][sel].mean()),
        gen_len_p90=float(torch.quantile(gen_len[sel_rows], 0.9)),
    )
    sample = adv[:: max(1, len(adv) // MAX_QUANTILE_TOKENS + 1)].float()
    m["adv_p01"], m["adv_p99"] = (
        float(q)
        for q in torch.quantile(
            sample, torch.tensor([0.01, 0.99], device=sample.device)
        )
    )
    if tok["gen_lp"] is not None:
        m["student_nll"] = float(-tok["gen_lp"][sel].mean())

    if correct is not None:
        sel_correct = correct[sel_rows]
        m.update(
            acc=_mean(sel_correct.double()),
            n_correct=float(sel_correct.sum()),
            n_wrong=float((~sel_correct).sum()),
        )
        tok_correct = correct[row]
        for label, keep in (("correct", tok_correct), ("wrong", ~tok_correct)):
            if keep.any():
                m[f"k1_tok_{label}"] = float(-adv[keep].mean())

    bucket = tok["bucket"][sel]
    for b, label in enumerate(POS_NAMES):
        keep = bucket == b
        if keep.any():
            m[f"k1_tok_pos_{label}"] = float(-adv[keep].mean())
            m[f"abs_adv_pos_{label}"] = float(adv[keep].abs().mean())
    return m


def _mean(x: torch.Tensor) -> float:
    return float(x.mean()) if x.numel() else 0.0

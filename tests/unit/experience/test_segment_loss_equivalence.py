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
"""Same-evidence per-call versus segment-row loss/gradient conservation.

Oracle in the spirit of upstream NeMo RL PR #4124's
``test_cc_loss_equivalence.py`` on this repo's finalizer: the rows the
finalizer publishes for a rollout (the terminal chain as the canonical row
plus one extra row per other chain -- here a ``compaction_segment`` and a
``compaction_summary``) must carry exactly the same per-token training
evidence as rows built independently per model call from the staged
records: for every generated token, the same conditioning prefix, the same
generation log-prob and the same (rollout-level) advantage, each exactly
once. On a causal model the two layouts then yield identical policy-gradient
loss and gradients under the same token denominator.

Host-runnable part (pure): finalize through Gym's real verifier for the
terminal chain and a fake ``verify_and_linearize_all`` for the extra chains
(the scaffolding ``tests/unit/data_plane/test_rollout_reassembler.py`` uses),
then compare the per-token action multisets of the two layouts and run the
corruption controls at the action level.

Container-only part: run the real ``_advantage_stage`` (tensordict, Ray
class) on the published rows and compare ``ClippedPGLossFn`` loss and
gradients of a tiny GRU model between the two layouts, with the four
corruption controls (reordered prefix, duplicated row, inverse-segment
weighting, wrong denominator) that must each break equality.
"""

from __future__ import annotations

import importlib.util
import types
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

import nemo_gym.token_id_capture.staging.rebuild as rebuild_module  # noqa: E402

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient  # noqa: E402
from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS, TQTokenSink  # noqa: E402
from nemo_rl.experience.rollout_reassembler import RolloutReassembler  # noqa: E402

pytestmark = pytest.mark.nemo_gym


def _load_test_module(name: str, relative: str):
    """Import a helper module from tests/unit by path (no ``tests`` package import)."""
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_fixture_artifacts = _load_test_module(
    "_token_capture_test_fixtures", "data_plane/token_capture_test_fixtures.py"
).build_fixture_artifacts

STAGING = "rollout_staging_eq"
CANONICAL = "rollout_data_eq"
GROUP = "eq"
ROLLOUT_A = f"{GROUP}_g0"  # worked_example: two calls + two extra chains
ROLLOUT_B = f"{GROUP}_g1"  # single_call: one call, no extras
# One group of two rollouts, rewards 1 / 0: leave-one-out GRPO gives +1 / -1
# (a single sibling has no spread, so normalize_rewards does not divide).
REWARDS = {ROLLOUT_A: 1.0, ROLLOUT_B: 0.0}
ADVANTAGES = {ROLLOUT_A: 1.0, ROLLOUT_B: -1.0}
CANONICAL_FIELDS = [
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "token_mask",
    "sample_mask",
    "prompt_ids_for_adv",
    "total_reward",
    "mask_sample",
    "truncated",
]

# Extra chains of rollout A: (kind, tokens, prompt_len, generated logprob).
EXTRA_CHAINS = [
    ("compaction_segment", [30, 31, 32, 33, 34], 2, -0.5),
    ("compaction_summary", [40, 41, 42, 43], 2, -0.7),
]

_CHAIN_FIELDS = (
    "rollout_id",
    "token_ids",
    "token_mask",
    "logprobs",
    "model_call_ids",
    "prompt_len",
    "weight_versions",
    "weight_version_spans",
    "link_spans",
    "extras_commitments",
)


# ── staging + finalization ─────────────────────────────────────────────────


def _chain_view(row, **overrides):
    values = {name: getattr(row, name) for name in _CHAIN_FIELDS}
    values.update(
        terminal_model_call_id=row.model_call_ids[-1],
        chain_index=0,
        chain_kind="terminal",
        segment_index=0,
        boundary_parent_call_id=None,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _extra_chain(terminal, *, chain_index: int, kind: str, tokens, prompt_len, logprob):
    gen = len(tokens) - prompt_len
    return _chain_view(
        terminal,
        token_ids=list(tokens),
        token_mask=[0.0] * prompt_len + [1.0] * gen,
        logprobs=[0.0] * prompt_len + [logprob] * gen,
        model_call_ids=[f"seg{chain_index}"],
        prompt_len=prompt_len,
        link_spans=[(f"seg{chain_index}", prompt_len, gen)],
        extras_commitments=[],
        terminal_model_call_id=f"seg{chain_index}",
        chain_index=chain_index,
        chain_kind=kind,
        segment_index=chain_index,
        boundary_parent_call_id="c1" if kind != "subagent" else None,
    )


def _install_fake_linearize_all(monkeypatch) -> None:
    def fake(receipt, snapshots):
        terminal = rebuild_module.verify_and_linearize(receipt, snapshots)
        rows = [_chain_view(terminal)]
        if receipt.rollout_id == ROLLOUT_A:
            for idx, (kind, tokens, prompt_len, logprob) in enumerate(EXTRA_CHAINS, 1):
                rows.append(
                    _extra_chain(
                        terminal,
                        chain_index=idx,
                        kind=kind,
                        tokens=tokens,
                        prompt_len=prompt_len,
                        logprob=logprob,
                    )
                )
        return types.SimpleNamespace(
            rows=rows,
            skipped=[],
            num_roots=len(rows),
            num_boundary_roots=sum(1 for r in rows if r.boundary_parent_call_id),
        )

    monkeypatch.setattr(rebuild_module, "verify_and_linearize_all", fake)


def _per_call_evidence(records, *, rollout_id: str) -> list[dict[str, Any]]:
    """Reference rows straight from the staged records (one per model call).

    Call k's row is the cumulative token stream through call k with only call
    k's generated tokens in the loss mask -- the worker-side evidence, built
    without consulting the finalizer's output.
    """
    rows = []
    prefix_ids: list[int] = []
    prefix_lp: list[float] = []
    for record in records:
        ids = prefix_ids + list(record.token_ids_delta)
        mask = [0.0] * len(prefix_ids) + [float(m) for m in record.token_mask_delta]
        lps = prefix_lp + [float(v) for v in record.generation_log_probs_delta]
        rows.append(
            {
                "rollout": rollout_id,
                "input_ids": torch.tensor(ids),
                "token_mask": torch.tensor(mask),
                "generation_logprobs": torch.tensor(lps),
            }
        )
        prefix_ids, prefix_lp = ids, lps
    return rows


def _extra_chain_evidence(rollout_id: str) -> list[dict[str, Any]]:
    rows = []
    for kind, tokens, prompt_len, logprob in EXTRA_CHAINS:
        gen = len(tokens) - prompt_len
        rows.append(
            {
                "rollout": rollout_id,
                "kind": kind,
                "input_ids": torch.tensor(tokens),
                "token_mask": torch.tensor([0.0] * prompt_len + [1.0] * gen),
                "generation_logprobs": torch.tensor([0.0] * prompt_len + [logprob] * gen),
            }
        )
    return rows


def _finalize(monkeypatch):
    """Stage both rollouts, finalize the group; return (plane, finalized, reference rows)."""
    plane = NoOpDataPlaneClient()
    plane.register_partition(
        partition_id=STAGING,
        fields=list(STAGING_FIELDS),
        num_samples=64,
        consumer_tasks=["finalize"],
    )
    plane.register_partition(
        partition_id=CANONICAL,
        fields=CANONICAL_FIELDS,
        num_samples=64,
        consumer_tasks=["train"],
    )
    sink = TQTokenSink(plane, staging_partition=STAGING)
    reference: list[dict[str, Any]] = []
    receipts = []
    for rollout_id, fixture in ((ROLLOUT_A, "worked_example"), (ROLLOUT_B, "single_call")):
        records, receipt, _ = build_fixture_artifacts(fixture, rollout_id=rollout_id)
        for record in records:
            assert sink.stage(record).ok
        receipts.append(receipt.model_dump())
        reference.extend(_per_call_evidence(records, rollout_id=rollout_id))
    reference.extend(_extra_chain_evidence(ROLLOUT_A))
    _install_fake_linearize_all(monkeypatch)
    finalizer = RolloutReassembler(
        plane,
        partition_id=CANONICAL,
        staging_partition=STAGING,
        pad_token_id=0,
        max_seq_len=4096,
        segment_rows_enabled=True,
        max_rows_per_rollout=4,
    )
    finalized = finalizer.finalize_group(
        GROUP,
        [ROLLOUT_A, ROLLOUT_B],
        receipts,
        [REWARDS[ROLLOUT_A], REWARDS[ROLLOUT_B]],
        mask_sample=[False, False],
        fallback_weight_version=4,
        prompt_idx=0,
    )
    assert not finalized.dropped and finalized.meta is not None
    assert finalized.meta.sample_ids == [ROLLOUT_A, ROLLOUT_B, f"{ROLLOUT_A}_t1", f"{ROLLOUT_A}_t2"]
    assert finalized.extra_row_count == 2
    return plane, finalized, reference


def _dense(value) -> torch.Tensor:
    """Rows from the in-memory plane may be nested (jagged); pad to a matrix."""
    rows = [torch.as_tensor(r).reshape(-1) for r in value.unbind()]
    return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)


def _published_batch(plane, meta, *, advantages: dict[str, float] | None = None):
    """The finalizer's rows as dense tensors (+ advantages by rollout when given)."""
    fields = ["input_ids", "token_mask", "sample_mask", "generation_logprobs"]
    if advantages is None:
        fields.append("advantages")
    data = plane.get_samples(sample_ids=meta.sample_ids, partition_id=CANONICAL, select_fields=fields)
    batch = {key: _dense(data[key]) for key in fields}
    batch["sample_mask"] = batch["sample_mask"].reshape(-1)
    if advantages is not None:
        rollout_of = [tag["rollout_local_idx"] for tag in meta.tags]
        per_row = torch.tensor(
            [advantages[[ROLLOUT_A, ROLLOUT_B][i]] for i in rollout_of], dtype=torch.float32
        )
        batch["advantages"] = per_row.unsqueeze(-1).expand_as(batch["token_mask"]).clone()
    return batch


def _reference_batch(reference, *, advantages: dict[str, float]):
    pad = torch.nn.utils.rnn.pad_sequence
    out = {
        key: pad([row[key] for row in reference], batch_first=True)
        for key in ("input_ids", "token_mask", "generation_logprobs")
    }
    adv = torch.tensor([advantages[row["rollout"]] for row in reference], dtype=torch.float32)
    out["advantages"] = adv.unsqueeze(-1).expand_as(out["token_mask"]).clone()
    out["sample_mask"] = torch.ones(len(reference))
    return out


def _actions(batch) -> Counter:
    """Per-token training evidence, independent of row layout.

    One entry per generated token the loss trains on: its full conditioning
    prefix, the token, its advantage and its generation log-prob. Keeping the
    full prefix (not just the position) is what makes a reordered history, a
    duplicated row or an inverse-segment weight visible.
    """
    counted: Counter = Counter()
    mask = batch["token_mask"] * batch["sample_mask"].unsqueeze(-1)
    for row in range(mask.shape[0]):
        for position in mask[row].nonzero().reshape(-1).tolist():
            counted[
                (
                    tuple(batch["input_ids"][row, :position].tolist()),
                    int(batch["input_ids"][row, position]),
                    round(float(batch["advantages"][row, position]), 6),
                    round(float(batch["generation_logprobs"][row, position]), 6),
                )
            ] += 1
    return counted


# ── host-runnable: action-level conservation ──────────────────────────────


def test_segment_rows_carry_each_generated_token_exactly_once(monkeypatch):
    plane, finalized, reference = _finalize(monkeypatch)
    published = _published_batch(plane, finalized.meta, advantages=ADVANTAGES)
    expected = _reference_batch(reference, advantages=ADVANTAGES)

    # 2 + 2 (A's two calls) + 3 + 2 (A's extra chains) + 2 (B) generated tokens.
    assert sum(expected["token_mask"].sum(-1).tolist()) == 11
    assert published["token_mask"].sum() == 11
    assert all(count == 1 for count in _actions(expected).values())
    assert _actions(published) == _actions(expected)
    # Action coverage: the concatenation of every row's generated tokens of a
    # rollout is every generated token of its calls exactly once.
    tags = finalized.meta.tags
    for rollout_idx, rollout_id in enumerate((ROLLOUT_A, ROLLOUT_B)):
        rows = [r for r, tag in enumerate(tags) if tag["rollout_local_idx"] == rollout_idx]
        got = Counter(
            int(published["input_ids"][r, p])
            for r in rows
            for p in published["token_mask"][r].nonzero().reshape(-1).tolist()
        )
        want = Counter(
            int(row["input_ids"][p])
            for row in reference
            if row["rollout"] == rollout_id
            for p in row["token_mask"].nonzero().reshape(-1).tolist()
        )
        assert got == want
    assert [tag["trace_kind"] for tag in tags] == [
        "terminal",
        "terminal",
        "compaction_segment",
        "compaction_summary",
    ]
    assert [tag["rows_in_rollout"] for tag in tags] == [3, 1, 3, 3]


@pytest.mark.parametrize(
    "corruption", ["reordered_history", "duplicate_loss", "owner_weight", "missing_row"]
)
def test_action_oracle_detects_corruptions(monkeypatch, corruption):
    plane, finalized, reference = _finalize(monkeypatch)
    expected = _actions(_reference_batch(reference, advantages=ADVANTAGES))
    batch = _published_batch(plane, finalized.meta, advantages=ADVANTAGES)
    if corruption == "reordered_history":
        batch["input_ids"][0, :2] = batch["input_ids"][0, :2].flip(0)
    elif corruption == "duplicate_loss":
        batch = {key: torch.cat([value, value[:1]]) for key, value in batch.items()}
        assert max(_actions(batch).values()) == 2
    elif corruption == "owner_weight":
        # Dividing a rollout's advantage by its row count is not GRPO.
        rows_a = [r for r, tag in enumerate(finalized.meta.tags) if tag["rollout_local_idx"] == 0]
        batch["advantages"][rows_a] /= len(rows_a)
    elif corruption == "missing_row":
        batch = {key: value[:-1] for key, value in batch.items()}
    assert _actions(batch) != expected


# ── container-only: real advantage stage + policy-gradient loss ───────────


class _CausalModel(torch.nn.Module):
    """A tiny recurrent model: token order and prefix both matter."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(1024, 8)
        self.recurrent = torch.nn.GRU(8, 8, batch_first=True)
        self.head = torch.nn.Linear(8, 1024)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.recurrent(self.embedding(tokens[:, :-1]))
        return (
            self.head(hidden)
            .log_softmax(-1)
            .gather(-1, tokens[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )


def _loss_gradient(model, data, *, denominator: int):
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn

    model = deepcopy(model)
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0.0, use_importance_sampling_correction=True
        )
    )
    loss, _ = loss_fn(
        next_token_logprobs=model(data["input_ids"]),
        data=data,
        global_valid_seqs=data["sample_mask"].sum(),
        global_valid_toks=torch.tensor(float(denominator)),
    )
    loss.backward()
    gradients = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert torch.isfinite(loss) and torch.isfinite(gradients).all()
    assert gradients.norm() > 1e-8, "zero gradients cannot establish conservation"
    return loss.detach(), gradients


def _loss_batch(batch):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    return BatchedDataDict(
        {
            "input_ids": batch["input_ids"].long(),
            "token_mask": batch["token_mask"].double(),
            "sample_mask": batch["sample_mask"].double(),
            "advantages": batch["advantages"].double(),
            "generation_logprobs": batch["generation_logprobs"].double(),
        }
    )


@pytest.mark.parametrize(
    "corruption",
    [None, "reordered_history", "duplicate_loss", "owner_weight", "denominator"],
)
def test_same_calls_as_segment_rows_preserve_loss_and_gradients(monkeypatch, corruption):
    """Container-only: the real advantage stage + ClippedPGLossFn on both layouts."""
    pytest.importorskip("tensordict")
    pytest.importorskip("nemo_rl.algorithms.loss", reason="needs the trainer stack")
    import asyncio

    _controller = _load_test_module(
        "_advantage_stage_segments", "single_controller/test_advantage_stage_segments.py"
    )._controller

    plane, finalized, reference = _finalize(monkeypatch)
    meta = finalized.meta
    ctrl = _controller(plane, num_generations_per_prompt=2, seq_logprob_error_threshold=None)
    ctrl._policy_logprobs_required = False  # no prev_logprobs column was published
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid

    published = _published_batch(plane, meta)  # advantages as the stage wrote them
    rollout_of = [tag["rollout_local_idx"] for tag in meta.tags]
    for r, idx in enumerate(rollout_of):
        assert published["advantages"][r, 0].item() == pytest.approx(
            ADVANTAGES[[ROLLOUT_A, ROLLOUT_B][idx]]
        )
    expected = _reference_batch(reference, advantages=ADVANTAGES)
    assert _actions(published) == _actions(expected)

    batches = [_loss_batch(expected), _loss_batch(published)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = _CausalModel().double()
    for batch in batches:
        with torch.no_grad():
            # Freeze old-policy evidence before any corruption so a damaged
            # prefix cannot hide behind a ratio of exactly 1.
            batch["prev_logprobs"] = torch.nn.functional.pad(
                model(batch["input_ids"]) - 0.05, (1, 0)
            )
    denominator = 11
    wanted = _loss_gradient(model, batches[0], denominator=denominator)
    merged = batches[1]
    if corruption == "reordered_history":
        merged["input_ids"][0, :2] = merged["input_ids"][0, :2].flip(0)
    elif corruption == "duplicate_loss":
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        merged = BatchedDataDict(
            {key: torch.cat([value, value[:1]]) for key, value in merged.items()}
        )
    elif corruption == "owner_weight":
        rows_a = [r for r, idx in enumerate(rollout_of) if idx == 0]
        merged["advantages"][rows_a] /= len(rows_a)
    elif corruption == "denominator":
        denominator = 4  # physical-row normalization instead of tokens
    actual = _loss_gradient(model, merged, denominator=denominator)
    if corruption is None:
        for observed, target in zip(actual, wanted, strict=True):
            torch.testing.assert_close(observed, target, rtol=1e-9, atol=1e-12)
    else:
        assert not torch.allclose(actual[0], wanted[0], rtol=1e-6, atol=1e-12)
        assert not torch.allclose(actual[1], wanted[1], rtol=1e-6, atol=1e-12)

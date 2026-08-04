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
"""Blackbox finalization: token-free receipts + staged deltas -> canonical rows.

Orchestration only (docs/design-docs/tq-gym-gate-authoritative.md § 5, § 9.1):
per rollout, fetch the staged rows the receipt manifest names through the
``TokenSource``, re-verify them (digest recomputation over fetched values,
shape/mask/finite-logprob checks, length chaining, weight-version tag
equality), then delegate semantics to Gym's pure ``linearize``
(``main_chain_only`` + ``terminal_hint``). Any rejection becomes a masked
placeholder row — the group always builds exactly N rows so GRPO group shape
survives; validity folds into ``sample_mask`` (no new train field) and
placeholders copy ``prompt_ids_for_adv`` from a valid sibling so per-prompt
baselines stay well-formed.

The finalizer is the only reader of the staging partition. It returns the
canonical tensor payload and staging keys to the replay buffer, which owns
publication, index mutation, and cleanup under the checkpoint barrier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.tq_token_sink import TQTokenSource
from nemo_rl.experience.payload import pack_payload
from nemo_rl.experience.row_dump import maybe_dump_train_rows
from nemo_rl.models.generation.interfaces import (
    ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL as _ROUTED_EXPERTS_SENTINEL,
)


@dataclass(frozen=True)
class FinalizedRollout:
    """One rollout's canonical row, or its rejection."""

    rollout_id: str
    valid: bool
    rejection_reason: Optional[str]
    token_ids: list[int]
    token_mask: list[float]
    logprobs: list[float]
    prompt_len: int
    reward: float
    staging_keys: list[str]
    min_wv: Optional[int] = None
    max_wv: Optional[int] = None
    # Router replay (R3): [len(token_ids)][num_moe_layers][topk] from the
    # rebuilt chain; None when the rollout staged no extras.
    routed_experts: Optional[list] = None


@dataclass
class FinalizedGroup:
    """What ``finalize_group`` hands back for ``commit_finalized``."""

    meta: Optional[KVBatchMeta]
    fields: Optional[Any]
    group_min_wv: int
    group_max_wv: int
    staging_keys: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    # True when min_valid_fraction_per_group rejected the whole group; the
    # caller aborts the slot instead of committing it.
    dropped: bool = False


class BlackboxFinalizer:
    """Receipts -> verified canonical rows, off the generation hot path."""

    def __init__(
        self,
        dp_client: Any,
        *,
        partition_id: str,
        staging_partition: str,
        pad_token_id: int,
        mixed_weight_version_policy: str,
        min_valid_fraction_per_group: Optional[float],
        router_replay_enabled: bool = False,
    ) -> None:
        self._partition_id = partition_id
        self._pad_token_id = int(pad_token_id)
        self._mixed_weight_version_policy = mixed_weight_version_policy
        self._min_valid_fraction = min_valid_fraction_per_group
        self._router_replay_enabled = router_replay_enabled
        # (num_moe_layers, topk), learned from the first rebuilt row that
        # carries routes; placeholder-only groups need it to shape their
        # sentinel tensors consistently with the model.
        self._routed_dims: Optional[tuple[int, int]] = None
        self._source = TQTokenSource(
            dp_client,
            staging_partition=staging_partition,
            include_routed_experts=router_replay_enabled,
        )

    # ── per rollout ─────────────────────────────────────────────────────────

    def finalize_rollout(
        self, rollout_id: str, receipt: Optional[dict[str, Any]], *, reward: float
    ) -> FinalizedRollout:
        """Verify one receipt against its staged rows and linearize the main chain.

        Never raises for rollout-level problems: every rejection returns an
        invalid row whose reason feeds the metrics; the group publisher
        substitutes a placeholder.
        """
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.digest import compute_staging_digest
        from nemo_gym.token_id_capture.staging.rebuild import RebuildError, linearize
        from nemo_gym.token_id_capture.staging.records import RolloutReceipt

        def rejected(reason: str, staging_keys: list[str]) -> FinalizedRollout:
            return FinalizedRollout(
                rollout_id=rollout_id,
                valid=False,
                rejection_reason=reason,
                token_ids=[],
                token_mask=[],
                logprobs=[],
                prompt_len=0,
                reward=reward,
                staging_keys=staging_keys,
            )

        if receipt is None:
            return rejected("missing_receipt", [])
        try:
            parsed = RolloutReceipt.model_validate(receipt)
        except ValueError as error:
            return rejected(f"invalid_receipt:{error}", [])
        staging_keys = [record.staging_key for record in parsed.manifest]
        if parsed.rollout_id != rollout_id:
            return rejected(f"identity_mismatch:{parsed.rollout_id}", staging_keys)
        if parsed.failure_reason is not None:
            return rejected(f"rollout_failed:{parsed.failure_reason}", staging_keys)
        if parsed.capture_poisoned:
            return rejected("capture_poisoned", staging_keys)
        if not parsed.manifest:
            return rejected("empty_manifest", staging_keys)

        try:
            snapshots = self._source.fetch(staging_keys)
        except KeyError as error:
            return rejected(f"missing_staging_row:{error}", staging_keys)

        for record, snapshot in zip(parsed.manifest, snapshots):
            if not (
                len(snapshot.token_ids_delta)
                == len(snapshot.token_mask_delta)
                == len(snapshot.logprobs_delta)
            ):
                return rejected(f"misaligned_delta:{record.call_id}", staging_keys)
            if any(m not in (0.0, 1.0) for m in snapshot.token_mask_delta):
                return rejected(f"invalid_token_mask:{record.call_id}", staging_keys)
            if any(not math.isfinite(p) for p in snapshot.logprobs_delta):
                return rejected(f"non_finite_logprob:{record.call_id}", staging_keys)
            if record.delta_len != len(snapshot.token_ids_delta) or (
                snapshot.prev_len + record.delta_len != record.cum_len
            ):
                return rejected(f"length_mismatch:{record.call_id}", staging_keys)
            if snapshot.weight_version != record.weight_version:
                return rejected(
                    f"weight_version_mismatch:{record.call_id}", staging_keys
                )
            digest = compute_staging_digest(
                rollout_id=rollout_id,
                call_id=record.call_id,
                prev_len=snapshot.prev_len,
                token_ids_delta=snapshot.token_ids_delta,
                token_mask_delta=snapshot.token_mask_delta,
                logprobs_delta=snapshot.logprobs_delta,
            )
            if digest != record.digest:
                return rejected(f"digest_mismatch:{record.call_id}", staging_keys)

        # Parent pointers are lineage state, not storage state: rejoin from
        # the manifest before rebuilding (the storage rows carry them too,
        # but the receipt is authoritative).
        rejoined = [
            snapshot.model_copy(update={"parent_call_id": record.parent_call_id})
            for snapshot, record in zip(snapshots, parsed.manifest)
        ]
        try:
            row = linearize(
                rollout_id,
                rejoined,
                parsed.manifest,
                terminal_hint=parsed.terminal_call_id,
            )
        except (RebuildError, NotImplementedError) as error:
            return rejected(f"rebuild_failed:{error}", staging_keys)

        weight_versions = [record.weight_version for record in parsed.manifest]
        min_wv, max_wv = min(weight_versions), max(weight_versions)
        if self._mixed_weight_version_policy == "reject" and min_wv != max_wv:
            return rejected(f"mixed_weight_versions:{min_wv}..{max_wv}", staging_keys)

        return FinalizedRollout(
            rollout_id=rollout_id,
            valid=True,
            rejection_reason=None,
            token_ids=row.token_ids,
            token_mask=row.token_mask,
            logprobs=row.logprobs,
            prompt_len=row.prompt_len,
            reward=reward,
            staging_keys=staging_keys,
            min_wv=min_wv,
            max_wv=max_wv,
            # getattr: pre-R3 Gym pins' LinearizedRow has no routed_experts;
            # capture-R3 then degrades to the sentinel/group-drop path.
            routed_experts=getattr(row, "routed_experts", None),
        )

    # ── per group ───────────────────────────────────────────────────────────

    def finalize_group(
        self,
        group_id: str,
        rollout_ids: list[str],
        receipts: list[Optional[dict[str, Any]]],
        rewards: list[float],
        *,
        fallback_weight_version: int,
        canonical_sample_ids: Optional[list[str]] = None,
    ) -> FinalizedGroup:
        """Build exactly N canonical rows for one prompt group.

        Blocking (TQ round trips); run via ``asyncio.to_thread`` from the
        dispatch task. ``fallback_weight_version`` stamps a group none of
        whose rollouts produced a valid row (placeholder-only groups still
        need a staleness tag). ``rollout_ids`` are physical gate attempt IDs;
        ``canonical_sample_ids`` are stable logical sibling IDs. They are the
        same on the legacy first-attempt path.
        """
        assert len(rollout_ids) == len(receipts) == len(rewards), (
            "rollout_ids, receipts, and rewards must be parallel"
        )
        if canonical_sample_ids is None:
            canonical_sample_ids = rollout_ids
        assert len(canonical_sample_ids) == len(rollout_ids), (
            "canonical_sample_ids must be one per rollout"
        )
        rows = [
            self.finalize_rollout(rollout_id, receipt, reward=reward)
            for rollout_id, receipt, reward in zip(rollout_ids, receipts, rewards)
        ]
        valid_rows = [row for row in rows if row.valid]
        staging_keys = [key for row in rows for key in row.staging_keys]
        metrics = {
            "finalize/invalid_row_rate": 1.0 - len(valid_rows) / len(rows),
            "finalize/calls_per_rollout": (
                sum(len(row.staging_keys) for row in rows) / len(rows)
            ),
        }
        for row in rows:
            if not row.valid:
                print(
                    f"  finalize: rollout {row.rollout_id} rejected "
                    f"({row.rejection_reason}) — placeholder",
                    flush=True,
                )

        group_min_wv = min(
            (r.min_wv for r in valid_rows), default=fallback_weight_version
        )
        group_max_wv = max(
            (r.max_wv for r in valid_rows), default=fallback_weight_version
        )

        valid_fraction = len(valid_rows) / len(rows)
        if (
            self._min_valid_fraction is not None
            and valid_fraction < self._min_valid_fraction
        ):
            metrics["finalize/group_dropped"] = 1.0
            return FinalizedGroup(
                meta=None,
                fields=None,
                group_min_wv=group_min_wv,
                group_max_wv=group_max_wv,
                staging_keys=staging_keys,
                metrics=metrics,
                dropped=True,
            )

        # Placeholders borrow a valid sibling's prompt ids so per-prompt
        # baselines group correctly; an all-placeholder group uses a single
        # pad token (its rows all carry sample_mask 0 and never train).
        sibling_prompt = (
            valid_rows[0].token_ids[: valid_rows[0].prompt_len] if valid_rows else []
        ) or [self._pad_token_id]

        n = len(rows)
        seq_lens = [max(1, len(row.token_ids)) for row in rows]
        max_len = max(seq_lens)
        input_ids = torch.full((n, max_len), self._pad_token_id, dtype=torch.int64)
        token_mask = torch.zeros((n, max_len), dtype=torch.float32)
        logprobs = torch.zeros((n, max_len), dtype=torch.float32)
        prompt_ids_for_adv = torch.tensor([sibling_prompt] * n, dtype=torch.int64)
        sample_mask = torch.zeros(n, dtype=torch.float32)
        lengths = torch.tensor(seq_lens, dtype=torch.long)
        rewards_t = torch.tensor([row.reward for row in rows], dtype=torch.float32)
        for i, row in enumerate(rows):
            if not row.valid:
                continue
            length = len(row.token_ids)
            input_ids[i, :length] = torch.tensor(row.token_ids, dtype=torch.int64)
            token_mask[i, :length] = torch.tensor(row.token_mask, dtype=torch.float32)
            logprobs[i, :length] = torch.tensor(row.logprobs, dtype=torch.float32)
            sample_mask[i] = 1.0

        train_batch = {
            "input_ids": input_ids,
            "input_lengths": lengths,
            "generation_logprobs": logprobs,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "prompt_ids_for_adv": prompt_ids_for_adv,
            "total_reward": rewards_t,
        }
        if self._router_replay_enabled:
            has_routed_row = any(r.valid and r.routed_experts for r in rows)
            if not has_routed_row and self._routed_dims is None:
                # Nothing to learn (L, K) from yet — e.g. an all-poisoned
                # group before the first healthy rollout. Dropping loses no
                # training signal (no valid rows or routes) and keeps the
                # partition schema consistent for groups that do publish.
                print(
                    f"  finalize: group {group_id} dropped — router replay on "
                    "but no rollout carried routed_experts and (L, K) is "
                    "unknown yet",
                    flush=True,
                )
                metrics["finalize/group_dropped"] = 1.0
                return FinalizedGroup(
                    meta=None,
                    fields=None,
                    group_min_wv=group_min_wv,
                    group_max_wv=group_max_wv,
                    staging_keys=staging_keys,
                    metrics=metrics,
                    dropped=True,
                )
            train_batch["routed_experts"] = self._build_routed_experts_tensor(
                rows, max_len=max_len, metrics=metrics
            )
        sample_ids, fields, tags = pack_payload(
            train_batch, weight_version=group_min_wv, group_id=group_id
        )
        maybe_dump_train_rows(
            source="finalizer",
            group_id=group_id,
            sample_ids=list(sample_ids),
            train_batch=train_batch,
            weight_version=group_min_wv,
        )
        assert sample_ids == canonical_sample_ids, (
            "canonical sample ids must equal the stable logical rollout ids: "
            f"{sample_ids} != {canonical_sample_ids}"
        )
        meta = KVBatchMeta(
            partition_id=self._partition_id,
            task_name="train",
            sample_ids=list(sample_ids),
            fields=list(fields.keys()),
            sequence_lengths=[int(s) for s in lengths.tolist()],
            tags=[dict(t) for t in tags],
        )
        return FinalizedGroup(
            meta=meta,
            fields=fields,
            group_min_wv=group_min_wv,
            group_max_wv=group_max_wv,
            staging_keys=staging_keys,
            metrics=metrics,
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _build_routed_experts_tensor(
        self,
        rows: list[FinalizedRollout],
        *,
        max_len: int,
        metrics: dict[str, float],
    ) -> torch.Tensor:
        """[n, max_len, L, K] int16 routes for the group; sentinel elsewhere.

        Padding, placeholder rows, and valid rows whose rebuild carried no
        routes are all-sentinel: Megatron's replay falls back to its own
        router for exactly those positions. (L, K) is learned from the first
        rebuilt row that carries routes and cached for placeholder-only
        groups; a group arriving before any routed row has been seen cannot
        be shaped and fails loudly (unreachable once the first real rollout
        of the run finalizes).
        """
        for row in rows:
            if row.valid and row.routed_experts:
                first = row.routed_experts[0]
                self._routed_dims = (len(first), len(first[0]))
                break
        if self._routed_dims is None:
            raise RuntimeError(
                "policy.router_replay.enabled=true (token-capture mode) but no "
                "finalized rollout has carried routed_experts yet, so the "
                "placeholder group tensor cannot be shaped. Check vLLM "
                "enable_return_routed_experts and the staging-extras path."
            )
        num_moe_layers, topk = self._routed_dims
        routed = torch.full(
            (len(rows), max_len, num_moe_layers, topk),
            _ROUTED_EXPERTS_SENTINEL,
            dtype=torch.int16,
        )
        rows_with_routes = 0
        valid_rows = 0
        sentinel_tokens = 0
        covered_tokens = 0
        for i, row in enumerate(rows):
            if not row.valid:
                continue
            valid_rows += 1
            covered_tokens += len(row.token_ids)
            if not row.routed_experts:
                sentinel_tokens += len(row.token_ids)
                continue
            rows_with_routes += 1
            row_routes = torch.tensor(row.routed_experts, dtype=torch.int16)
            if row_routes.shape != (len(row.token_ids), num_moe_layers, topk):
                raise RuntimeError(
                    "rebuilt routed_experts shape "
                    f"{tuple(row_routes.shape)} does not match "
                    f"({len(row.token_ids)}, {num_moe_layers}, {topk}) for "
                    f"rollout {row.rollout_id}"
                )
            routed[i, : row_routes.shape[0]] = row_routes
            sentinel_tokens += int(
                row_routes.eq(_ROUTED_EXPERTS_SENTINEL).all(-1).all(-1).sum().item()
            )
        if valid_rows:
            metrics["finalize/routed_experts_row_coverage"] = (
                rows_with_routes / valid_rows
            )
        if covered_tokens:
            metrics["finalize/routed_experts_sentinel_token_fraction"] = (
                sentinel_tokens / covered_tokens
            )
        return routed

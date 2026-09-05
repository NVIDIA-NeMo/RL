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

Orchestration only:
per rollout, apply the rollout-level receipt guards, fetch the staged base
rows the receipt manifest names through the ``TokenSource`` (normally
validated ``StagedCallBaseSnapshot`` values), and delegate all token, digest,
lineage, and terminal-chain semantics to Gym's ``verify_and_linearize``. Any
rejection becomes a masked placeholder row — the group always publishes
exactly N rows so GRPO group shape survives; validity folds into
``sample_mask`` (no new train field) and placeholders copy
``prompt_ids_for_adv`` from a valid sibling so per-prompt baselines stay
well-formed.

Router replay runs one unified flow: both modes construct the same
``RouteAssemblyPlan`` from Gym's link spans and extras commitments. Deferred
mode publishes the encoded plan beside the canonical row and leaves staged
route fragments live until policy consumption; direct mode executes the plan
eagerly with fragments fetched in the same batch — any executor failure is a
pre-publication ``route_assembly:<reason>`` rejection.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import MASK_SAMPLE, ROUTE_PLAN_TAG, TRUNCATED
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.experience.payload import pack_payload
from nemo_rl.experience.route_assembly import (
    ROUTE_MISSING_SENTINEL,
    RouteFragment,
    execute_route_plan,
)
from nemo_rl.experience.route_plan import (
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    RouteSpan,
    encode_route_plan,
    encoded_route_plan_size_bytes,
    validate_route_plan,
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
    # Router replay (R3): [len(token_ids), num_moe_layers, topk] int16 from
    # the executed route plan; None when the rollout staged no routes.
    routed_experts: Optional[torch.Tensor] = None
    route_plan: Optional[RouteAssemblyPlan] = None


@dataclass
class FinalizedGroup:
    """What ``finalize_group`` hands back for ``commit_finalized``."""

    meta: Optional[KVBatchMeta]
    group_min_wv: int
    group_max_wv: int
    staging_keys: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    # True when the finalizer rejected the whole group as a structural outcome
    # (see drop_reason); the caller aborts the slot instead of committing it.
    # Policy decisions like a low valid-row fraction are no longer made here --
    # the caller reads valid_row_count/total_row_count and decides whether to
    # replace the group, since only the caller can source a replacement.
    dropped: bool = False
    # Which policy dropped the group, for the caller's log line.
    drop_reason: Optional[str] = None
    # Rows that verified vs. total rows in the group. 0/0 on a dropped group
    # (the caller does not read these when dropped is True).
    valid_row_count: int = 0
    total_row_count: int = 0


class RolloutReassembler:
    """Receipts -> verified rows -> N-row publish, off the generation hot path."""

    def __init__(
        self,
        dp_client: Any,
        *,
        partition_id: str,
        staging_partition: str,
        pad_token_id: int,
        max_seq_len: int,
        router_replay_enabled: bool = False,
        defer_routed_experts_to_policy: bool = False,
    ) -> None:
        self._dp_client = dp_client
        self._partition_id = partition_id
        self._pad_token_id = int(pad_token_id)
        self._max_seq_len = int(max_seq_len)
        self._router_replay_enabled = router_replay_enabled
        self._defer_routed_experts_to_policy = defer_routed_experts_to_policy
        if self._defer_routed_experts_to_policy and not self._router_replay_enabled:
            raise ValueError(
                "defer_routed_experts_to_policy requires router replay to be enabled"
            )
        self._staging_partition = staging_partition
        # (num_moe_layers, topk), learned from the first rebuilt row that
        # carries routes; placeholder-only groups need it to shape their
        # sentinel tensors consistently with the model.
        self._routed_dims: Optional[tuple[int, int]] = None
        self._source = TQTokenSource(dp_client, staging_partition=staging_partition)
        # The sink's clear() is the staging-partition delete; no staging
        # writes happen here.
        self._staging = TQTokenSink(dp_client, staging_partition=staging_partition)

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
        from nemo_gym.token_id_capture.staging.rebuild import (
            RebuildError,
            ReceiptVerificationError,
            verify_and_linearize,
        )
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
        if len(set(staging_keys)) != len(staging_keys):
            return rejected(
                "duplicate_staging_key",
                list(dict.fromkeys(staging_keys)),
            )
        records_by_call = {record.model_call_id: record for record in parsed.manifest}
        if len(records_by_call) != len(parsed.manifest):
            return rejected("duplicate_manifest_call_id", staging_keys)

        fetch_fragments = (
            self._router_replay_enabled and not self._defer_routed_experts_to_policy
        )
        try:
            fetched = self._source.fetch_for_finalization(
                staging_keys, include_route_fragments=fetch_fragments
            )
        except KeyError as error:
            return rejected(f"missing_staging_row:{error}", staging_keys)
        except (TypeError, ValueError) as error:
            return rejected(f"invalid_staging_row:{error}", staging_keys)
        fetched_by_call = {}
        for record, item in zip(parsed.manifest, fetched):
            if item.staging_key != record.staging_key:
                return rejected(
                    f"staging_key_mismatch:{record.model_call_id}", staging_keys
                )
            if item.snapshot.model_call_id != record.model_call_id:
                return rejected(
                    f"call_id_mismatch:{record.model_call_id}", staging_keys
                )
            fetched_by_call[record.model_call_id] = item
        if len(fetched_by_call) != len(fetched):
            return rejected("duplicate_fetched_call_id", staging_keys)

        # All base token/digest/lineage/terminal semantics belong to Gym; the
        # finalizer never re-verifies them.
        try:
            row = verify_and_linearize(parsed, [item.snapshot for item in fetched])
        except (
            KeyError,
            ValueError,
            TypeError,
            ReceiptVerificationError,
            RebuildError,
            NotImplementedError,
        ) as error:
            return rejected(f"rebuild_failed:{error}", staging_keys)
        weight_versions = [record.weight_version for record in parsed.manifest]
        min_wv, max_wv = min(weight_versions), max(weight_versions)

        route_plan = None
        routed_experts: Optional[torch.Tensor] = None
        if self._router_replay_enabled:
            # One plan construction for both modes: join Gym's link spans and
            # extras commitments with the fetch's staging keys and route
            # lengths. Cleanup keys cover the whole manifest; off-chain rows
            # stay cleanup-owned but produce no spans.
            commitments_by_call = {
                commitment.model_call_id: commitment
                for commitment in row.extras_commitments
            }
            route_spans: list[RouteSpan] = []
            seen_span_call_ids: set[str] = set()
            for call_id, carry_len, generation_len in row.link_spans:
                if call_id in seen_span_call_ids:
                    return rejected(f"duplicate_route_span:{call_id}", staging_keys)
                seen_span_call_ids.add(call_id)
                record = records_by_call.get(call_id)
                item = fetched_by_call.get(call_id)
                commitment = commitments_by_call.get(call_id)
                if record is None or item is None or commitment is None:
                    return rejected(f"route_span_identity:{call_id}", staging_keys)
                if item.routed_len not in (0, record.delta_len):
                    return rejected(f"routed_len_mismatch:{call_id}", staging_keys)
                if generation_len < 0 or generation_len > record.delta_len:
                    return rejected(
                        f"route_generation_span_mismatch:{call_id}", staging_keys
                    )
                if carry_len < 0:
                    return rejected(
                        f"route_carry_span_mismatch:{call_id}", staging_keys
                    )
                route_spans.append(
                    RouteSpan(
                        staging_key=record.staging_key,
                        carry_len=int(carry_len),
                        generation_len=int(generation_len),
                        staged_route_len=item.routed_len,
                        extras_digest_version=commitment.extras_digest_version,
                        extras_digest=commitment.extras_digest,
                    )
                )
            if sum(span.carry_len + span.generation_len for span in route_spans) != len(
                row.token_ids
            ):
                return rejected("route_span_length_mismatch", staging_keys)
            plan = RouteAssemblyPlan(
                schema_version=ROUTE_PLAN_SCHEMA_VERSION,
                staging_partition=self._staging_partition,
                spans=tuple(route_spans),
                cleanup_staging_keys=tuple(staging_keys),
                expected_token_length=len(row.token_ids),
            )
            try:
                validate_route_plan(plan)
            except (TypeError, ValueError) as error:
                return rejected(f"invalid_route_plan:{error}", staging_keys)
            # Both modes carry the constructed plan on the rollout; only
            # deferred mode publishes it (direct mode executes it eagerly and
            # the published row carries the assembled tensor instead).
            route_plan = plan
            if not self._defer_routed_experts_to_policy:
                routed_experts, failure = self._execute_direct_plan(plan, fetched)
                if failure is not None:
                    return rejected(f"route_assembly:{failure}", staging_keys)

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
            routed_experts=routed_experts,
            route_plan=route_plan,
        )

    def _execute_direct_plan(
        self,
        plan: RouteAssemblyPlan,
        fetched: list[Any],
    ) -> tuple[Optional[torch.Tensor], Optional[str]]:
        """Run the shared executor eagerly with locally fetched fragments.

        Returns ``(None, None)`` when the rollout staged no routes at all —
        the group tensor build fills those rows with sentinels, exactly like
        a deferred row whose plan is all-sentinel.
        """
        fragments: dict[str, RouteFragment] = {
            item.staging_key: item.fragment
            for item in fetched
            if item.fragment is not None
        }
        if not any(span.staged_route_len > 0 for span in plan.spans):
            return None, None
        if fragments:
            # Direct mode has no policy model in-process; the fragments'
            # own (num_moe_layers, topk) is the learned-dims heuristic, and
            # the trainer's model-shape check remains authoritative.
            first = next(iter(fragments.values())).routes
            if first.dim() != 3:
                return None, "fragment_rank"
            self._routed_dims = (int(first.shape[1]), int(first.shape[2]))
        if self._routed_dims is None:
            return None, "missing_fragment"
        return execute_route_plan(
            plan,
            fragments,
            dims=self._routed_dims,
            canonical_len=plan.expected_token_length,
        )

    # ── per group ───────────────────────────────────────────────────────────

    def finalize_group(
        self,
        group_id: str,
        rollout_ids: list[str],
        receipts: list[Optional[dict[str, Any]]],
        rewards: list[float],
        *,
        mask_sample: list[bool],
        fallback_weight_version: int,
        prompt_idx: int,
    ) -> FinalizedGroup:
        """Publish exactly N canonical rows for one prompt group.

        Blocking (TQ round trips); run via ``asyncio.to_thread`` from the
        dispatch task. ``fallback_weight_version`` stamps a group none of
        whose rollouts produced a valid row (placeholder-only groups still
        need a staleness tag). ``mask_sample`` is the per-rollout
        advantage-stage flag the native ``pack_payload`` path emits from each
        ``Completion``; it rides along unchanged so the train pump's
        environment masking reads the same field on both paths (placeholder
        rows already train nothing through ``sample_mask`` 0). ``truncated``
        is not carried from the dispatcher -- the receipt path has no real
        tokens to measure it from at dispatch time -- so it is computed here
        instead, from each row's rebuilt length against ``max_seq_len``.
        """
        assert len(rollout_ids) == len(receipts) == len(rewards) == len(mask_sample), (
            "rollout_ids, receipts, rewards, and mask_sample must be parallel"
        )
        _group_t0 = time.perf_counter()
        rows = [
            self.finalize_rollout(rollout_id, receipt, reward=reward)
            for rollout_id, receipt, reward in zip(rollout_ids, receipts, rewards)
        ]
        _rollouts_ms = (time.perf_counter() - _group_t0) * 1000.0
        valid_rows = [row for row in rows if row.valid]
        staging_keys = [key for row in rows for key in row.staging_keys]
        metrics = {
            "finalize/invalid_row_rate": 1.0 - len(valid_rows) / len(rows),
            "finalize/calls_per_rollout": (
                sum(len(row.staging_keys) for row in rows) / len(rows)
            ),
        }
        # Ledger-derived admission counters (per group): each manifest row
        # carries its admission mode. token_in_rate near 1.0 is the capture
        # health signal (a text root only opens each chain); this replaces the
        # deleted gate metrics route.
        manifest_rows = [
            record
            for receipt in receipts
            if isinstance(receipt, dict)
            for record in (receipt.get("manifest") or [])
            if isinstance(record, dict)
        ]
        if manifest_rows:
            token_in_calls = sum(
                1 for record in manifest_rows if record.get("mode") == "token_in"
            )
            metrics["finalize/token_in_calls"] = float(token_in_calls)
            metrics["finalize/text_root_calls"] = float(
                len(manifest_rows) - token_in_calls
            )
            metrics["finalize/token_in_rate"] = token_in_calls / len(manifest_rows)
        metrics["finalize/capture_poisoned_rollouts"] = float(
            sum(
                1
                for receipt in receipts
                if isinstance(receipt, dict) and receipt.get("capture_poisoned")
            )
        )
        # Per-method terminal-selection breakdown. Witness methods
        # (declared/response_id/content) resolve from evidence; heuristic is
        # the no-witness parent-link fallback — a nonzero heuristic fraction
        # on a declaring harness is a regression signal. Failed selections
        # stamp the last stage attempted, so masked rollouts stay visible in
        # their method's bucket (cross-reference finalize/invalid_row_rate).
        # Method list is derived from Gym's own type rather than hand-copied,
        # so a new resolution method Gym adds gets a bucket automatically
        # instead of silently missing from these metrics.
        from typing import get_args

        from nemo_gym.token_id_capture.staging.records import RolloutReceipt

        terminal_selection_methods = get_args(
            RolloutReceipt.model_fields["terminal_selection"].annotation
        )
        for method in terminal_selection_methods:
            method_receipts = sum(
                1
                for receipt in receipts
                if isinstance(receipt, dict)
                and receipt.get("terminal_selection") == method
            )
            metrics[f"finalize/terminal_selection_{method}_count"] = float(
                method_receipts
            )
            metrics[f"finalize/terminal_selection_{method}_fraction"] = (
                method_receipts / len(receipts)
            )
        witness_disagreements = sum(
            1
            for receipt in receipts
            if isinstance(receipt, dict)
            and "witness_disagreement"
            in str(receipt.get("terminal_attribution_reason") or "")
        )
        metrics["finalize/terminal_witness_disagreement_count"] = float(
            witness_disagreements
        )
        rejection_reasons: Counter[str] = Counter()
        for row in rows:
            if not row.valid:
                reason_bucket = (row.rejection_reason or "unknown").split(":", 1)[0]
                rejection_reasons[reason_bucket] += 1
                print(
                    f"  finalize: rollout {row.rollout_id} rejected "
                    f"({row.rejection_reason}) — placeholder",
                    flush=True,
                )
        for reason_bucket, count in rejection_reasons.items():
            metrics[f"finalize/capture_failure_reason_{reason_bucket}_count"] = float(
                count
            )

        group_min_wv = min(
            (r.min_wv for r in valid_rows if r.min_wv is not None),
            default=fallback_weight_version,
        )
        group_max_wv = max(
            (r.max_wv for r in valid_rows if r.max_wv is not None),
            default=fallback_weight_version,
        )

        _tensorize_t0 = time.perf_counter()
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
            MASK_SAMPLE: torch.tensor(mask_sample, dtype=torch.bool),
            TRUNCATED: torch.tensor(
                [seq_len == self._max_seq_len for seq_len in seq_lens],
                dtype=torch.bool,
            ),
        }
        if self._router_replay_enabled and not self._defer_routed_experts_to_policy:
            has_routed_row = any(r.valid and r.routed_experts is not None for r in rows)
            if not has_routed_row and self._routed_dims is None and not valid_rows:
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
                self._clear_staging(staging_keys)
                metrics["finalize/group_dropped"] = 1.0
                return FinalizedGroup(
                    meta=None,
                    group_min_wv=group_min_wv,
                    group_max_wv=group_max_wv,
                    staging_keys=[],
                    metrics=metrics,
                    dropped=True,
                    drop_reason=(
                        "router replay on, no rollout carried routed_experts, "
                        "and (L, K) is unknown yet"
                    ),
                    valid_row_count=0,
                    total_row_count=0,
                )
            train_batch["routed_experts"] = self._build_routed_experts_tensor(
                rows, max_len=max_len, metrics=metrics
            )
        sample_ids, fields, tags = pack_payload(
            train_batch,
            weight_version=group_min_wv,
            group_id=group_id,
            prompt_idx=prompt_idx,
        )
        if self._defer_routed_experts_to_policy:
            encoded_sizes = 0
            span_count = 0
            for tag, row, expected_length in zip(tags, rows, seq_lens):
                plan = row.route_plan
                if plan is None:
                    plan = RouteAssemblyPlan(
                        schema_version=ROUTE_PLAN_SCHEMA_VERSION,
                        staging_partition=self._staging_partition,
                        spans=(),
                        cleanup_staging_keys=tuple(row.staging_keys),
                        expected_token_length=expected_length,
                    )
                encoded = encode_route_plan(plan)
                tag[ROUTE_PLAN_TAG] = encoded
                encoded_sizes += encoded_route_plan_size_bytes(plan)
                span_count += len(plan.spans)
            metrics["finalize/route_plan_span_count"] = float(span_count)
            metrics["finalize/route_plan_encoded_bytes"] = float(encoded_sizes)
            valid_route_rows = sum(
                1 for row in valid_rows if row.route_plan and row.route_plan.spans
            )
            if valid_rows:
                metrics["finalize/routed_experts_row_coverage"] = (
                    valid_route_rows / len(valid_rows)
                )
        assert sample_ids == rollout_ids, (
            "canonical sample ids must equal the ledger-registered rollout ids: "
            f"{sample_ids} != {rollout_ids}"
        )
        _tensorize_ms = (time.perf_counter() - _tensorize_t0) * 1000.0
        _put_t0 = time.perf_counter()
        self._call_dp(
            "put_samples",
            sample_ids=sample_ids,
            partition_id=self._partition_id,
            fields=fields,
            tags=tags,
        )
        _put_ms = (time.perf_counter() - _put_t0) * 1000.0
        _clear_ms = 0.0
        if not self._defer_routed_experts_to_policy:
            _clear_t0 = time.perf_counter()
            self._clear_staging(staging_keys)
            _clear_ms = (time.perf_counter() - _clear_t0) * 1000.0
        # Per-step W&B breakdown of training-row assembly (capture arm) rides
        # FinalizedGroup.metrics into the controller's rollout metrics.
        metrics["row_assembly/rollouts_ms"] = _rollouts_ms
        metrics["row_assembly/tensorize_ms"] = _tensorize_ms
        metrics["row_assembly/tq_put_ms"] = _put_ms
        if not self._defer_routed_experts_to_policy:
            metrics["row_assembly/clear_staging_ms"] = _clear_ms
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
            group_min_wv=group_min_wv,
            group_max_wv=group_max_wv,
            staging_keys=(staging_keys if self._defer_routed_experts_to_policy else []),
            metrics=metrics,
            valid_row_count=len(valid_rows),
            total_row_count=len(rows),
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
            if row.valid and row.routed_experts is not None:
                self._routed_dims = (
                    int(row.routed_experts.shape[1]),
                    int(row.routed_experts.shape[2]),
                )
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
            ROUTE_MISSING_SENTINEL,
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
            if row.routed_experts is None:
                sentinel_tokens += len(row.token_ids)
                continue
            rows_with_routes += 1
            row_routes = row.routed_experts
            if row_routes.shape != (len(row.token_ids), num_moe_layers, topk):
                raise RuntimeError(
                    "rebuilt routed_experts shape "
                    f"{tuple(row_routes.shape)} does not match "
                    f"({len(row.token_ids)}, {num_moe_layers}, {topk}) for "
                    f"rollout {row.rollout_id}"
                )
            routed[i, : row_routes.shape[0]] = row_routes
            sentinel_tokens += int(
                row_routes.eq(ROUTE_MISSING_SENTINEL).all(-1).all(-1).sum().item()
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

    def _clear_staging(self, staging_keys: list[str]) -> None:
        if not staging_keys:
            return
        try:
            self._staging.clear(staging_keys)
        except Exception as error:
            raise RuntimeError(
                "finalizer staging cleanup failed for known keys "
                f"partition={self._staging_partition!r}, keys={staging_keys!r}"
            ) from error

    def _call_dp(self, method_name: str, **kwargs: Any) -> Any:
        import ray

        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return ray.get(remote(**kwargs))
        return method(**kwargs)

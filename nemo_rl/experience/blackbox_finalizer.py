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
"""One assembler, two verifiers: black-box rollout finalization.

NeMo-RL keeps transport integrity — this module reconciles the sealed Gym
call manifest against the forest cursor's committed nodes exactly, fetches
the staged rows, and re-verifies digests, hash chains, mask layout, finite
logprobs, and the weight version. Gym keeps semantic assembly — chain
construction is delegated to Gym's pure ``prefix_merging`` builder over the
verified snapshot, and the projected main chain must satisfy Gym's vendored
``assert_nemo_rl_contiguity`` consumer contract.

Any reconciliation or verification failure invalidates the *rollout* (the
caller substitutes a masked placeholder row so the batch survives); it never
invalidates the batch. Reward rides the receipt as driver-carry — it is never
read from staging.

Initial cardinality rule: only one unambiguous eligible chain produces the
one canonical row keyed by ``rollout_id``. Multiple valid roots or branches,
or any quarantined node, yield a masked placeholder until branch reward and
group semantics are designed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from nemo_rl.experience.rollout_writer import (
    ForestManifest,
    ForestNodeRecord,
    compute_staging_digest,
    hash_token_ids,
)
from nemo_rl.experience.staged_token_source import (
    StagedCallSnapshot,
    StagedSnapshotTokenSource,
)


class BlackboxFinalizationError(RuntimeError):
    """A rollout-level rejection; the reason becomes the manifest row's."""


@dataclass(frozen=True)
class BlackboxRolloutRow:
    """The canonical training row for one rollout, or its rejection."""

    rollout_id: str
    valid: bool
    rejection_reason: str | None
    input_ids: list[int]
    token_mask: list[float]
    generation_logprobs: list[float]
    reward: float
    staging_keys: tuple[str, ...]


def _reconcile(
    receipt: Any, forest: ForestManifest
) -> list[tuple[str, ForestNodeRecord]]:
    """Match the sealed manifest against the forest cursor's nodes exactly.

    Staged calls join committed nodes one-to-one; missing, extra, duplicate,
    and uncollected calls all reject the rollout.
    """
    if receipt.rollout_id != forest.rollout_id:
        raise BlackboxFinalizationError(
            f"identity_mismatch:{receipt.rollout_id}!={forest.rollout_id}"
        )
    if receipt.status != "completed":
        raise BlackboxFinalizationError(f"receipt_status:{receipt.status}")
    if forest.failure_reason is not None:
        raise BlackboxFinalizationError(f"cursor_failed:{forest.failure_reason}")

    staged_calls = [
        entry.call_id
        for entry in receipt.manifest.entries
        if entry.stage_disposition == "staged"
    ]
    nodes = {node.call_id: node for node in forest.nodes}
    ordered: list[tuple[str, ForestNodeRecord]] = []
    for call_id in staged_calls:
        node = nodes.get(call_id)
        if node is None:
            raise BlackboxFinalizationError(f"missing_staged_node:{call_id}")
        if node.state != "committed":
            raise BlackboxFinalizationError(
                f"uncollected_call:{call_id}:{node.state}:{node.failure_reason or ''}"
            )
        ordered.append((call_id, node))
    committed_extra = {
        node.call_id
        for node in forest.nodes
        if node.state == "committed" and node.call_id not in set(staged_calls)
    }
    if committed_extra:
        raise BlackboxFinalizationError(f"orphan_staged_node:{sorted(committed_extra)}")
    if not ordered:
        raise BlackboxFinalizationError("no_staged_calls")
    return ordered


def _fetch_and_verify_snapshots(
    *,
    dp_client: Any,
    staging_partition: str,
    rollout_id: str,
    ordered_nodes: list[tuple[str, ForestNodeRecord]],
    expected_weight_version: int,
) -> list[StagedCallSnapshot]:
    snapshots: list[StagedCallSnapshot] = []
    for call_id, node in ordered_nodes:
        if node.weight_version != expected_weight_version:
            raise BlackboxFinalizationError(
                f"weight_version_mismatch:{call_id}:{node.weight_version}"
                f"!={expected_weight_version}"
            )
        staging_key = node.staging_key or f"{rollout_id}/{call_id}"
        try:
            row = dp_client.get_samples(
                sample_ids=[staging_key],
                partition_id=staging_partition,
                select_fields=[
                    "token_ids_delta",
                    "token_mask_delta",
                    "generation_logprobs_delta",
                ],
            )
        except (KeyError, RuntimeError, ValueError) as error:
            raise BlackboxFinalizationError(
                f"missing_staging_row:{staging_key}"
            ) from error
        token_ids = [int(t) for t in row["token_ids_delta"][0]]
        token_mask = [float(m) for m in row["token_mask_delta"][0]]
        logprobs = [float(p) for p in row["generation_logprobs_delta"][0]]
        if not (len(token_ids) == len(token_mask) == len(logprobs)):
            raise BlackboxFinalizationError(f"invalid_delta_shape:{call_id}")
        if any(m not in (0.0, 1.0) for m in token_mask):
            raise BlackboxFinalizationError(f"invalid_token_mask:{call_id}")
        if any(not math.isfinite(p) for p in logprobs):
            raise BlackboxFinalizationError(f"non_finite_generation_logprob:{call_id}")
        if node.prev_len + len(token_ids) != node.new_len:
            raise BlackboxFinalizationError(f"delta_length_mismatch:{call_id}")
        digest = compute_staging_digest(
            rollout_id=rollout_id,
            call_id=call_id,
            prev_len=node.prev_len,
            token_ids_delta=token_ids,
            token_mask_delta=token_mask,
            logprobs_delta=logprobs,
        )
        if digest != node.digest:
            raise BlackboxFinalizationError(f"staging_digest_mismatch:{call_id}")
        snapshots.append(
            StagedCallSnapshot(
                call_id=call_id,
                prev_len=node.prev_len,
                token_ids_delta=token_ids,
                token_mask_delta=token_mask,
                logprobs_delta=logprobs,
                weight_version=node.weight_version,
                parent_call_id=node.parent_call_id,
            )
        )
    return snapshots


def finalize_blackbox_rollout(
    *,
    dp_client: Any,
    staging_partition: str,
    receipt: dict[str, Any],
    forest_manifest: ForestManifest,
    expected_weight_version: int,
    builder: str = "prefix_merging",
    policy_model: str = "",
) -> BlackboxRolloutRow:
    """Finalize one black-box rollout into its canonical training row.

    Never raises for rollout-level problems: any reconciliation, integrity,
    or cardinality failure returns an invalid row (``valid=False``) whose
    reason feeds the manifest log, and the caller substitutes a placeholder.
    """
    from nemo_gym.observability.integration import RolloutReceipt
    from nemo_gym.trajectory.builder import (
        assert_nemo_rl_contiguity,
        build_trajectories,
        project_main_chain_response,
    )
    from nemo_gym.trajectory.registry import get_builder

    rollout_id = str(receipt.get("rollout_id", forest_manifest.rollout_id))
    staging_keys = tuple(
        node.staging_key
        for node in forest_manifest.nodes
        if node.staging_key is not None
    )

    def rejected(reason: str) -> BlackboxRolloutRow:
        return BlackboxRolloutRow(
            rollout_id=rollout_id,
            valid=False,
            rejection_reason=reason,
            input_ids=[],
            token_mask=[],
            generation_logprobs=[],
            reward=0.0,
            staging_keys=staging_keys,
        )

    try:
        parsed = RolloutReceipt.model_validate(receipt)
        ordered_nodes = _reconcile(parsed, forest_manifest)
        snapshots = _fetch_and_verify_snapshots(
            dp_client=dp_client,
            staging_partition=staging_partition,
            rollout_id=forest_manifest.rollout_id,
            ordered_nodes=ordered_nodes,
            expected_weight_version=expected_weight_version,
        )
        source = StagedSnapshotTokenSource(
            forest_manifest.rollout_id, snapshots, model=policy_model
        )
        # Per-node terminal check: each reconstructed cumulative sequence must
        # reproduce the node's committed length and hash exactly.
        entries = source.entries(forest_manifest.rollout_id)
        node_by_call = dict(ordered_nodes)
        for entry in entries:
            cumulative = list(entry.prompt_token_ids) + list(entry.generation_token_ids)
            node = node_by_call[entry.request_id]
            if len(cumulative) != node.new_len or (
                hash_token_ids(cumulative) != node.new_hash
            ):
                raise BlackboxFinalizationError(
                    f"terminal_hash_mismatch:{entry.request_id}"
                )
        trajectories = build_trajectories(
            forest_manifest.rollout_id,
            source,
            builder=builder,
            reward=parsed.reward,
            policy_model=policy_model,
        )
        if not trajectories:
            raise BlackboxFinalizationError("empty_forest")
        quarantined = trajectories[0].provenance.get("quarantined_branches", 0)
        if len(trajectories) != 1 or trajectories[0].chain_id != "main" or quarantined:
            raise BlackboxFinalizationError(
                f"ambiguous_forest:chains={len(trajectories)}:quarantined={quarantined}"
            )
        # Shared consumer contract: the projected chain must be prefix-
        # contiguous exactly as NeMo-RL asserts on native responses.
        chains = get_builder(builder)(source.entries(forest_manifest.rollout_id)).chains
        projection = project_main_chain_response(
            forest_manifest.rollout_id, chains, {}, model=policy_model
        )
        assert_nemo_rl_contiguity(projection)
    except BlackboxFinalizationError as error:
        return rejected(str(error))
    except (KeyError, RuntimeError, ValueError, AssertionError) as error:
        return rejected(f"{type(error).__name__}:{error}")

    main = trajectories[0]
    return BlackboxRolloutRow(
        rollout_id=rollout_id,
        valid=True,
        rejection_reason=None,
        input_ids=list(main.token_ids),
        token_mask=[float(m) for m in main.loss_mask],
        generation_logprobs=[0.0 if lp is None else float(lp) for lp in main.logprobs],
        reward=parsed.reward,
        staging_keys=staging_keys,
    )

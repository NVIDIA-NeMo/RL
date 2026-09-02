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

"""On-policy distillation (OPD) helpers for async GRPO.

Teacher routing, config helpers, and teacher worker group creation.
Advantage computation lives in advantage_estimator.OPDAdvantageEstimator.
IS truncation lives in loss_functions.ClippedPGLoss (ICE-POP mode).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

import ray
import torch
from pydantic import BaseModel, Field

from nemo_rl.data_plane.column_io import read_columns, write_columns
from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta
from nemo_rl.data_plane.schema import TEACHER_LP_FIELDS
from nemo_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
    prepare_segment_topology,
)
from nemo_rl.experience.interfaces import PromptGroupRecord

# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------


class TeacherResourceConfig(BaseModel, extra="allow"):
    """Per-teacher resourcing for a non-colocated teacher worker group.

    ``extra="allow"`` keeps the escape hatch for arbitrary megatron settings:
    any unknown top-level key is folded into ``megatron_cfg_overrides``.
    """

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    num_nodes: int = 1
    gpus_per_node: int = 8
    precision: str = "bf16"
    micro_batch_size: int = 4
    megatron_cfg_overrides: dict[str, Any] = Field(default_factory=dict)


class NonColocatedTeachersConfig(BaseModel, extra="allow"):
    """Non-colocated (separate-GPU) teacher resourcing for on-policy distillation."""

    enabled: bool = False
    default_teacher_cfg: TeacherResourceConfig = Field(
        default_factory=TeacherResourceConfig
    )
    teacher_overrides: dict[str, TeacherResourceConfig] = Field(default_factory=dict)


class OnPolicyDistillationConfig(BaseModel, extra="allow"):
    """User-facing config for the top-level ``on_policy_distillation`` block."""

    enabled: bool = False
    teacher_model_by_agent_name: dict[str, str] = Field(default_factory=dict)
    default_teacher_alias: Optional[str] = None
    strict_agent_name_match: bool = False
    deduplicate_shared_teacher_checkpoints: bool = True
    non_colocated_teachers: Optional[NonColocatedTeachersConfig] = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _opd_cfg(master_config: Any) -> dict[str, Any]:
    """Return the on_policy_distillation sub-config as a plain dict.

    Accepts a MasterConfig (where the field is an OnPolicyDistillationConfig
    BaseModel), a plain dict, or a config object missing the field (non-OPD
    recipes like math). Downstream code reads the result dict-style.
    """
    if isinstance(master_config, dict):
        cfg = master_config.get("on_policy_distillation")
    else:
        cfg = getattr(master_config, "on_policy_distillation", None)
    if cfg is None:
        return {}
    if isinstance(cfg, BaseModel):
        return cfg.model_dump(exclude_none=True)
    return cfg


def is_opd_enabled(master_config: Any) -> bool:
    """Whether on-policy distillation is enabled in the config."""
    return bool(_opd_cfg(master_config).get("enabled", False))


def is_non_colocated_teachers_enabled(master_config: Any) -> bool:
    """Whether OPD is enabled with non-colocated (separate-GPU) teachers."""
    if not is_opd_enabled(master_config):
        return False
    return bool(
        _opd_cfg(master_config).get("non_colocated_teachers", {}).get("enabled", False)
    )


def _skip_prev_logprobs(master_config: Any) -> bool:
    """Whether the training loop will zero ``prev_logprobs`` instead of computing it.

    Mirrors the predicate in ``grpo_train``: ``force_on_policy_ratio`` with no
    ``seq_logprob_error_threshold`` skips the student logprob pass.
    """
    force_on_policy_ratio = master_config.loss_fn.force_on_policy_ratio
    seq_logprob_error_threshold = master_config.grpo.seq_logprob_error_threshold
    return bool(force_on_policy_ratio and seq_logprob_error_threshold is None)


def assert_prev_logprobs_available(master_config: Any) -> None:
    """Raise if OPD is enabled but the config would zero ``prev_logprobs``.

    OPD's advantage is ``teacher_logprobs - prev_logprobs``, so it needs a real
    student logprob.
    """
    if is_opd_enabled(master_config) and _skip_prev_logprobs(master_config):
        raise ValueError(
            "adv_estimator='opd' requires real prev_logprobs, but the config zeros them "
            "(loss_fn.force_on_policy_ratio=True with grpo.seq_logprob_error_threshold unset). "
            "Set seq_logprob_error_threshold or disable force_on_policy_ratio."
        )


# ---------------------------------------------------------------------------
# Teacher routing
# ---------------------------------------------------------------------------


def resolve_reference_aliases(
    agent_refs: list[dict],
    teacher_model_by_agent_name: dict[str, str],
    default_teacher_alias: Optional[str] = None,
    strict_agent_name_match: bool = False,
) -> list[str]:
    """Map each agent_ref to a teacher alias.

    Unmapped agents fall back to ``default_teacher_alias``; with
    ``strict_agent_name_match`` an unmapped agent raises instead.
    """
    aliases: list[str] = []
    for ref in agent_refs:
        name = ref["name"]
        if name in teacher_model_by_agent_name:
            aliases.append(name)
        elif strict_agent_name_match:
            raise ValueError(
                f"No teacher model mapping for agent '{name}'. "
                f"Available: {sorted(teacher_model_by_agent_name.keys())}"
            )
        elif default_teacher_alias:
            print(
                f"[OPD] Agent '{name}' not in teacher mapping, falling back to '{default_teacher_alias}'"
            )
            aliases.append(default_teacher_alias)
        else:
            raise ValueError(
                f"No teacher model mapping for agent '{name}' and no default_teacher_alias set."
            )
    return aliases


def get_teacher_routing_metrics(
    reference_aliases: list[str],
    teacher_model_by_agent_name: dict[str, str],
) -> dict[str, float]:
    """Compute teacher-routing diagnostics.

    Reports unique aliases, unique underlying models, and the alias→model
    compression ratio (how many aliases share each underlying teacher model).
    """
    alias_unique = len(set(reference_aliases))
    unique_models: set[str] = set()
    for alias in reference_aliases:
        if alias not in teacher_model_by_agent_name:
            raise KeyError(f"Alias '{alias}' not found in teacher_model_by_agent_name")
        unique_models.add(teacher_model_by_agent_name[alias])
    model_unique = len(unique_models)
    return {
        "on_policy_distillation/teacher_alias_unique": float(alias_unique),
        "on_policy_distillation/teacher_model_unique": float(model_unique),
        "on_policy_distillation/teacher_alias_to_model_compression": float(
            model_unique / max(alias_unique, 1)
        ),
    }


class TQTeacherLogprobCoordinator:
    """Enrich SingleController rollout rows with teacher logprobs through TQ.

    A prompt group is already present in the data plane when :meth:`enrich` is
    called. The coordinator sends its metadata to the inference-only teacher;
    teacher workers fetch their own DP shards and write the token column back
    under the same sample IDs. The controller materializes only the final row
    when unique temporary rows are needed for DP divisibility. Calls targeting
    the same deduplicated teacher group are serialized to keep NCCL collective
    ordering identical across that group's workers; distinct teachers remain
    independent and can run concurrently.
    """

    teacher_logprobs_field = "teacher_reference_logprobs"

    def __init__(
        self,
        *,
        dp_client: DataPlaneClient,
        teacher_worker_groups: dict[str, Any],
        alias_to_group_alias: dict[str, str],
        on_policy_distillation_cfg: dict[str, Any],
    ) -> None:
        if not teacher_worker_groups:
            raise ValueError(
                "TQTeacherLogprobCoordinator requires at least one teacher worker group"
            )
        self._dp_client = dp_client
        self._teacher_worker_groups = dict(teacher_worker_groups)
        self._alias_to_group_alias = dict(alias_to_group_alias)
        self._opd_cfg = dict(on_policy_distillation_cfg)
        # Physical (deduplicated) groups own locks, not routing aliases. Two
        # aliases sharing one checkpoint therefore share one collective FIFO.
        self._teacher_locks = {
            group_alias: asyncio.Lock() for group_alias in self._teacher_worker_groups
        }
        self._teacher_batches = 0
        self._teacher_samples = 0
        self._teacher_logprob_time_s = 0.0
        self._teacher_inference_time_s = 0.0
        self._teacher_lock_wait_time_s = 0.0
        self._aliases_seen: set[str] = set()

    def _resolve_teacher(self, record: PromptGroupRecord) -> tuple[str, str]:
        extra_env_info = record.extra_env_info
        agent_ref = (
            extra_env_info.get("agent_ref")
            if isinstance(extra_env_info, dict)
            else None
        )
        if not isinstance(agent_ref, dict):
            raise ValueError(
                "on_policy_distillation is enabled but this prompt group has no "
                "extra_env_info['agent_ref'] mapping to route it to a teacher. "
                "SingleController MOPD requires the NeMo-Gym rollout path, and "
                "regenerating this prompt cannot repair missing routing metadata."
            )

        teacher_model_by_agent_name = dict(
            self._opd_cfg.get("teacher_model_by_agent_name", {})
        )
        alias = resolve_reference_aliases(
            [agent_ref],
            teacher_model_by_agent_name,
            default_teacher_alias=self._opd_cfg.get("default_teacher_alias"),
            strict_agent_name_match=bool(
                self._opd_cfg.get("strict_agent_name_match", False)
            ),
        )[0]
        group_alias = self._alias_to_group_alias.get(alias, alias)
        if group_alias not in self._teacher_worker_groups:
            raise ValueError(
                f"Teacher alias {alias!r} resolved to unavailable worker group "
                f"{group_alias!r}; available groups: "
                f"{sorted(self._teacher_worker_groups)}"
            )
        return alias, group_alias

    def _enrich_sync(
        self,
        meta: KVBatchMeta,
        group_alias: str,
    ) -> float:
        """Run the blocking TQ-read, teacher inference, and TQ-write sequence."""
        if not meta.sequence_lengths:
            raise ValueError("MOPD teacher enrichment requires sequence_lengths")

        teacher = self._teacher_worker_groups[group_alias]
        dp_size = teacher.sharding_annotations.get_axis_size("data_parallel")
        if dp_size <= 0:
            raise ValueError(
                f"Teacher {group_alias!r} has invalid data-parallel size {dp_size}"
            )
        actual_batch_size = meta.size
        remainder = actual_batch_size % dp_size
        padded_meta = meta
        temporary_sample_ids: list[str] = []
        try:
            if remainder:
                pad_count = dp_size - remainder
                source_meta = meta.slice(actual_batch_size - 1, actual_batch_size)
                source_data = read_columns(
                    self._dp_client,
                    source_meta,
                    select_fields=TEACHER_LP_FIELDS,
                    pad_value_dict={"input_ids": 0},
                )
                input_ids = source_data["input_ids"]
                input_lengths = source_data["input_lengths"]
                if not isinstance(input_ids, torch.Tensor) or not isinstance(
                    input_lengths, torch.Tensor
                ):
                    raise TypeError("MOPD teacher padding inputs must be tensors")
                temporary_prefix = uuid.uuid4().hex
                temporary_sample_ids = [
                    f"{meta.sample_ids[-1]}__teacher_pad_{temporary_prefix}_{index}"
                    for index in range(pad_count)
                ]
                pad_meta = KVBatchMeta(
                    partition_id=meta.partition_id,
                    task_name=meta.task_name,
                    sample_ids=temporary_sample_ids,
                    fields=list(TEACHER_LP_FIELDS),
                    sequence_lengths=[meta.sequence_lengths[-1]] * pad_count,
                )
                # Keep this write inside the cleanup lifetime. A backend may
                # write only some rows before reporting failure.
                write_columns(
                    self._dp_client,
                    pad_meta,
                    fields={
                        "input_ids": input_ids.expand(pad_count, *input_ids.shape[1:]),
                        "input_lengths": input_lengths.expand(
                            pad_count, *input_lengths.shape[1:]
                        ),
                    },
                )
                padded_meta = meta.concat(pad_meta)

            inference_started_at = time.perf_counter()
            teacher.get_logprobs_from_meta(padded_meta)
            inference_finished_at = time.perf_counter()
        except BaseException as enrichment_error:
            if temporary_sample_ids:
                try:
                    self._dp_client.clear_samples(
                        sample_ids=temporary_sample_ids,
                        partition_id=meta.partition_id,
                    )
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        f"teacher enrichment and temporary-row cleanup both failed "
                        f"for group {group_alias!r}",
                        [enrichment_error, cleanup_error],
                    )
            raise
        if temporary_sample_ids:
            self._dp_client.clear_samples(
                sample_ids=temporary_sample_ids,
                partition_id=meta.partition_id,
            )
        return inference_finished_at - inference_started_at

    async def enrich(
        self,
        meta: KVBatchMeta,
        record: PromptGroupRecord,
    ) -> KVBatchMeta:
        """Write teacher logprobs before the replay-buffer slot becomes ready."""
        alias, group_alias = self._resolve_teacher(record)
        started_at = time.perf_counter()
        lock_started_at = time.perf_counter()
        # Wait for a physical teacher without occupying a default-executor
        # thread. Only the active inference consumes a thread-pool slot.
        async with self._teacher_locks[group_alias]:
            lock_wait_s = time.perf_counter() - lock_started_at
            # asyncio cannot cancel a running thread. Shield and explicitly
            # drain it so replay-buffer rollback never clears rows while teacher
            # workers are still fetching from or writing to those rows.
            enrichment_task = asyncio.create_task(
                asyncio.to_thread(self._enrich_sync, meta, group_alias)
            )
            try:
                inference_time_s = await asyncio.shield(enrichment_task)
            except asyncio.CancelledError:
                try:
                    await enrichment_task
                except BaseException as drain_error:
                    raise asyncio.CancelledError(
                        f"cancelled while draining teacher enrichment for {group_alias!r}"
                    ) from drain_error
                raise
        total_time_s = time.perf_counter() - started_at

        self._teacher_batches += 1
        self._teacher_samples += meta.size
        self._teacher_logprob_time_s += total_time_s
        self._teacher_inference_time_s += inference_time_s
        self._teacher_lock_wait_time_s += lock_wait_s
        self._aliases_seen.add(alias)
        print(
            f"[teacher_logprob] group={group_alias} samples={meta.size} "
            f"lock_wait={lock_wait_s:.2f}s inference={inference_time_s:.2f}s "
            f"total={total_time_s:.2f}s",
            flush=True,
        )
        return meta.with_fields([self.teacher_logprobs_field])

    def drain_metrics(self) -> dict[str, float]:
        """Return and reset teacher activity accumulated since the last drain."""
        metrics = {
            "on_policy_distillation/teacher_batches": float(self._teacher_batches),
            "on_policy_distillation/teacher_samples": float(self._teacher_samples),
            "on_policy_distillation/teacher_logprob_time_s": self._teacher_logprob_time_s,
            "on_policy_distillation/teacher_inference_time_s": self._teacher_inference_time_s,
            "on_policy_distillation/teacher_lock_wait_time_s": self._teacher_lock_wait_time_s,
        }
        if self._teacher_batches:
            # Cardinality describes what ran. On an idle step, zero reads as
            # "zero teacher models" rather than "no teacher activity."
            metrics.update(
                get_teacher_routing_metrics(
                    sorted(self._aliases_seen),
                    self._opd_cfg["teacher_model_by_agent_name"],
                )
            )
        self._teacher_batches = 0
        self._teacher_samples = 0
        self._teacher_logprob_time_s = 0.0
        self._teacher_inference_time_s = 0.0
        self._teacher_lock_wait_time_s = 0.0
        self._aliases_seen.clear()
        return metrics


# ---------------------------------------------------------------------------
# Setup helper — teacher worker group creation
# ---------------------------------------------------------------------------


def teacher_seq_pad_multiple(
    teacher_worker_groups: dict[str, Any], policy_make_seq_div_by: int
) -> int:
    """Sequence divisor to pre-pad teacher logprob inputs to.

    Packed teachers re-pad internally, so no pre-pad is needed (1). Non-packed
    teachers need the ``[B, S]`` forward pre-padded to the policy divisor, which
    must be a multiple of every teacher's ``sequence_length_pad_multiple``. All
    teachers must share one packing mode.
    """
    packing_modes = {twg.use_sequence_packing for twg in teacher_worker_groups.values()}
    if len(packing_modes) > 1:
        raise ValueError("All teachers must use the same sequence-packing mode.")
    if packing_modes != {False}:
        return 1  # no teachers, or all packed (they re-pad internally)
    for alias, twg in teacher_worker_groups.items():
        if policy_make_seq_div_by % twg.sequence_length_pad_multiple:
            raise ValueError(
                f"policy.make_sequence_length_divisible_by ({policy_make_seq_div_by}) "
                f"must be a multiple of teacher '{alias}'s pad requirement "
                f"({twg.sequence_length_pad_multiple})."
            )
    return policy_make_seq_div_by


def _validate_default_teacher_alias(opd_cfg: dict[str, Any]) -> None:
    """Validate the fallback teacher alias before reserving resources."""
    teacher_model_by_agent_name = dict(opd_cfg.get("teacher_model_by_agent_name", {}))
    default_teacher_alias = opd_cfg.get("default_teacher_alias")
    if (
        not opd_cfg.get("strict_agent_name_match", False)
        and default_teacher_alias is not None
        and default_teacher_alias not in teacher_model_by_agent_name
    ):
        raise ValueError(
            f"default_teacher_alias '{default_teacher_alias}' is not a key in "
            f"teacher_model_by_agent_name (available: "
            f"{sorted(teacher_model_by_agent_name.keys())})."
        )


def reserve_teacher_clusters(
    master_config: Any,
    *,
    segment_size: Optional[int] = None,
    teacher_segment_topology: Optional[dict[str, tuple[str, int]]] = None,
) -> dict[str, RayVirtualCluster]:
    """Create and reserve topology-aware clusters for non-colocated teachers.

    This reserves the teachers' Ray placement groups without starting teacher
    workers or loading model checkpoints. Call it before starting other
    opportunistically placed GPU services, then pass the result to
    :func:`create_teacher_worker_groups` after policy initialization.

    Args:
        master_config: Full training configuration containing the OPD settings.
        segment_size: NVLink-domain segment size from the cluster config. When
            set, every teacher is constrained to one NVLink domain.
        teacher_segment_topology: Topology remaining after policy and inference
            placement.

    Returns:
        A mapping from each deduplicated teacher alias to its reserved cluster.

    Raises:
        ValueError: If the configured fallback teacher alias is invalid.
        ResourceInsufficientError: If the requested topology segments cannot
            be formed.
        TimeoutError: If Ray cannot reserve a teacher placement group.
    """
    # Imported lazily to break the cycle: teacher_worker_group imports the OPD
    # config schemas defined in this module.
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    opd_cfg = _opd_cfg(master_config)
    _validate_default_teacher_alias(opd_cfg)
    teacher_configs = create_teacher_configs_from_opd_config(opd_cfg)

    # Running topology of still-free nodes; each teacher consumes a segment and
    # passes the remainder to the next so teachers don't collide.
    running_topology = (
        dict(teacher_segment_topology) if teacher_segment_topology else None
    )

    teacher_clusters: dict[str, RayVirtualCluster] = {}
    try:
        for teacher_config in teacher_configs:
            alias = teacher_config.alias
            num_nodes = teacher_config.num_nodes
            gpus_per_node = teacher_config.gpus_per_node

            # Pin each teacher within one NVLink domain (its whole node span is
            # one segment) so its TP/PP/CP collectives stay on NVLink.
            teacher_segment_size = None
            node_resource_constraints = None
            if segment_size is not None:
                teacher_segment_size = num_nodes
                (
                    node_resource_constraints,
                    remaining_ids,
                    _,
                ) = prepare_segment_topology(
                    num_nodes,
                    num_nodes,
                    topology=running_topology,
                    role=f"teacher:{alias}",
                )
                if running_topology is not None:
                    running_topology = {
                        node_id: running_topology[node_id] for node_id in remaining_ids
                    }

            teacher_cluster = RayVirtualCluster(
                name=f"teacher_{alias}",
                bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
                use_gpus=True,
                num_gpus_per_node=gpus_per_node,
                max_colocated_worker_groups=1,
                segment_size=teacher_segment_size,
                node_resource_constraints=node_resource_constraints,
            )
            teacher_clusters[alias] = teacher_cluster

            # Claim the resources now. Teacher workers are deliberately created
            # later so model loading cannot race with the policy checkpoint
            # conversion.
            teacher_cluster.get_placement_groups()
            print(
                f"  ✓ Reserved teacher '{alias}' cluster: "
                f"{num_nodes} node(s), {gpus_per_node} GPUs/node",
                flush=True,
            )
    except Exception:
        for teacher_cluster in teacher_clusters.values():
            teacher_cluster.shutdown()
        raise

    return teacher_clusters


def create_teacher_worker_groups(
    master_config: Any,
    policy_config: dict[str, Any],
    tokenizer: Any,
    *,
    teacher_clusters: dict[str, RayVirtualCluster],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Create TeacherWorkerGroup instances for non-colocated teachers.

    Args:
        master_config: Full training configuration containing the OPD settings.
        policy_config: Student policy configuration used as the teacher worker
            configuration template.
        tokenizer: Tokenizer passed to every teacher worker.
        teacher_clusters: Clusters already reserved by
            :func:`reserve_teacher_clusters`, keyed by teacher alias.

    Returns:
        A tuple containing the worker groups by primary teacher alias and the
        mapping from every configured alias to its primary group alias.

    Raises:
        ValueError: If teacher routing or the supplied cluster aliases are
            invalid, or teacher sequence-packing settings are incompatible.
        RuntimeError: If any teacher worker fails during initialization.
    """
    # Imported lazily to break the cycle: teacher_worker_group imports the OPD
    # config schemas defined in this module.
    from nemo_rl.models.policy.teacher_worker_group import (
        TeacherWorkerGroup,
        create_teacher_configs_from_opd_config,
    )

    opd_cfg = _opd_cfg(master_config)
    teacher_model_by_agent_name = dict(opd_cfg.get("teacher_model_by_agent_name", {}))
    _validate_default_teacher_alias(opd_cfg)

    teacher_configs = create_teacher_configs_from_opd_config(opd_cfg)
    expected_aliases = {teacher_config.alias for teacher_config in teacher_configs}
    if set(teacher_clusters) != expected_aliases:
        raise ValueError(
            "Reserved teacher cluster aliases do not match the resolved teacher "
            f"configs: expected {sorted(expected_aliases)}, "
            f"got {sorted(teacher_clusters)}."
        )

    teacher_worker_groups: dict[str, Any] = {}
    for teacher_config in teacher_configs:
        alias = teacher_config.alias
        twg = TeacherWorkerGroup(
            teacher_cfg=teacher_config,
            cluster=teacher_clusters[alias],
            policy_config=policy_config,
            tokenizer=tokenizer,
        )
        teacher_worker_groups[alias] = twg
        print(
            f"  ✓ Initialized teacher '{alias}' workers",
            flush=True,
        )

    # Verify all teacher workers are alive (actor __init__ runs async and
    # failures are otherwise silent until the first remote call).
    print("  Verifying teacher workers are healthy...", flush=True)
    for alias, twg in teacher_worker_groups.items():
        try:
            refs = [w.__ray_ready__.remote() for w in twg.worker_group.workers]
            ray.get(refs, timeout=1800)
        except Exception as e:
            raise RuntimeError(
                f"Teacher '{alias}' worker(s) failed during initialization. "
                f"This often means a stale cached mcore checkpoint — try deleting "
                f"the cached checkpoint under $HF_HOME/nemo_rl/ and rerunning.\n"
                f"Original error: {e}"
            ) from e
    print("  ✓ All teacher workers healthy", flush=True)

    # Reject a mixed/incompatible teacher packing config (raises).
    teacher_seq_pad_multiple(
        teacher_worker_groups, policy_config["make_sequence_length_divisible_by"]
    )

    # Build alias -> group_alias mapping for deduplication
    alias_to_group_alias: dict[str, str] = {}
    model_to_primary: dict[str, str] = {}
    for teacher_config in teacher_configs:
        model_to_primary[teacher_config.model_name] = teacher_config.alias
    for alias, model_name in teacher_model_by_agent_name.items():
        alias_to_group_alias[alias] = model_to_primary.get(model_name, alias)

    return teacher_worker_groups, alias_to_group_alias

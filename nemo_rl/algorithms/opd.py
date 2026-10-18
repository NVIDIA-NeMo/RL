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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, Optional

import ray
from pydantic import BaseModel, Field, PositiveInt, field_validator

from nemo_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
    prepare_segment_topology,
)

if TYPE_CHECKING:
    # Typing only: teacher_worker_group imports this module's schemas.
    from nemo_rl.models.policy.teacher_worker_group import TeacherConfig

FP32_LM_HEAD_TF32 = "tf32"


def mopd_fp32_lm_head_mode(value: object, *, key: str) -> Literal[False, "tf32"]:
    """Normalize an ``fp32_lm_head`` setting for MOPD: ``False`` (off) or ``"tf32"``.

    MOPD supports only the TF32 fp32 head. ``true`` (the strict-fp32 head, whose
    GEMM costs ~25x a bf16 one) is not implemented for teachers, and teachers
    must match the student, so it is rejected for both.

    Raises:
        NotImplementedError: For ``true``.
        ValueError: For any value other than false/unset, ``true``, or ``"tf32"``.
    """
    if value is None or value is False:
        return False
    if isinstance(value, str) and value == FP32_LM_HEAD_TF32:
        return FP32_LM_HEAD_TF32
    if value is True:
        raise NotImplementedError(
            f"{key}=true (the strict-fp32 LM head) is not implemented for MOPD "
            f'teachers, which must match the student; use "{FP32_LM_HEAD_TF32}" or '
            "false."
        )
    raise ValueError(
        f'{key} must be false or "{FP32_LM_HEAD_TF32}" for MOPD; got {value!r}.'
    )


def _reject_fp32_lm_head_in_overrides(
    value: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if value and "fp32_lm_head" in value:
        raise ValueError(
            "set fp32_lm_head as a teacher field (on_policy_distillation."
            "non_colocated_teachers.default_teacher_cfg.fp32_lm_head or "
            "teacher_overrides.<alias>.fp32_lm_head), not inside "
            "megatron_cfg_overrides."
        )
    return value


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
    # Run this teacher's LM head in fp32 on TF32 tensor cores ("tf32"; see
    # policy.megatron_cfg.fp32_lm_head). Set it explicitly: teachers do not
    # inherit the student's value, and setup rejects a teacher whose setting
    # differs from the student's. ``true`` (strict fp32) is not implemented.
    fp32_lm_head: bool | Literal["tf32"] = False
    megatron_cfg_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("megatron_cfg_overrides")
    @classmethod
    def _reject_fp32_lm_head_override(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_fp32_lm_head_in_overrides(value)


class TeacherResourceOverrides(BaseModel, extra="allow"):
    """Partial ``TeacherResourceConfig`` whose fields all default to ``None``.

    An omitted field never clobbers ``default_teacher_cfg`` in the
    defaults<-override merge (these blocks reach the merge via
    ``model_dump(exclude_none=True)`` in ``_opd_cfg``, which strips unset
    fields). The merged result is validated as ``TeacherResourceConfig`` in
    ``create_teacher_configs_from_opd_config``.
    """

    tensor_model_parallel_size: Optional[int] = None
    pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: Optional[int] = None
    expert_model_parallel_size: Optional[int] = None
    num_nodes: Optional[int] = None
    gpus_per_node: Optional[int] = None
    precision: Optional[str] = None
    micro_batch_size: Optional[int] = None
    fp32_lm_head: Optional[bool | Literal["tf32"]] = None
    megatron_cfg_overrides: Optional[dict[str, Any]] = None

    @field_validator("megatron_cfg_overrides")
    @classmethod
    def _reject_fp32_lm_head_override(
        cls, value: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        return _reject_fp32_lm_head_in_overrides(value)


class NonColocatedTeachersConfig(BaseModel, extra="allow"):
    """Non-colocated (separate-GPU) teacher resourcing for on-policy distillation."""

    enabled: bool = False
    default_teacher_cfg: TeacherResourceOverrides = Field(
        default_factory=TeacherResourceOverrides
    )
    teacher_overrides: dict[str, TeacherResourceOverrides] = Field(default_factory=dict)


class OnPolicyDistillationConfig(BaseModel, extra="allow"):
    """User-facing config for the top-level ``on_policy_distillation`` block."""

    enabled: bool = False
    log_sample_stats: bool = False
    sample_stats_log_period: PositiveInt = 1
    log_sample_responses: bool = False
    sample_response_max_tokens: Optional[PositiveInt] = None
    log_token_stats: bool = False
    token_stats_log_period: PositiveInt = 1
    log_topk_stats: bool = False
    topk_stats_log_period: PositiveInt = 1
    topk_stats_mode: Literal["student_online_teacher_deferred", "online"] = (
        "student_online_teacher_deferred"
    )
    topk_stats_k: PositiveInt = 32
    # Applied after full top-k tensors reach the driver: this bounds saved
    # payload size, not teacher/student inference transfer or host memory.
    topk_stats_max_tokens: Optional[PositiveInt] = None
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


def assert_topk_stats_supported(master_config: Any) -> None:
    """Reject top-k diagnostic combinations that cannot run successfully."""
    opd_cfg = master_config.on_policy_distillation
    if opd_cfg is None or not opd_cfg.enabled or not opd_cfg.log_topk_stats:
        return

    policy_cfg = master_config.policy
    if policy_cfg["dtensor_cfg"]["enabled"]:
        raise ValueError(
            "OPD top-k diagnostics require the Megatron policy backend; "
            "policy.dtensor_cfg.enabled must be false."
        )
    megatron_cfg = policy_cfg.get("megatron_cfg")
    if megatron_cfg is not None and megatron_cfg.get("use_fused_linear_logprobs"):
        raise ValueError(
            "OPD top-k diagnostics are incompatible with "
            "policy.megatron_cfg.use_fused_linear_logprobs=true."
        )
    if opd_cfg.topk_stats_mode == "online" and (
        opd_cfg.non_colocated_teachers is None
        or not opd_cfg.non_colocated_teachers.enabled
    ):
        raise ValueError(
            "on_policy_distillation.topk_stats_mode='online' requires "
            "non_colocated_teachers.enabled=true."
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


def _policy_cfg(master_config: Any) -> dict[str, Any]:
    """Return the student's policy config as a plain dict (MasterConfig or dict)."""
    if isinstance(master_config, dict):
        return dict(master_config.get("policy") or {})
    return dict(getattr(master_config, "policy", None) or {})


def validate_teacher_fp32_lm_head(
    master_config: Any, teacher_configs: Iterable[TeacherConfig]
) -> None:
    """Require every non-colocated teacher's fp32 LM head to match the student's.

    Teachers never inherit ``policy.megatron_cfg.fp32_lm_head``: each sets its
    own (``non_colocated_teachers.default_teacher_cfg`` or
    ``teacher_overrides.<alias>``), and MOPD supports only ``false`` and
    ``"tf32"``. A one-sided fp32 head would mix logit precisions in the
    teacher/student log-probability gap. With the head on, a teacher
    ``use_fused_linear_logprobs: true`` (inherited or overridden) is rejected too:
    the fused linear+CE kernel bypasses ``output_layer`` and with it the head.

    Args:
        master_config: Full training configuration (MasterConfig or dict).
        teacher_configs: Resolved teacher configs, see
            ``create_teacher_configs_from_opd_config``.

    Raises:
        NotImplementedError: If the student or a teacher sets ``true``.
        ValueError: On a mismatch, an invalid value, or fused logprobs.
    """
    teacher_configs = list(teacher_configs)
    if not teacher_configs:
        return
    megatron_cfg = _policy_cfg(master_config).get("megatron_cfg") or {}
    student_value = megatron_cfg.get("fp32_lm_head", False)
    student_mode = mopd_fp32_lm_head_mode(
        student_value, key="policy.megatron_cfg.fp32_lm_head"
    )
    mismatched = [
        f"{config.alias}={config.fp32_lm_head!r}"
        for config in teacher_configs
        if mopd_fp32_lm_head_mode(
            config.fp32_lm_head, key=f"fp32_lm_head of OPD teacher {config.alias!r}"
        )
        != student_mode
    ]
    if mismatched:
        raise ValueError(
            "MOPD teachers must use the same fp32 LM head setting as the student: "
            f"policy.megatron_cfg.fp32_lm_head={student_value!r}, but teacher "
            f"fp32_lm_head is {', '.join(mismatched)}. Teachers do not inherit the "
            "student's value; set on_policy_distillation.non_colocated_teachers."
            "default_teacher_cfg.fp32_lm_head (or teacher_overrides.<alias>."
            "fp32_lm_head) to match."
        )
    if not student_mode:
        return

    student_fused = bool(megatron_cfg.get("use_fused_linear_logprobs", False))
    fused = [
        config.alias
        for config in teacher_configs
        if config.megatron_cfg_overrides.get("use_fused_linear_logprobs", student_fused)
    ]
    if fused:
        raise ValueError(
            f"OPD teacher(s) {fused}: fp32_lm_head has no effect with "
            "use_fused_linear_logprobs=true (the fused linear+CE kernel bypasses "
            "output_layer). Disable one of them."
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
    validate_teacher_fp32_lm_head(master_config, teacher_configs)
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

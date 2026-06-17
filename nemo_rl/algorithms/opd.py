# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------


class NonColocatedTeachersConfig(BaseModel, extra="allow"):
    """Non-colocated (separate-GPU) teacher resourcing for on-policy distillation."""

    enabled: bool = False
    default_teacher_cfg: dict[str, Any] = Field(default_factory=dict)
    teacher_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


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
    return bool(_opd_cfg(master_config).get("non_colocated_teachers", {}).get("enabled", False))


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
            print(f"[OPD] Agent '{name}' not in teacher mapping, falling back to '{default_teacher_alias}'")
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
    """Routing diagnostics: unique aliases, unique models, and the alias→model
    compression ratio (how many aliases share each underlying teacher model)."""
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


def create_teacher_worker_groups(
    master_config: Any,
    policy_config: dict[str, Any],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Create TeacherWorkerGroup instances for non-colocated teachers.

    Returns (teacher_worker_groups, alias_to_group_alias).
    """
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
    from nemo_rl.models.policy.teacher_worker_group import (
        TeacherWorkerGroup,
        create_teacher_configs_from_opd_config,
    )

    opd_cfg = _opd_cfg(master_config)
    non_coloc_cfg = opd_cfg.get("non_colocated_teachers", {})
    teacher_configs = create_teacher_configs_from_opd_config(opd_cfg)

    teacher_worker_groups: dict[str, Any] = {}
    for tcfg in teacher_configs:
        alias = tcfg["alias"]
        num_nodes = tcfg.get("num_nodes", 1)
        gpus_per_node = tcfg.get("gpus_per_node", 8)

        teacher_cluster = RayVirtualCluster(
            name=f"teacher_{alias}",
            bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
            use_gpus=True,
            num_gpus_per_node=gpus_per_node,
            max_colocated_worker_groups=1,
        )
        twg = TeacherWorkerGroup(
            teacher_cfg=tcfg,
            cluster=teacher_cluster,
            policy_config=policy_config,
            tokenizer=tokenizer,
        )
        teacher_worker_groups[alias] = twg
        print(
            f"  ✓ Teacher '{alias}' cluster: {num_nodes} node(s), {gpus_per_node} GPUs/node",
            flush=True,
        )

    # Verify all teacher workers are alive (actor __init__ runs async and
    # failures are otherwise silent until the first remote call).
    import ray

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
    teacher_model_by_agent_name = dict(opd_cfg.get("teacher_model_by_agent_name", {}))
    alias_to_group_alias: dict[str, str] = {}
    model_to_primary: dict[str, str] = {}
    for tcfg in teacher_configs:
        model_to_primary[tcfg["model_name"]] = tcfg["alias"]
    for alias, model_name in teacher_model_by_agent_name.items():
        alias_to_group_alias[alias] = model_to_primary.get(model_name, alias)

    return teacher_worker_groups, alias_to_group_alias

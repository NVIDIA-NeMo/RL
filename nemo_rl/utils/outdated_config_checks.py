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
"""Rejects config the code no longer accepts, pointing at the migration to apply."""

from typing import Any, Iterator


def _train_backend_configs(
    config: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yields (dotted path, config) for every block that selects a training backend.

    The policy, the value model, the teachers and the reward-model environment. A check
    for any backend key can iterate these rather than re-deriving the locations.

    Distillation keeps its single teacher under "teacher", while the multi-teacher
    algorithms use a "teachers" list, so both spellings are visited.
    """
    # policy, value model and single-teacher (distillation) blocks
    blocks = [
        (section, config.get(section)) for section in ("policy", "value", "teacher")
    ]

    # multi-teacher blocks
    teachers = config.get("teachers")
    if isinstance(teachers, (list, tuple)):
        blocks += [(f"teachers.{i}", t) for i, t in enumerate(teachers)]

    # reward-model environment block
    env = config.get("env")
    if isinstance(env, dict):
        blocks.append(("env.reward_model", env.get("reward_model")))

    for path, block in blocks:
        if isinstance(block, dict):
            yield path, block


def reject_outdated_automodel_block(config: dict[str, Any]) -> None:
    """Fail when a config still uses the old name or the removed _v2 key for the Automodel block.

    Blocks that select Megatron are skipped: the run reads neither. If such a config
    relied on the old key to disable Automodel, Policy/Value.__init__ reports the rename
    when both backends end up enabled.

    Args:
        config: The config as the user wrote it, resolved to plain dicts.
    """
    for path, backend_config in _train_backend_configs(config):
        megatron_cfg = backend_config.get("megatron_cfg")
        if isinstance(megatron_cfg, dict) and megatron_cfg.get("enabled"):
            continue

        if "dtensor_cfg" in backend_config:
            raise ValueError(
                f"{path}.dtensor_cfg has been renamed to {path}.automodel_cfg. The "
                f"contents are unchanged, only the key. Automodel is the old "
                f"dtensor_cfg with _v2=true, which is the only mode left."
            )

        automodel_cfg = backend_config.get("automodel_cfg")
        if isinstance(automodel_cfg, dict) and "_v2" in automodel_cfg:
            raise ValueError(
                f"{path}.automodel_cfg._v2 has been removed. DTensor v1 (_v2=false) is "
                f"gone and Automodel is what _v2=true selected, so delete the key."
            )


def reject_outdated_teacher_projection_matrix_path(config: dict[str, Any]) -> None:
    """Fail when an x-token teacher's projection path is outside its aligner block.

    Args:
        config: The config as the user wrote it, resolved to plain dicts.
    """
    for path, teacher_config in _train_backend_configs(config):
        if path.startswith("teachers.") and "projection_matrix_path" in teacher_config:
            raise ValueError(
                f"{path}.projection_matrix_path is no longer supported; "
                f"move it to {path}.aligner.projection_matrix_path."
            )


def reject_outdated_dataset_config(config: dict[str, Any]) -> None:
    """Fail when data still uses the flat pre-train/validation layout.

    Args:
        config: The config as the user wrote it, resolved to plain dicts.
    """
    data = config.get("data")
    if isinstance(data, dict) and "train" not in data:
        raise ValueError(
            "data has no train section. The dataset config structure changed: datasets "
            "now live under data.train and data.validation. See the dataset section of "
            "your algorithm's guide (https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/grpo.md#dataset, "
            "https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/sft.md#datasets, "
            "https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/dpo.md#datasets) and "
            "the migration guides in https://github.com/NVIDIA-NeMo/RL/pull/1649 "
            "(response datasets) and https://github.com/NVIDIA-NeMo/RL/pull/1763 "
            "(preference datasets)."
        )


def reject_outdated_metric_name_format(config: dict[str, Any]) -> None:
    """Fail when checkpointing.metric_name still uses the bare-name format.

    Args:
        config: The config as the user wrote it, resolved to plain dicts.
    """
    checkpointing = config.get("checkpointing")
    if not isinstance(checkpointing, dict):
        return

    metric_name = checkpointing.get("metric_name")
    if metric_name is None or metric_name.startswith(("train:", "val:")):
        return

    raise ValueError(
        f"checkpointing.metric_name={metric_name!r} must start with 'train:' or 'val:', "
        f"followed by the name in the matching metrics dictionary. The bare-name format "
        f"is gone, and the name after the prefix depends on the algorithm: SFT "
        f"'val:val_loss', DPO/RM 'val:validation-<dataset>_loss' (default "
        f"'val:validation-default_loss'), GRPO/PPO/distillation 'val:accuracy'."
    )


def reject_outdated_mcore_generation_keys(config: dict[str, Any]) -> None:
    """Fail when a removed `policy.generation.mcore_generation_config` key is present.

    A config copied from an old exemplar usually carries several of these at once;
    they are reported in the order below, one per run.

    Args:
        config: The config as the user wrote it, resolved to plain dicts.
    """
    policy = config.get("policy")
    if not isinstance(policy, dict):
        return
    generation = policy.get("generation")
    if not isinstance(generation, dict):
        return
    mcore_generation = generation.get("mcore_generation_config")
    if not isinstance(mcore_generation, dict):
        return
    removed = (
        (
            "offload_policy_before_refit",
            "was removed. Set policy.offload_policy_before_refit instead.",
        ),
        (
            "buffer_guaranteed_fraction",
            "was removed: Megatron-Core replaced it with paused_buffer_size_gb and "
            "NeMo-RL stopped forwarding it in PR #1902. Delete the key; buffer_size_gb "
            "alone sizes the KV pool.",
        ),
        (
            "unified_memory_level",
            "was removed: NeMo-RL stopped forwarding it to Megatron-Core in PR #2355, "
            "so it had no effect. Delete the key; kv_cache_management_mode controls KV "
            "cache placement across suspend/resume.",
        ),
    )
    for key, message in removed:
        if key in mcore_generation:
            raise ValueError(
                f"policy.generation.mcore_generation_config.{key} {message}"
            )


def reject_outdated_dynamo_router_reset_states(config: dict[str, Any]) -> None:
    """Fail when the Dynamo frontend's removed router-reset-states key is present.

    Args:
        config: The config as the user wrote it, resolved to plain dicts.
    """
    policy = config.get("policy")
    if not isinstance(policy, dict):
        return
    generation = policy.get("generation")
    if not isinstance(generation, dict):
        return
    dynamo_cfg = generation.get("dynamo_cfg")
    if not isinstance(dynamo_cfg, dict):
        return
    frontend_args = dynamo_cfg.get("frontend_args")
    if isinstance(frontend_args, dict) and "router_reset_states" in frontend_args:
        raise ValueError(
            "policy.generation.dynamo_cfg.frontend_args.router_reset_states was "
            "removed: Dynamo 1.4 dropped the frontend's --router-reset-states flag. "
            "Delete the key."
        )


# Removed rollout_recovery key -> the key that replaced it.
_RENAMED_ROLLOUT_RECOVERY_KEYS = {
    "default_granularity": "target_level",
    "task_granularity_overrides": "task_source_target_level_overrides",
    "task_source_granularity_overrides": "task_source_target_level_overrides",
    "agent_granularity_overrides": "agent_target_level_overrides",
}


def reject_outdated_rollout_recovery_granularity(config: dict[str, Any]) -> None:
    """Fail when rollout_recovery still uses the *_granularity keys.

    Args:
        config: The config as the user wrote it, resolved to plain dicts.
    """
    rollout_recovery = config.get("rollout_recovery")
    if not isinstance(rollout_recovery, dict):
        return

    removed = sorted(_RENAMED_ROLLOUT_RECOVERY_KEYS.keys() & rollout_recovery.keys())
    if not removed:
        return

    renames = ", ".join(
        f"rollout_recovery.{key} -> rollout_recovery.{_RENAMED_ROLLOUT_RECOVERY_KEYS[key]}"
        for key in removed
    )
    message = (
        f"rollout_recovery no longer accepts {removed!r}. Rename them ({renames}); "
        f"the values ('sibling', 'prompt_group') are unchanged."
    )
    if "task_granularity_overrides" in removed:
        message += (
            " task_granularity_overrides was keyed by task name; its replacement is "
            "keyed by extra_env_info.task_source."
        )
    raise ValueError(message)


def check_outdated_config(config: dict[str, Any]) -> None:
    """Fail fast on config the code no longer accepts, naming the migration to apply.

    Call this from every entrypoint on the resolved config, before the MasterConfig is
    built. Validation rejects a missing required key on its own terms, so a check that
    runs after it can never explain a removal that changed such a key's shape. Add a
    check here whenever a key is removed or the shape it accepts changes.

    Args:
        config: The config as the user wrote it, resolved by OmegaConf.to_container.
    """
    reject_outdated_automodel_block(config)
    reject_outdated_teacher_projection_matrix_path(config)
    reject_outdated_dataset_config(config)
    reject_outdated_metric_name_format(config)
    reject_outdated_mcore_generation_keys(config)
    reject_outdated_dynamo_router_reset_states(config)
    reject_outdated_rollout_recovery_granularity(config)

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
"""Driver-side factory for the SingleController (async-RL) training path.

setup builds the full SingleControllerActorArgs on the driver and the caller passes it to
SingleControllerActor.remote. Everything lives on the driver because driver-side
TQPolicy owns the worker group directly — running this inside another Ray actor nests
runtime_envs and breaks Ray's resource resolution (see the PR #2692 follow-up).
"""

from __future__ import annotations

import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DATA_PLANE_CHECKPOINT_DIR,
    DATA_PLANE_CHECKPOINT_SCHEMA_VERSION,
    LEGACY_REPLAY_BUFFER_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    TQReplayBuffer,
)
from nemo_rl.algorithms.grpo import MasterConfig as GrpoMasterConfig
from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    _create_advantage_estimator,
    _default_grpo_save_state,
    _should_use_nemo_gym,
)
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.algorithms.single_controller_utils.rollout_checkpoint import (
    BOOTSTRAP_DIRNAME,
    bootstrap_fingerprint,
    ensure_bootstrap_anchor,
    resolve_latest_snapshot,
    validate_bootstrap_anchor,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import load_dataloader_state, setup_response_data
from nemo_rl.data_plane import DataPlaneClient, build_data_plane_client
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor
from nemo_rl.experience.rollout_manager import RolloutManager
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    ROLLOUT_RECOVERY_STATE_FILENAME,
    RolloutRecoveryLedger,
)
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.router_replay import (
    configure_vllm_for_router_replay,
    router_replay_enabled,
)
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.weight_sync import WeightSynchronizer, create_weight_synchronizer


@dataclass
class SingleControllerActorArgs:
    """All inputs SingleControllerActor needs, built driver-side by setup_single_controller().

    Passed as a single arg to SingleControllerActor.remote so the actor's __init__ does
    no construction work — every heavy object is cloudpickled in.
    """

    gen_handle: Any
    trainer_handle: Any  # driver-side TQPolicy
    env_handles: dict[str, EnvironmentInterface]
    train_cluster: RayVirtualCluster
    inference_cluster: RayVirtualCluster
    dp_client: DataPlaneClient
    dataloader: StatefulDataLoader
    weight_synchronizer: WeightSynchronizer
    advantage_estimator: Any
    loss_fn: LossFunction
    rollout_manager: RolloutManager
    tq_buffer: TQReplayBuffer
    partition_id: str
    save_state: GRPOSaveState
    last_checkpoint_path: Optional[str]
    data_plane_checkpoint_metadata: Optional[dict[str, Any]] = None
    bootstrap_fingerprint: Optional[str] = None


def _maybe_restore_native_data_plane_checkpoint(
    policy: TQPolicy,
    *,
    last_checkpoint_path: Optional[str],
    save_state: GRPOSaveState,
    partition_id: str,
    sampler_name: str,
) -> Optional[dict[str, Any]]:
    """Load and validate an authoritative native TQ checkpoint when present.

    A replay-metadata or rollout-recovery sidecar is the authoritative format
    marker. Checkpoints without either artifact resume trainer state with an
    empty replay buffer; legacy tensor-bearing replay files are rejected
    rather than silently ignored. Rollout tensors are never serialized into a
    controller-side checkpoint.
    """
    if last_checkpoint_path is None:
        return None
    checkpoint_path = Path(last_checkpoint_path)
    replay_metadata_path = checkpoint_path / REPLAY_BUFFER_METADATA_FILENAME
    rollout_recovery_path = checkpoint_path / ROLLOUT_RECOVERY_STATE_FILENAME
    has_replay_metadata = replay_metadata_path.is_file()
    has_rollout_recovery = rollout_recovery_path.is_file()
    if not has_replay_metadata and not has_rollout_recovery:
        legacy_replay_path = checkpoint_path / LEGACY_REPLAY_BUFFER_FILENAME
        if legacy_replay_path.is_file():
            raise RuntimeError(
                "Checkpoint contains legacy replay_buffer.pt state, which "
                "predates authoritative native TQ replay recovery. Resume it "
                "with the older implementation or explicitly start without "
                "restoring buffered rollouts."
            )
        return None

    data_plane_path = checkpoint_path / DATA_PLANE_CHECKPOINT_DIR
    if not data_plane_path.is_dir():
        raise FileNotFoundError(
            "Metadata-only replay checkpoint requires a matching native TQ "
            f"checkpoint at {data_plane_path}"
        )

    print(f"📦 Restoring native TQ checkpoint: {data_plane_path}", flush=True)
    metadata = policy.load_data_plane_checkpoint(data_plane_path)
    if not isinstance(metadata, dict):
        raise TypeError(
            "Native TQ checkpoint load must return a metadata dictionary, "
            f"got {type(metadata).__name__}"
        )
    expected_values: dict[str, Any] = {
        "data_plane_checkpoint_schema_version": (DATA_PLANE_CHECKPOINT_SCHEMA_VERSION),
        "single_controller_train_steps": save_state["current_step"],
        "single_controller_trainer_version": save_state.get(
            "trainer_version", save_state["current_step"]
        ),
        "single_controller_epoch": save_state["current_epoch"],
        "partition_id": partition_id,
        "sampler_name": sampler_name,
        "mode": "authoritative",
    }
    if has_replay_metadata:
        expected_values["replay_metadata_schema_version"] = (
            REPLAY_BUFFER_METADATA_SCHEMA_VERSION
        )
    if has_rollout_recovery:
        expected_values["rollout_recovery_schema_version"] = (
            ROLLOUT_RECOVERY_SCHEMA_VERSION
        )
    mismatches = {
        key: {"checkpoint": metadata.get(key), "expected": expected}
        for key, expected in expected_values.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Native TQ checkpoint metadata does not match the trainer "
            f"checkpoint: {mismatches}"
        )
    group_count = 0
    if has_replay_metadata:
        manifest_digest = metadata.get("replay_manifest_digest")
        if not isinstance(manifest_digest, str) or not manifest_digest:
            raise ValueError(
                "Native TQ checkpoint metadata is missing replay_manifest_digest"
            )
        group_count = metadata.get("replay_group_count")
        if not isinstance(group_count, int) or group_count < 0:
            raise ValueError(
                "Native TQ checkpoint metadata has invalid replay_group_count: "
                f"{group_count!r}"
            )
    elif any(
        key in metadata
        for key in (
            "replay_metadata_schema_version",
            "replay_manifest_digest",
            "replay_group_count",
        )
    ):
        raise FileNotFoundError(
            "Native TQ checkpoint advertises replay metadata, but its sidecar "
            f"is missing at {replay_metadata_path}"
        )

    if has_rollout_recovery:
        recovery_digest = metadata.get("rollout_recovery_payload_sha256")
        if not isinstance(recovery_digest, str) or not recovery_digest:
            raise ValueError(
                "Native TQ checkpoint metadata is missing "
                "rollout_recovery_payload_sha256"
            )
        recovery_group_count = metadata.get("rollout_recovery_group_count")
        if not isinstance(recovery_group_count, int) or recovery_group_count < 0:
            raise ValueError(
                "Native TQ checkpoint metadata has invalid "
                f"rollout_recovery_group_count: {recovery_group_count!r}"
            )
    elif any(
        key in metadata
        for key in (
            "rollout_recovery_schema_version",
            "rollout_recovery_payload_sha256",
            "rollout_recovery_group_count",
        )
    ):
        raise FileNotFoundError(
            "Native TQ checkpoint advertises rollout recovery, but its "
            f"sidecar is missing at {rollout_recovery_path}"
        )
    print(
        f"📦 Native TQ checkpoint restored and validated: groups={group_count}",
        flush=True,
    )
    return metadata


def _maybe_restore_rollout_recovery_ledger(
    *,
    last_checkpoint_path: Optional[str],
    data_plane_checkpoint_metadata: Optional[dict[str, Any]],
    token_capture_enabled: bool,
) -> Optional[RolloutRecoveryLedger]:
    """Load and bind the rollout-recovery sidecar to the restored TQ cut."""
    if last_checkpoint_path is None:
        return None
    recovery_path = Path(last_checkpoint_path) / ROLLOUT_RECOVERY_STATE_FILENAME
    if not recovery_path.is_file():
        return None
    if not token_capture_enabled:
        raise RuntimeError(
            "Checkpoint contains rollout-recovery state, but token capture is disabled"
        )
    if data_plane_checkpoint_metadata is None:
        raise RuntimeError(
            "Rollout-recovery sidecar requires a matching restored native TQ checkpoint"
        )

    payload = recovery_path.read_bytes()
    actual_digest = hashlib.sha256(payload).hexdigest()
    expected_digest = data_plane_checkpoint_metadata.get(
        "rollout_recovery_payload_sha256"
    )
    if actual_digest != expected_digest:
        raise ValueError(
            "Rollout-recovery sidecar digest does not match the native TQ "
            f"checkpoint metadata: actual={actual_digest!r}, "
            f"expected={expected_digest!r}"
        )
    # weights_only=False: the trusted same-job sidecar may contain DatumSpec
    # values unsupported by torch's restricted weights-only unpickler.
    state = torch.load(io.BytesIO(payload), weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(
            "Rollout-recovery sidecar must contain a state dictionary, "
            f"got {type(state).__name__}"
        )
    ledger = RolloutRecoveryLedger.from_state_dict(state)
    expected_group_count = data_plane_checkpoint_metadata.get(
        "rollout_recovery_group_count"
    )
    if len(ledger) != expected_group_count:
        raise ValueError(
            "Rollout-recovery sidecar group count does not match the native "
            f"TQ checkpoint metadata: sidecar={len(ledger)}, "
            f"checkpoint={expected_group_count!r}"
        )
    print(
        f"📦 Restored rollout recovery ledger: groups={len(ledger)}",
        flush=True,
    )
    return ledger


def _build_clusters(
    master_config: MasterConfig,
) -> tuple[RayVirtualCluster, RayVirtualCluster]:
    """Allocate train + inference clusters; one shared cluster when colocated."""
    cluster_config = master_config.cluster
    generation_config = master_config.policy["generation"]
    colocated = generation_config["colocated"]["enabled"]
    backend = generation_config["backend"]
    num_nodes = cluster_config["num_nodes"]
    gpus_per_node = cluster_config["gpus_per_node"]
    port_range_low = cluster_config.get("master_port_range_low")
    port_range_high = cluster_config.get("master_port_range_high")

    if colocated:
        # Policy + generation share GPUs — one cluster.
        cluster = RayVirtualCluster(
            name="sc_policy_cluster",
            bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
            use_gpus=True,
            num_gpus_per_node=gpus_per_node,
            max_colocated_worker_groups=1 if backend == "megatron" else 2,
            port_range_low=port_range_low,
            port_range_high=port_range_high,
        )
        return cluster, cluster

    # Non-colocated: split node into train + inference clusters.
    assert backend != "megatron", (
        "The Megatron generation backend does not support non-colocated inference "
        "in SingleController."
    )
    inference_resources = generation_config["colocated"]["resources"]
    inference_gpus_per_node = inference_resources["gpus_per_node"]
    if inference_gpus_per_node is None:
        raise ValueError(
            "Non-colocated generation requires "
            "policy.generation.colocated.resources.gpus_per_node."
        )
    inference_nodes = inference_resources["num_nodes"] or 1
    if num_nodes == 1:
        train_gpus_per_node = gpus_per_node - inference_gpus_per_node
        train_nodes = 1
        assert train_gpus_per_node > 0, (
            f"Not enough GPUs for training: {gpus_per_node} - {inference_gpus_per_node} = {train_gpus_per_node}"
        )
    else:
        train_gpus_per_node = gpus_per_node
        train_nodes = num_nodes - inference_nodes
        assert train_nodes > 0, (
            f"train_nodes must be > 0: {num_nodes} - {inference_nodes} = {train_nodes}"
        )

    train_cluster = RayVirtualCluster(
        name="sc_train_cluster",
        bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
        use_gpus=True,
        num_gpus_per_node=train_gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=port_range_low,
        port_range_high=port_range_high,
    )
    inference_cluster = RayVirtualCluster(
        name="sc_inference_cluster",
        bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
        use_gpus=True,
        num_gpus_per_node=inference_gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=port_range_low,
        port_range_high=port_range_high,
    )
    return train_cluster, inference_cluster


def _build_generation(
    inference_cluster: RayVirtualCluster,
    master_config: MasterConfig,
):
    """Spin up the generation backend (vLLM or SGLang)."""
    generation_config = master_config.policy["generation"]
    generation_config["model_name"] = master_config.policy["model_name"]
    backend = generation_config["backend"]
    if backend == "vllm":
        vllm_config = cast(VllmConfig, generation_config)
        vllm_config.setdefault("vllm_kwargs", {})["hf_overrides"] = (
            master_config.policy.get("hf_config_overrides", {})
        )
        configure_vllm_for_router_replay(master_config.policy)
        gen = VllmGeneration(cluster=inference_cluster, config=vllm_config)
    elif backend == "sglang":
        sglang_config = cast(SGLangConfig, generation_config)
        sglang_config["sglang_cfg"].setdefault(
            "model_path", master_config.policy["model_name"]
        )
        gen = SGLangGeneration(
            cluster=inference_cluster,
            sglang_cfg=sglang_config,
        )
    else:
        raise ValueError(
            f"single_controller_utils.setup only supports vllm or sglang generation; got {backend!r}"
        )
    gen.finish_generation()
    return gen


def _build_trainer(
    train_cluster: RayVirtualCluster,
    master_config: MasterConfig,
    tokenizer,
    processor,
    *,
    weights_path: Optional[Path],
    optimizer_path: Optional[Path],
):
    """Build the TQ-mediated trainer (driver-side TQPolicy).

    Driver-side on purpose: instantiating TQPolicy inside another Ray
    actor nests runtime_envs and triggers Ray's
    get_accelerator_ids_for_accelerator_resource IndexError. Keep this
    here until PolicyTrainerActor (PR #2692) lands.
    """
    loss_config = master_config.loss_fn
    init_reference_model = loss_config.reference_policy_kl_penalty > 0
    return TQPolicy(
        cluster=train_cluster,
        config=master_config.policy,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=init_reference_model,
        dp_cfg=master_config.data_plane,
    )


def _generation_max_seq_len(generation_config) -> int:
    """Return the per-backend max sequence length.

    vllm uses vllm_cfg.max_model_len; sglang uses sglang_cfg.context_length;
    megatron generation has no dedicated field and routes max_new_tokens
    through as max_sequence_length on the inference worker.
    """
    backend = generation_config["backend"]
    if backend == "vllm":
        return generation_config["vllm_cfg"]["max_model_len"]
    if backend == "sglang":
        return generation_config["sglang_cfg"]["context_length"]
    if backend == "megatron":
        return generation_config["max_new_tokens"]
    raise ValueError(f"Unknown generation backend: {backend!r}")


def _clamp_max_num_steps(
    master_config: MasterConfig, dataloader: StatefulDataLoader
) -> None:
    """Clamp grpo.max_num_steps to max_num_epochs * len(dataloader)."""
    grpo_config = master_config.grpo
    max_num_epochs = grpo_config.get("max_num_epochs")
    if max_num_epochs is None:
        return
    grpo_config["max_num_steps"] = min(
        grpo_config["max_num_steps"],
        max_num_epochs * len(dataloader),
    )


def _maybe_inject_megatron_train_iters(master_config: MasterConfig) -> None:
    """Set train_iters from max_num_steps after its dataloader clamp."""
    policy_config = master_config.policy
    if not policy_config.get("megatron_cfg", {}).get("enabled", False):
        return
    grpo_config = master_config.grpo
    policy_config["megatron_cfg"]["train_iters"] = grpo_config["max_num_steps"]


def setup_single_controller(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    processor: Optional[AutoProcessor] = None,
    partition_id: str = "rollout_data",
) -> SingleControllerActorArgs:
    """Build the full SC actor args driver-side.

    Args:
        master_config: SC MasterConfig.
        tokenizer: Tokenizer used by the policy.
        processor: Optional AutoProcessor for VLM paths.
        partition_id: TQ partition the rollout writer + sampler share.

    Returns:
        SingleControllerActorArgs ready to be passed to SingleControllerActor.
    """
    validate_single_controller_config(master_config)

    # short names for config sections
    grpo_config = master_config.grpo
    dp_config = master_config.data_plane
    policy_config = master_config.policy
    generation_config = policy_config["generation"]
    data_config = master_config.data

    if (
        grpo_config["val_period"] > 0
        or grpo_config["val_at_start"]
        or grpo_config["val_at_end"]
    ):
        raise NotImplementedError(
            "SingleController doesn't support validation now, will support "
            "later. Set grpo.val_period=0, val_at_start=false, val_at_end=false."
        )
    if dp_config is None or not dp_config.get("enabled", False):
        raise ValueError(
            "single_controller_utils.setup requires "
            "master_config.data_plane.enabled=True. The async-RL "
            "SingleController path is built on the TransferQueue data plane."
        )
    if dp_config.get("checkpointing_enabled") and dp_config["backend"] != "simple":
        raise NotImplementedError(
            "SingleController data-plane checkpointing currently requires "
            "data_plane.backend='simple'; Mooncake storage cannot be restored "
            "by TQ v0.1.9."
        )
    if (
        master_config.checkpointing["enabled"]
        and master_config.async_rl.sampler.supports_buffer_checkpoint is True
        and not dp_config.get("checkpointing_enabled")
    ):
        raise ValueError(
            "SingleController checkpointing with a replay-checkpoint-capable "
            "sampler requires data_plane.checkpointing_enabled=true so "
            "completed, unconsumed rollouts are recoverable."
        )

    assert generation_config is not None, (
        "single_controller_utils.setup requires policy.generation in master_config"
    )

    if data_config["use_multiple_dataloader"]:
        raise NotImplementedError(
            "single_controller_utils does not support "
            "data.use_multiple_dataloader=True yet."
        )

    # Token capture: validate the MVP matrix loudly at setup (§ 6, § 10) and
    # give capture-enabled vLLM workers a venv that carries nemo_gym (the
    # worker hosts Gym's capture core + adapter in-process).
    token_capture_cfg = master_config.token_capture
    rollout_checkpoint_cfg = master_config.rollout_checkpointing
    if rollout_checkpoint_cfg.interval_s is not None:
        if rollout_checkpoint_cfg.interval_s <= 0:
            raise ValueError("rollout_checkpointing.interval_s must be positive")
        if rollout_checkpoint_cfg.keep_latest_k < 1:
            raise ValueError("rollout_checkpointing.keep_latest_k must be at least one")
        if not master_config.checkpointing["enabled"]:
            raise ValueError(
                "rollout checkpointing requires checkpointing.enabled=true"
            )
        if not dp_config.get("checkpointing_enabled"):
            raise ValueError(
                "rollout checkpointing requires data_plane.checkpointing_enabled=true"
            )
        if not token_capture_cfg.enabled:
            raise ValueError(
                "rollout checkpointing currently requires token_capture.enabled=true"
            )
    if token_capture_cfg.enabled:
        if master_config.checkpointing["enabled"] and not dp_config.get(
            "checkpointing_enabled"
        ):
            raise ValueError(
                "SingleController token-capture checkpointing requires "
                "data_plane.checkpointing_enabled=true so receipts and their "
                "TQ staging rows share one durable checkpoint cut."
            )
        if (
            master_config.checkpointing["enabled"]
            and master_config.async_rl.sampler.supports_buffer_checkpoint is not True
        ):
            raise NotImplementedError(
                "Token-capture checkpoint recovery currently requires a "
                "replay-checkpoint-capable sampler (windowed or in_order)."
            )
        if not _should_use_nemo_gym(master_config):
            raise ValueError(
                "token_capture.enabled requires the NeMo-Gym rollout path "
                "(env.should_use_nemo_gym=true) — the gate lives in Gym's "
                "policy model server"
            )
        if generation_config["backend"] != "vllm":
            raise NotImplementedError(
                "token_capture.enabled supports the vllm backend only; got "
                f"{generation_config['backend']!r}"
            )
        if not generation_config["vllm_cfg"]["async_engine"]:
            raise ValueError(
                "token_capture.enabled requires "
                "policy.generation.vllm_cfg.async_engine=true (the capture "
                "host is the worker's in-process HTTP server)"
            )
        from nemo_rl.distributed.ray_actor_environment_registry import (
            ACTOR_ENVIRONMENT_REGISTRY,
        )
        from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

        ACTOR_ENVIRONMENT_REGISTRY[
            "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
        ] = PY_EXECUTABLES.VLLM_GYM

        # Fill the derived gate-hosting fields (see TokenCaptureConfig):
        # a per-run control-plane bearer token, the base capture dir the
        # gate rides on, and LineageIndex capacity sized from the training
        # config (finding M: eviction of a live rollout must not happen
        # under normal operation).
        if token_capture_cfg.control_auth_token is None:
            # Deferred import: only needed on the capture path.
            import secrets

            token_capture_cfg.control_auth_token = secrets.token_hex(32)
        if token_capture_cfg.capture_dir is None:
            token_capture_cfg.capture_dir = os.path.abspath(
                os.path.join(
                    master_config.logger.get("log_dir") or "logs",
                    "gym_token_capture",
                )
            )
        if token_capture_cfg.lineage_max_rollouts is None:
            group_size = grpo_config["num_generations_per_prompt"]
            in_flight = (
                master_config.async_rl.max_buffered_rollouts
                + master_config.async_rl.max_inflight_prompts
            ) * group_size
            token_capture_cfg.lineage_max_rollouts = 2 * in_flight
        if token_capture_cfg.lineage_max_tokens is None:
            token_capture_cfg.lineage_max_tokens = (
                token_capture_cfg.lineage_max_rollouts
                * int(master_config.policy["max_total_sequence_length"])
            )

    checkpointing_pretrained = master_config.checkpointing.get("pretrained_checkpoint")
    if checkpointing_pretrained is not None:
        policy_config["pretrained_checkpoint"] = checkpointing_pretrained

    set_seed(grpo_config["seed"])

    # ==========================
    # Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config.checkpointing)
    trainer_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    save_state = cast(
        Optional[GRPOSaveState],
        checkpointer.load_training_info(trainer_checkpoint_path),
    )
    if save_state is None:
        save_state = _default_grpo_save_state()
    weights_path, optimizer_path = checkpointer.get_resume_paths(
        trainer_checkpoint_path
    )

    recovery_checkpoint_path = trainer_checkpoint_path
    bootstrap_anchor = checkpointer.checkpoint_dir / BOOTSTRAP_DIRNAME
    needs_bootstrap_identity = trainer_checkpoint_path is None and (
        master_config.rollout_checkpointing.interval_s is not None
        or bootstrap_anchor.is_dir()
    )
    bootstrap_digest = (
        bootstrap_fingerprint(master_config) if needs_bootstrap_identity else None
    )
    if trainer_checkpoint_path is not None:
        resolved_snapshot = resolve_latest_snapshot(
            Path(trainer_checkpoint_path),
            expected_train_step=save_state["current_step"],
            expected_trainer_version=save_state.get(
                "trainer_version", save_state["current_step"]
            ),
            expected_bootstrap_fingerprint=None,
        )
    else:
        if master_config.rollout_checkpointing.interval_s is not None:
            assert bootstrap_digest is not None
            bootstrap_anchor = ensure_bootstrap_anchor(
                checkpointer.checkpoint_dir,
                fingerprint=bootstrap_digest,
            )
        elif bootstrap_anchor.is_dir():
            assert bootstrap_digest is not None
            validate_bootstrap_anchor(
                bootstrap_anchor,
                fingerprint=bootstrap_digest,
            )
        resolved_snapshot = (
            resolve_latest_snapshot(
                bootstrap_anchor,
                expected_train_step=0,
                expected_trainer_version=0,
                expected_bootstrap_fingerprint=bootstrap_digest,
            )
            if bootstrap_anchor.is_dir()
            else None
        )
    if resolved_snapshot is not None:
        recovery_checkpoint_path = str(resolved_snapshot.path)
        save_state = cast(GRPOSaveState, dict(save_state))
        save_state["current_epoch"] = resolved_snapshot.manifest.current_epoch
        print(
            f"📦 Selected rollout recovery snapshot: {recovery_checkpoint_path}",
            flush=True,
        )

    # ==========================
    # Setup Dataset & Environments
    # ==========================
    # TODO: add validate dataset wiring.
    use_nemo_gym = _should_use_nemo_gym(cast(GrpoMasterConfig, master_config))
    if use_nemo_gym and generation_config["backend"] != "vllm":
        raise NotImplementedError(
            "SC NeMo-Gym integration currently supports the vllm backend "
            f"only; got {generation_config['backend']!r}"
        )
    if use_nemo_gym:
        # NeMo-Gym creates the env actor outside setup_response_data; we wire
        # it in after generation is up (it needs the OpenAI server URLs).
        response_data = setup_response_data(tokenizer, data_config, env_configs=None)
        assert len(response_data) == 2
        dataset, _val_dataset = response_data
        env_handles: dict[str, EnvironmentInterface] = {}
    else:
        response_data = setup_response_data(
            tokenizer, data_config, env_configs=master_config.env
        )
        assert len(response_data) == 4
        dataset, _val_dataset, env_handles, _val_env_handles = response_data
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=grpo_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    dataloader_checkpoint_path = recovery_checkpoint_path
    if dataloader_checkpoint_path is not None:
        dataloader_state_path = os.path.join(
            dataloader_checkpoint_path, "train_dataloader.pt"
        )
        if os.path.exists(dataloader_state_path):
            print(
                f"📦 Restoring dataloader state from checkpoint: {dataloader_state_path}"
            )
            load_dataloader_state(
                dataloader,
                dataloader_checkpoint_path,
                data_config,
            )
        else:
            print(
                f"⚠️ No dataloader state found at {dataloader_state_path}. "
                "Starting with a fresh dataloader position."
            )

    _clamp_max_num_steps(master_config, dataloader)
    _maybe_inject_megatron_train_iters(master_config)

    # ==========================
    # Setup Clusters & Workers
    # ==========================
    train_cluster, inference_cluster = _build_clusters(master_config)
    colocated = generation_config["colocated"]["enabled"]
    if colocated:
        # Colocated: vLLM prefers a clean GPU at load time, so generation
        # comes up before the policy.
        generation = _build_generation(inference_cluster, master_config)
        policy = _build_trainer(
            train_cluster,
            master_config,
            tokenizer,
            processor,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
        )
    else:
        # Non-colocated: generation + policy run on disjoint GPUs, so
        # bring them up in parallel.
        with ThreadPoolExecutor(max_workers=2) as executor:
            gen_future = executor.submit(
                _build_generation, inference_cluster, master_config
            )
            policy_future = executor.submit(
                _build_trainer,
                train_cluster,
                master_config,
                tokenizer,
                processor,
                weights_path=weights_path,
                optimizer_path=optimizer_path,
            )
            generation = gen_future.result()
            policy = policy_future.result()

    # Native TQ restore must run through the bootstrap client before the SC
    # client is created and before any rollout/train data-plane operation.
    data_plane_checkpoint_metadata = _maybe_restore_native_data_plane_checkpoint(
        policy,
        last_checkpoint_path=recovery_checkpoint_path,
        save_state=save_state,
        partition_id=partition_id,
        sampler_name=master_config.async_rl.sampler.name,
    )
    recovery_ledger = _maybe_restore_rollout_recovery_ledger(
        last_checkpoint_path=recovery_checkpoint_path,
        data_plane_checkpoint_metadata=data_plane_checkpoint_metadata,
        token_capture_enabled=token_capture_cfg.enabled,
    )

    # ==========================
    # NeMo-Gym actor (after generation is up so OpenAI URLs are available)
    # ==========================
    if use_nemo_gym:
        # TODO(#2625): Mirror GRPO's deferred vLLM load so NeMo-Gym spinup
        # overlaps model loading instead of running serially afterward.
        enable_router_replay = router_replay_enabled(master_config.policy)
        env_handles["nemo_gym"] = spinup_nemo_gym_actor(
            env_configs=master_config.env,
            base_urls=generation.dp_openai_server_base_urls,
            model_name=generation_config["model_name"],
            enable_router_replay=enable_router_replay,
            # Gate config rides into Gym's policy model server (§ 9.1).
            token_capture=(
                master_config.token_capture.model_dump()
                if master_config.token_capture.enabled
                else None
            ),
        )

    # ==========================
    # Setup Data Plane Client & Weight Sync
    # ==========================
    # Connect-only DP client; TQPolicy already bootstrapped the controller.
    dp_client = build_data_plane_client(dp_config, bootstrap=False)

    # Token-capture mode: pre-register both rollout partitions from this
    # single driver thread before any producer is live. TQ's controller
    # registers unseen field names lazily inside update_production_status
    # without a lock, so the first concurrent puts into an unregistered
    # partition can race kv_retrieve_meta and kill the controller thread
    # (see TQDataPlaneClient.register_partition).
    token_capture_cfg = master_config.token_capture
    if token_capture_cfg.enabled:
        from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS
        from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS

        group_size = grpo_config["num_generations_per_prompt"]
        num_rollout_samples = master_config.async_rl.max_buffered_rollouts * group_size
        dp_client.register_partition(
            partition_id=partition_id,
            fields=list(DP_TRAIN_FIELDS),
            num_samples=num_rollout_samples,
            consumer_tasks=["prev_lp", "ref_lp", "train"],
            grpo_group_size=group_size,
        )
        dp_client.register_partition(
            partition_id=token_capture_cfg.staging_partition,
            fields=list(STAGING_FIELDS),
            num_samples=num_rollout_samples,
            consumer_tasks=["finalize"],
        )
        # Host Gym's capture core in every vLLM DP leader (in-worker DP
        # client + TQTokenSink + the single install_capture call), and give
        # workers the initial weight version to stamp on captured calls.
        try:
            generation.setup_token_capture(
                dp_config, token_capture_cfg.staging_partition
            )
        except Exception as error:
            if "No module named 'nemo_gym'" in str(error):
                # Worker venvs are cached by actor class name
                # (nemo_rl/utils/venvs.py), so a venv prebuilt before token
                # capture predates the nemo_gym extra and is reused as-is.
                raise RuntimeError(
                    "token_capture.enabled requires nemo_gym inside the vLLM "
                    "worker venv, but the cached worker venv predates it. "
                    "Rebuild worker venvs (NRL_FORCE_REBUILD_VENVS=true) or "
                    "delete $NEMO_RL_VENV_DIR/nemo_rl.models.generation.vllm."
                    "vllm_worker_async.VllmAsyncGenerationWorker and rerun."
                ) from error
            raise
        generation.set_rollout_weight_version(0)

    backend = generation_config["backend"]
    weight_synchronizer = create_weight_synchronizer(
        policy=policy,
        generation=generation,
        generation_backend=backend,
        colocated=colocated,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
        refit_buffer_size_gb=policy_config.get("refit_buffer_size_gb"),
    )
    weight_synchronizer.init_communicator()

    # ==========================
    # Setup Algorithm + Rollout Wiring
    # ==========================
    advantage_estimator = _create_advantage_estimator(
        cast(GrpoMasterConfig, master_config)
    )
    loss_fn: LossFunction = ClippedPGLossFn(master_config.loss_fn)

    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    tq_buffer = TQReplayBuffer(
        dp_client,
        partition_id=partition_id,
        pad_value_dict={"token_ids": pad_id, "input_ids": pad_id},
        staging_partition_id=(
            token_capture_cfg.staging_partition if token_capture_cfg.enabled else None
        ),
    )
    finalizer = None
    if token_capture_cfg.enabled:
        from nemo_rl.experience.blackbox_finalizer import BlackboxFinalizer

        finalizer = BlackboxFinalizer(
            dp_client,
            partition_id=partition_id,
            staging_partition=token_capture_cfg.staging_partition,
            pad_token_id=pad_id,
            mixed_weight_version_policy=token_capture_cfg.mixed_weight_version_policy,
            min_valid_fraction_per_group=token_capture_cfg.min_valid_fraction_per_group,
        )
    rollout_manager = RolloutManager(
        tokenizer=tokenizer,
        task_to_env=env_handles,
        num_generations_per_prompt=grpo_config["num_generations_per_prompt"],
        max_seq_len=_generation_max_seq_len(generation_config),
        max_rollout_turns=grpo_config.get("max_rollout_turns"),
        policy_generation=generation,
        generation_config=generation_config,
        use_nemo_gym=use_nemo_gym,
        tq_buffer=tq_buffer,
        finalizer=finalizer,
        recovery_ledger=recovery_ledger,
    )

    return SingleControllerActorArgs(
        gen_handle=generation,
        trainer_handle=policy,
        env_handles=env_handles,
        train_cluster=train_cluster,
        inference_cluster=inference_cluster,
        dp_client=dp_client,
        dataloader=dataloader,
        weight_synchronizer=weight_synchronizer,
        advantage_estimator=advantage_estimator,
        loss_fn=loss_fn,
        rollout_manager=rollout_manager,
        tq_buffer=tq_buffer,
        partition_id=partition_id,
        save_state=save_state,
        last_checkpoint_path=recovery_checkpoint_path,
        data_plane_checkpoint_metadata=data_plane_checkpoint_metadata,
        bootstrap_fingerprint=bootstrap_digest,
    )

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""ModelExpress shard-to-shard refit for Megatron trainer to vLLM generation."""

from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


def check_mx_reshard_refit_support(master_config: Any) -> None:
    """Validate the deliberately narrow first production MX reshard path."""
    policy = master_config.policy
    generation = policy.get("generation", {}) or {}
    megatron = policy.get("megatron_cfg", {}) or {}
    dtensor = policy.get("dtensor_cfg", {}) or {}
    vllm = generation.get("vllm_cfg", {}) or {}
    violations: list[str] = []

    if generation.get("backend") != "vllm":
        violations.append("policy.generation.backend must be 'vllm'.")
    if generation.get("colocated", {}).get("enabled", False):
        violations.append("policy.generation.colocated.enabled must be False.")
    if not megatron.get("enabled", False):
        violations.append("policy.megatron_cfg.enabled must be True.")
    if dtensor.get("enabled", False):
        violations.append("policy.dtensor_cfg.enabled must be False.")
    if megatron.get("expert_tensor_parallel_size", 1) not in (None, 1):
        violations.append("policy.megatron_cfg.expert_tensor_parallel_size must be 1.")
    if str(vllm.get("kv_cache_dtype", "auto")).startswith("fp8"):
        violations.append("mx_reshard does not support FP8 KV-cache scale sync.")

    # Query heads still have to divide across trainer TP. KV heads do not:
    # Megatron slices the globally interleaved fused QKV rows and MX maps each
    # rank's raw interval, so KV<TP legitimately leaves most ranks with no K/V.
    tp = int(megatron.get("tensor_model_parallel_size", 1))
    heads = megatron.get("num_attention_heads")
    query_groups = megatron.get("num_query_groups")
    if heads is not None and int(heads) % tp:
        violations.append(
            "Megatron num_attention_heads must be divisible by tensor parallel size."
        )
    if query_groups is not None and int(query_groups) < 1:
        violations.append("Megatron num_query_groups must be positive.")
    if (
        heads is not None
        and query_groups is not None
        and int(query_groups) > 0
        and int(heads) % int(query_groups)
    ):
        violations.append(
            "Megatron num_attention_heads must be divisible by num_query_groups."
        )

    if violations:
        raise ValueError(
            "mx_reshard refit configuration is unsupported:\n- "
            + "\n- ".join(violations)
        )


class MxReshardWeightSynchronizer(WeightSynchronizer):
    """Serialize a stamped publish quorum before the matching receiver quorum."""

    def __init__(
        self,
        policy: Any,
        generation: Any,
        train_cluster: Any,
        inference_cluster: Any,
    ) -> None:
        self._policy = policy
        self._generation = generation
        self._train_cluster = train_cluster
        self._inference_cluster = inference_cluster
        self._version = 0
        self._stale = True
        self._shutdown = False

    @staticmethod
    def _require_all(results: Any, phase: str) -> None:
        leaves: list[Any] = []

        def flatten(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    flatten(item)
            else:
                leaves.append(value)

        flatten(results)
        if not leaves or not all(result is True for result in leaves):
            raise RuntimeError(
                f"ModelExpress mx_reshard {phase} failed on at least one rank: "
                f"{results!r}"
            )

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        if kv_scales is not None:
            raise ValueError("mx_reshard does not support FP8 KV-scale synchronization")

        # `_version` is the last *committed* version, so a failed refit leaves it
        # alone and the next attempt reuses this stamp. That is safe only because
        # a failure here aborts the run: publishers stamp the shard table with
        # this number, so a retry inside one process would advertise different
        # bytes under a version a receiver may already have seen. Anything that
        # adds a retry has to advance the counter per attempt instead.
        version = self._version + 1
        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )

        def phase(name: str):
            """Split the refit into its two serialized halves.

            Publish and receive are strictly sequential, so both sit on the
            critical path, but only the receive half reports MX_REFIT_STAGE
            telemetry. Without this split the publish cost is invisible: on
            Qwen3-30B-A3B it is roughly 8.4 s of an 11 s refit, which is far
            larger than the transfer MX does report, while on a dense 4B model it
            is only ~0.5 s. Attributing a refit therefore requires this timer, not
            just MX's stages.
            """
            if timer is None:
                return nullcontext()
            return timer.time(f"prepare_for_generation/mx_reshard_{name}")

        with timer_context:
            # This ordering is intentional and safety-critical. A receiver must
            # never discover a mixed fleet while some trainers still advertise
            # the previous version.
            with phase("publish"):
                published = ray.get(
                    self._policy.publish_mx_reshard_weights(version=version)
                )
                self._require_all(published, "publish")

            with phase("receive"):
                pulled = ray.get(
                    self._generation.update_weights_from_mx_reshard(version=version)
                )
                self._require_all(pulled, "receive")

        self._version = version
        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def init_communicator(self) -> None:
        from nemo_rl.distributed.mx_reshard_config import (
            validate_mx_reshard_listen_port_ranges,
        )

        train_world_size = self._train_cluster.world_size()
        inference_world_size = self._inference_cluster.world_size()
        refit_cfg = self._generation.cfg.get("refit_cfg") or {}
        mx_cfg = (
            refit_cfg.mx_reshard
            if hasattr(refit_cfg, "mx_reshard")
            else refit_cfg.get("mx_reshard", {})
        )
        validate_mx_reshard_listen_port_ranges(
            mx_cfg,
            train_world_size=train_world_size,
            inference_world_size=inference_world_size,
        )
        published = ray.get(
            self._policy.init_mx_reshard_publisher(train_world_size=train_world_size)
        )
        self._require_all(published, "publisher initialization")
        receivers = ray.get(
            self._generation.init_mx_reshard_receiver(
                train_world_size=train_world_size,
                inference_world_size=inference_world_size,
            )
        )
        self._require_all(receivers, "receiver initialization")

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        errors: list[Exception] = []
        try:
            # Receivers load publisher metadata and own the remote connection.
            # They must disconnect before the publisher destroys its local UCX
            # worker; reversing this order can abort in ucp_worker_destroy.
            receiver_results = ray.get(self._generation.shutdown_mx_reshard_receiver())
            self._require_all(receiver_results, "receiver shutdown")
        except Exception as error:
            errors.append(error)
        try:
            trainer_results = ray.get(self._policy.shutdown_mx_reshard_publisher())
            self._require_all(trainer_results, "publisher shutdown")
        except Exception as error:
            errors.append(error)
        if errors:
            raise RuntimeError(
                "ModelExpress mx_reshard shutdown failed: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]

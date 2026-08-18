# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Rank-local ModelExpress integration for Megatron policy workers."""

from __future__ import annotations

from typing import Any, Self

from nemo_rl.weight_sync.model_express.megatron import build_megatron_tensor_specs


class ModelExpressTrainerIntegration:
    """Connect Megatron tensors to the public MX trainer client."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._tensor_specs: list[Any] | None = None

    @classmethod
    def from_config(cls, *, config: dict[str, Any], local_rank: int) -> Self | None:
        """Preinitialize NIXL before Megatron creates NCCL resources."""
        if config.get("generation", {}).get("refit_transport") != "model_express":
            return None

        from modelexpress_rl import ModelExpressTrainerClient

        refit_config = config["generation"].get("refit_cfg")
        model_express_config = (
            refit_config.model_express if refit_config is not None else None
        )
        return cls(
            ModelExpressTrainerClient.initialize(
                model_name=config["model_name"],
                device_id=local_rank,
                server_url=(
                    model_express_config.server_url
                    if model_express_config is not None
                    else None
                ),
            )
        )

    def initialize(
        self,
        *,
        model_name: str,
        conversion_tasks: list[Any],
        transformer_config: Any,
        tensor_parallel_size: int,
        tensor_parallel_rank: int,
        server_url: str | None,
    ) -> str:
        """Register stable tensors after Megatron distributed setup completes."""
        specs = build_megatron_tensor_specs(
            conversion_tasks=conversion_tasks,
            transformer_config=transformer_config,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
        )
        self._client.register_tensors({spec.name: spec.tensor for spec in specs})
        self._tensor_specs = specs
        return self._client.source_slot_id

    def publish(self, version: Any) -> None:
        """Publish this rank's immutable shard for one MX version."""
        if self._tensor_specs is None:
            return
        staged = self._client.stage_shard(version=version, tensors=self._tensor_specs)
        staged.publish()

    def release(self, version: Any) -> None:
        """Release this rank's buffers after the version is retired."""
        self._client.release_version(version=version)

    def close(self) -> None:
        """Close all rank-local ModelExpress resources."""
        self._client.close()
        self._tensor_specs = None


__all__ = ["ModelExpressTrainerIntegration"]

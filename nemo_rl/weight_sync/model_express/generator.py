# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Rank-local ModelExpress integration for vLLM generator workers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig

    from modelexpress_rl import ModelExpressGeneratorClient, WeightVersionRef


class ModelExpressGeneratorIntegration:
    """Own the MX generator client and staged installation lifecycle."""

    def __init__(self, client: ModelExpressGeneratorClient) -> None:
        self._client = client

    @staticmethod
    def initialize(
        *,
        model: Module,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        server_url: str | None,
    ) -> ModelExpressGeneratorIntegration:
        # ModelExpress is optional and only imported when this transport is selected.
        try:
            from modelexpress_rl import ModelExpressGeneratorClient
        except ImportError as error:
            raise RuntimeError(
                "ModelExpress refit requires the modelexpress_rl package in "
                "the vLLM worker environment"
            ) from error
        client = ModelExpressGeneratorClient.initialize(
            model=model,
            vllm_config=vllm_config,
            model_config=model_config,
            model_name=model_config.model,
            server_url=server_url,
        )
        return ModelExpressGeneratorIntegration(client)

    def update(self, version: WeightVersionRef) -> bool:
        """Stage, verify, and install one exact MX version."""
        staged = self._client.stage_weight(version=version)
        try:
            staged.wait()
            self._client.apply_weight(staged)
        finally:
            staged.release()
        return True

    def close(self) -> None:
        self._client.close()


__all__ = ["ModelExpressGeneratorIntegration"]

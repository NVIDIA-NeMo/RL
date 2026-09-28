# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small deterministic workloads; checkpoint coordination stays in Gym and RL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from pydantic import BaseModel, Field

from nemo_rl.weight_sync.interfaces import WeightSynchronizer


def tensor_bytes(value: torch.Tensor) -> bytes:
    return bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist())


def weight_digest(weights: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(weights)).hexdigest()


def logprobs_for(weights: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    vocabulary = torch.arange(257)
    logits = -(vocabulary.float() + weights[vocabulary % len(weights)].float()) / 512
    return logits.log_softmax(dim=0)[tokens]


@runtime_checkable
class WeightSource(Protocol):
    def export_weights(self) -> torch.Tensor: ...


@runtime_checkable
class WeightTarget(Protocol):
    def accept_weights(self, weights: torch.Tensor) -> str: ...


@dataclass(frozen=True)
class Turn:
    prompt: str
    sibling: int
    index: int
    seconds: float


@dataclass(frozen=True)
class Generated:
    text: str
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    weight_digest: str


class Policy:
    class Config(BaseModel, extra="forbid"):
        train_seconds: float = Field(default=0, ge=0)

    def __init__(self, config: Config | None = None):
        self.config = config if config is not None else self.Config()
        self.steps = 0
        self._weights = torch.arange(32, dtype=torch.uint8)

    def export_weights(self) -> torch.Tensor:
        return self._weights.clone()

    def logprobs(self, tokens: torch.Tensor) -> torch.Tensor:
        return logprobs_for(self._weights, tokens)

    def train(self, batch: Mapping[str, torch.Tensor]) -> None:
        digest = hashlib.sha256(tensor_bytes(self._weights))
        for field in ("input_ids", "input_lengths", "token_mask", "total_reward"):
            tensor = batch[field]
            digest.update(
                json.dumps([field, str(tensor.dtype), list(tensor.shape)]).encode()
            )
            digest.update(tensor_bytes(tensor))
        time.sleep(self.config.train_seconds)
        self._weights = torch.tensor(list(digest.digest()), dtype=torch.uint8)
        self.steps += 1

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: str | None = None,
        tokenizer_path: str | None = None,
        *,
        is_final_checkpoint: bool,
    ) -> None:
        path = Path(weights_path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": self._weights, "steps": self.steps}, path / "mock.pt")

    def load_checkpoint(self, weights_path: Path) -> None:
        state = torch.load(
            weights_path / "mock.pt", map_location="cpu", weights_only=True
        )
        self._weights = state["weights"]
        self.steps = state["steps"]


class Generation:
    class Config(BaseModel, extra="forbid"):
        pass

    def __init__(self, config: Config | None = None):
        self._weights: torch.Tensor | None = None

    def accept_weights(self, weights: torch.Tensor) -> str:
        if weights.dtype != torch.uint8 or weights.shape != (32,):
            raise ValueError("Generation expects 32 uint8 weights")
        self._weights = weights.detach().cpu().clone()
        return weight_digest(self._weights)

    async def generate(self, turn: Turn) -> Generated:
        if self._weights is None:
            raise RuntimeError("Refit must complete before generation")
        weights = self._weights.clone()
        identity = json.dumps([turn.prompt, turn.sibling, turn.index]).encode()
        tokens = tuple(byte + 1 for byte in hashlib.sha256(identity).digest()[:4])
        probs = logprobs_for(weights, torch.tensor(tokens)).tolist()
        await asyncio.sleep(turn.seconds)
        return Generated(
            text=f"{turn.prompt}/{turn.sibling}/{turn.index}",
            token_ids=tokens,
            logprobs=tuple(probs),
            weight_digest=weight_digest(weights),
        )


class CopyRefit(WeightSynchronizer):
    class Config(BaseModel, extra="forbid"):
        pass

    def __init__(
        self,
        policy: WeightSource,
        generation: WeightTarget,
        config: Config | None = None,
    ):
        if not isinstance(policy, WeightSource) or not isinstance(
            generation, WeightTarget
        ):
            raise TypeError("CopyRefit requires export_weights and accept_weights")
        self.policy = policy
        self.generation = generation
        self._is_stale = True

    @property
    def is_stale(self) -> bool:
        return self._is_stale

    def sync_weights(self, *, timer=None, kv_scales=None):
        if kv_scales is not None:
            raise ValueError("CPU copy refit does not support KV scales")
        self._is_stale = True
        weights = self.policy.export_weights()
        if self.generation.accept_weights(weights) != weight_digest(weights):
            raise RuntimeError("Generation did not acknowledge the copied weights")
        self._is_stale = False

    def init_communicator(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

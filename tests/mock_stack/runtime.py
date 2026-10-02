# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU adapters for the controller's existing handles."""

from __future__ import annotations

import socket
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import uvicorn

from nemo_rl.data_plane import build_data_plane_client
from nemo_rl.data_plane.column_io import read_columns, write_columns
from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from tests.mock_stack.servers import GenerationServer
from tests.mock_stack.tracing import TraceRun


@runtime_checkable
class TrainablePolicy(Protocol):
    def train(self, batch: dict[str, torch.Tensor]) -> None: ...
    def logprobs(self, tokens: torch.Tensor) -> torch.Tensor: ...
    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: str | None,
        tokenizer_path: str | None,
        *,
        is_final_checkpoint: bool,
    ) -> None: ...
    def load_checkpoint(self, weights_path: Path) -> None: ...


class Trainer:
    def __init__(
        self,
        policy: TrainablePolicy,
        plane: DataPlaneClient,
        trace: TraceRun | None = None,
    ):
        self.policy = policy
        self.plane = plane
        self.trace = trace
        self.batches: list[tuple[list[str], dict[str, torch.Tensor]]] = []
        self._pending: tuple[list[str], dict[str, torch.Tensor]] | None = None
        self._step_open = False

    def begin_train_step(self, loss_fn) -> None:
        if self._step_open:
            raise RuntimeError("A training step is already open")
        self._step_open = True

    def train_microbatches_from_meta(self, meta: KVBatchMeta, *, train_fields) -> None:
        if not self._step_open or self._pending is not None:
            raise RuntimeError("CPU trainer supports one prompt group per step")
        batch = read_columns(
            self.plane,
            meta,
            [
                "input_ids",
                "input_lengths",
                "token_mask",
                "total_reward",
                "generation_logprobs",
            ],
        )
        self._pending = (
            list(meta.sample_ids),
            {k: v.clone() for k, v in batch.items()},
        )

    def finish_train_step(self) -> dict:
        if self._pending is None:
            raise RuntimeError("Training step has no batch")
        with (
            self.trace.span(
                "test.policy.train", **{"test.sample_ids": self._pending[0]}
            )
            if self.trace
            else nullcontext()
        ):
            self.policy.train(self._pending[1])
        self.batches.append(self._pending)
        self._pending = None
        self._step_open = False
        return {"loss": 0.0, "grad_norm": 0.0, "all_mb_metrics": {}}

    def save_checkpoint(self, *args, **kwargs) -> None:
        self.policy.save_checkpoint(*args, **kwargs)

    def load_data_plane_checkpoint(self, checkpoint_dir):
        return self.plane.load_checkpoint(checkpoint_dir)

    def get_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        batch = read_columns(self.plane, meta, ["input_ids", "input_lengths"])
        write_columns(
            self.plane,
            meta,
            {"prev_logprobs": self.policy.logprobs(batch["input_ids"])},
        )

    def prepare_for_lp_inference(self, *, keep_train_buffers: bool) -> None:
        pass

    def finish_inference(self) -> None:
        pass

    def offload_to_cpu(self) -> None:
        pass

    def prepare_for_training(self) -> None:
        pass

    def sync_params_before_refit(self) -> None:
        pass

    def offload_before_refit(self) -> None:
        pass

    def offload_after_refit(self) -> None:
        pass

    def finalize_async_save(self, **kwargs) -> None:
        pass


class GenerationHandle:
    def __init__(self, server: GenerationServer):
        self.server = server
        self.weight_synchronizer: WeightSynchronizer | None = None
        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self.dp_openai_server_base_urls = [
            f"http://127.0.0.1:{self._socket.getsockname()[1]}/v1"
        ]
        self._http = uvicorn.Server(
            uvicorn.Config(server.app, log_level="error", lifespan="off")
        )
        self._thread = threading.Thread(
            target=self._http.run, kwargs={"sockets": [self._socket]}, daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._http.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                self.close()
                raise RuntimeError("CPU generation HTTP server did not start")
            time.sleep(0.01)

    def close(self) -> None:
        self._http.should_exit = True
        self._thread.join(timeout=10)
        self._socket.close()
        if self._thread.is_alive():
            raise RuntimeError("CPU generation HTTP server did not stop")

    def setup_token_capture(self, dp_config, partition) -> None:
        self.server.setup_token_capture(
            build_data_plane_client(dp_config, bootstrap=False), partition
        )

    def set_rollout_weight_version(self, version: int) -> None:
        self.server.set_rollout_weight_version(version)

    def load_and_start(self) -> None:
        pass

    def blocks_training(self) -> bool:
        return False

    def wake_carries_weight_updates(self) -> bool:
        return False

    def finish_generation(self) -> None:
        pass

    def prepare_for_generation(self) -> None:
        pass

    def invalidate_kv_cache(self) -> None:
        pass

    def snapshot_step_metrics(self) -> None:
        pass

    def get_step_metrics(self) -> dict[str, float]:
        return {}

    def drain_latest_logger_metrics(self) -> dict:
        return {}

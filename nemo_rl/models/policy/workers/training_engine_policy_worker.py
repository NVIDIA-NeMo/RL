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

"""Policy worker that runs the model through Megatron-Bridge's TrainingEngine."""

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import ray
import torch
from megatron.bridge.engine import AdamParams, TrainingEngine
from megatron.core.transformer.moe.router import Router

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.numa_utils import bind_to_gpu_numa
from nemo_rl.distributed.refit_watchdog import RefitAborted, RefitAbortWatchdog
from nemo_rl.models.megatron.setup import setup_distributed
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
)
from nemo_rl.models.policy.training_engine_adapter.engine import (
    NeMoRLTrainingEngine,
    build_recipe,
    check_supported_loss,
)
from nemo_rl.models.policy.utils import (
    get_runtime_env_for_policy_worker,
    stream_weights_via_ipc_zmq_impl,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.utils.packed_tensor import packed_broadcast_producer


def _freeze_moe_routers(model: list[Any]) -> list[Any]:
    """Pre-wrap hook: stop gradient updates to every MoE router."""
    for chunk in model:
        for module in chunk.modules():
            if isinstance(module, Router):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
    return model


@dataclass
class _TrainState:
    """The loss function and running totals of one optimizer update while its chunks arrive."""

    loss_fn: LossFunction
    # Totals over all ranks.
    loss_sum: float = 0.0
    num_loss_tokens: int = 0
    metric_sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    # Samples with at least one loss token on this data-parallel replica.
    num_local_loss_seqs: int = 0


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
class TrainingEnginePolicyWorkerImpl(TQWorkerMixin, AbstractPolicyWorker):
    """Owns a trainable policy engine and a frozen reference engine on one GPU.

    The driver plans the packed microbatches and sends each data-parallel replica its
    own samples with that plan; every rank of a replica receives the same samples. The
    worker hands them to the engines as they are; the engines pack each planned
    microbatch with NeMo RL's microbatch processing.
    All workers must enter each call together: the engines' collectives span all ranks.
    """

    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs."""
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        return f"{self.__class__.__qualname__}"

    @staticmethod
    def configure_worker(
        num_gpus: int | float,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
        num_gpus_per_node: Optional[int] = None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
        """Worker-controlled Ray actor configuration (one GPU per worker)."""
        del bundle_indices, num_gpus_per_node
        resources: dict[str, Any] = {"num_gpus": num_gpus}
        env_vars: dict[str, str] = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
        return resources, env_vars, {}, {}

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: Any,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        processor: Optional[Any] = None,
        *,
        worker_sharding_annotations: NamedSharding,
        is_refit_destination: bool = False,
        **kwargs: Any,
    ):
        """Build the engines from ``policy.training_engine``."""
        # optimizer_path is not used: an engine checkpoint always holds the optimizer state.
        del tokenizer, processor, optimizer_path, worker_sharding_annotations, kwargs
        if not init_optimizer:
            # TODO: an inference-only policy is one frozen engine.
            raise NotImplementedError(
                "A TrainingEngine policy without an optimizer is not implemented yet."
            )
        if is_refit_destination:
            # TODO: receiving weights needs an engine import path.
            raise NotImplementedError(
                "A TrainingEngine policy cannot be a refit destination yet."
            )
        megatron_cfg = config["megatron_cfg"]
        scheduler_cfg = megatron_cfg["scheduler"]
        if scheduler_cfg["lr_decay_style"] not in (
            "constant",
            "linear",
            "cosine",
            "inverse-square-root",
        ):
            raise ValueError(
                f"Unsupported learning-rate decay style: {scheduler_cfg['lr_decay_style']!r}."
            )
        self._lr_decay_iters = scheduler_cfg.get("lr_decay_iters")
        if scheduler_cfg["lr_decay_style"] != "constant":
            if self._lr_decay_iters is None:
                self._lr_decay_iters = megatron_cfg["train_iters"]
            if not 0 <= scheduler_cfg["lr_warmup_iters"] < self._lr_decay_iters:
                raise ValueError(
                    "Learning-rate decay requires 0 <= lr_warmup_iters < lr_decay_iters."
                )
            optimizer_cfg = megatron_cfg["optimizer"]
            if not 0 <= optimizer_cfg["min_lr"] <= optimizer_cfg["lr"]:
                raise ValueError("Learning-rate decay requires 0 <= min_lr <= lr.")
        if scheduler_cfg["start_weight_decay"] != scheduler_cfg["end_weight_decay"]:
            # TODO: weight decay schedules.
            raise NotImplementedError(
                "The TrainingEngine policy needs a constant weight decay "
                "(scheduler.start_weight_decay == scheduler.end_weight_decay)."
            )

        # Must be the first CUDA-touching call in this process: configure_worker leaves
        # every device visible, so the worker selects its own.
        local_rank = int(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)
        bind_to_gpu_numa(local_rank)

        self.cfg = config
        self.rank = int(os.environ["RANK"])
        # [WAR] Triton's kernel cache defaults to the home directory, which is shared storage:
        # ranks compiling the same kernel at once race on its files. One local cache per rank.
        os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton-cache-rank{self.rank}")
        # The engine attaches to this process group instead of creating its own.
        setup_distributed(config)

        # The policy config was aligned with the recipe on the driver
        # (align_policy_config_with_recipe), so nothing is checked against it here.
        engine_config = config["training_engine"]
        recipe = build_recipe(engine_config)
        checkpoint_path = engine_config["checkpoint_path"]
        sample_pad_multiple = config["make_sequence_length_divisible_by"]
        hooks = [_freeze_moe_routers] if megatron_cfg["freeze_moe_router"] else []
        self.engine = NeMoRLTrainingEngine.from_recipe(
            recipe,
            sample_pad_multiple=sample_pad_multiple,
            checkpoint_path=checkpoint_path,
            hooks=hooks,
            spawn_async_save_worker=False,
        )
        self.reference_engine: Optional[TrainingEngine] = None
        if init_reference_model:
            self.reference_engine = NeMoRLTrainingEngine.from_recipe(
                recipe,
                sample_pad_multiple=sample_pad_multiple,
                checkpoint_path=checkpoint_path,
                trainable=False,
            )
        # train() and finish_train_step() calls so far; the learning-rate schedule
        # advances once per call.
        self._train_calls = 0
        self._train_state: Optional[_TrainState] = None
        if weights_path is not None:
            # Resume: the reference engine keeps the initial weights loaded above.
            saved = self.engine.load_state(str(weights_path))
            self._train_calls = saved["train_calls"]

    # ---- position among the workers, for the data-plane entry points ----

    def _local_coords(self) -> dict[str, int]:
        """This worker's position on the axes that share one replica's samples."""
        info = self.engine.parallel_info
        return {
            "tensor_parallel": info["tp_rank"],
            "context_parallel": info["cp_rank"],
            "pipeline_parallel": info["pp_rank"],
        }

    def _get_replica_group(self) -> Any:
        """The ranks that receive the same samples as this worker.

        Pipeline parallelism is rejected by the config check, so these are the
        worker's tensor-parallel and context-parallel partners.
        """
        return self.engine.process_groups.tp_cp

    # ---- scoring ----

    def _score(
        self, engine: TrainingEngine, data: BatchedDataDict[Any]
    ) -> torch.Tensor:
        """Target logprobs of this replica's samples as ``[B, S]``, in the order received."""
        return engine.forward_only(data, gather_output=True).cpu()

    # 'data' is keyword-only like the other policy workers (a Ray argument-passing quirk).
    def get_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Score this replica's samples with the policy.

        ``micro_batch_size`` is not used: the microbatches follow the packing plan
        on ``data``.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length],
          where position i holds the logprob of input token i and the first token's is 0.
        """
        del micro_batch_size
        return BatchedDataDict[LogprobOutputSpec](
            logprobs=self._score(self.engine, data)
        )

    def get_reference_policy_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Score this replica's samples with the frozen reference model.

        Returns: as get_logprobs, under the key "reference_logprobs".
        """
        del micro_batch_size
        if self.reference_engine is None:
            raise RuntimeError(
                "The reference model was not built (init_reference_model=False)."
            )
        return BatchedDataDict[ReferenceLogprobOutputSpec](
            reference_logprobs=self._score(self.reference_engine, data)
        )

    # ---- training ----

    def _adam_params(self) -> AdamParams:
        """Optimizer settings for the current train() call, from policy.megatron_cfg."""
        optimizer_cfg = self.cfg["megatron_cfg"]["optimizer"]
        scheduler_cfg = self.cfg["megatron_cfg"]["scheduler"]
        learning_rate = optimizer_cfg["lr"]
        warmup = scheduler_cfg["lr_warmup_iters"]
        if warmup > 0 and self._train_calls <= warmup:
            start = scheduler_cfg["lr_warmup_init"]
            learning_rate = start + (learning_rate - start) * self._train_calls / warmup
        elif scheduler_cfg["lr_decay_style"] != "constant":
            min_lr = optimizer_cfg["min_lr"]
            assert self._lr_decay_iters is not None
            if self._train_calls > self._lr_decay_iters:
                learning_rate = min_lr
            elif scheduler_cfg["lr_decay_style"] == "inverse-square-root":
                learning_rate = max(
                    min_lr,
                    learning_rate
                    * math.sqrt(max(warmup, 1))
                    / math.sqrt(max(self._train_calls, 1)),
                )
            else:
                progress = (self._train_calls - warmup) / (
                    self._lr_decay_iters - warmup
                )
                if scheduler_cfg["lr_decay_style"] == "linear":
                    coefficient = 1.0 - progress
                else:
                    coefficient = 0.5 * (math.cos(math.pi * progress) + 1.0)
                learning_rate = min_lr + (learning_rate - min_lr) * coefficient
        return AdamParams(
            learning_rate=learning_rate,
            beta1=optimizer_cfg["adam_beta1"],
            beta2=optimizer_cfg["adam_beta2"],
            eps=optimizer_cfg["adam_eps"],
            weight_decay=optimizer_cfg["weight_decay"],
        )

    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        check_dim_skip_keys: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        """Run one optimizer update per global batch of the driver's plan.

        ``data`` holds this replica's share of every global batch. ``mbs`` is not used:
        the microbatches follow the packing plan on ``data``.

        Returns:
          "global_loss" (mean loss per loss token, averaged over the updates), "grad_norm",
          and "all_mb_metrics" with one entry per update for each metric. Token-level
          metrics are per-token means of the update over all replicas.
        """
        del mbs
        if eval_mode:
            # TODO: a forward with the loss and no gradient accumulation.
            raise NotImplementedError(
                "eval_mode is not implemented for the TrainingEngine policy yet."
            )
        if check_dim_skip_keys is not None:
            raise ValueError(
                "check_dim_skip_keys is only supported by the Automodel worker."
            )
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        local_gbs = gbs // self.engine.parallel_info["dp_size"]
        # Every update of one train() call uses the same schedule position.
        adam_params = self._adam_params()
        updates = []
        for index in range(len(data.micro_batch_indices)):
            self.begin_train_step(loss_fn)
            self.train_microbatch(data=data.get_batch(index, local_gbs))
            updates.append(self._apply_update(adam_params))
        self._train_calls += 1
        return self._train_result(updates)

    # ---- streaming train step ----
    # One optimizer update fed in several chunks: begin_train_step, train_microbatch once
    # per chunk, then finish_train_step. Gradients add up in the engine across the chunks
    # and are normalized once, by the loss tokens of the whole update.

    def begin_train_step(
        self,
        loss_fn: LossFunction,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> None:
        """Open an update. ``gbs`` and ``mbs`` are not used: each chunk brings its plan."""
        del gbs, mbs
        check_supported_loss(loss_fn)
        if self._train_state is not None:
            raise RuntimeError(
                "A train step is already open; call finish_train_step or "
                "abort_train_step first."
            )
        self._train_state = _TrainState(loss_fn=loss_fn)

    def train_microbatch(self, *, data: BatchedDataDict[Any]) -> None:
        """Add this replica's share of one chunk to the open update (forward and backward only)."""
        state = self._open_train_state("train_microbatch")
        output = self.engine.forward_backward(
            data,
            loss_fn="clipped_pg_loss",
            loss_kwargs={"loss_fn": state.loss_fn},
        )
        state.loss_sum += output.loss_sum
        state.num_loss_tokens += output.num_loss_tokens
        for name, total in output.metrics.items():
            state.metric_sums[name] += total
        loss_mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
        state.num_local_loss_seqs += int(loss_mask[:, 1:].bool().any(dim=-1).sum())

    def finish_train_step(self) -> dict[str, Any]:
        """Apply the open update and advance the learning-rate schedule by one.

        Returns: as train(), for the one update.
        """
        update = self._apply_update(self._adam_params())
        self._train_calls += 1
        return self._train_result([update])

    def abort_train_step(self) -> None:
        """Discard the open update, if any, without touching the weights."""
        if self._train_state is None:
            return
        self._train_state = None
        self.engine.zero_grad()

    def pause_train_step_with_offloading(self) -> None:
        """Move the policy engine, with the open update's gradients, to host memory.

        prepare_for_training brings it back.
        """
        self._open_train_state("pause_train_step_with_offloading")
        self.engine.cpu()

    def _open_train_state(self, caller: str) -> "_TrainState":
        if self._train_state is None:
            raise RuntimeError(f"{caller} needs an open train step.")
        return self._train_state

    def _apply_update(self, adam_params: AdamParams) -> dict[str, float]:
        """Step the optimizer on the open update; return its metrics and grad norm."""
        state = self._open_train_state("finish_train_step")
        step = self.engine.optim_step(adam_params)
        self._train_state = None
        tokens = max(state.num_loss_tokens, 1)
        update = {name: total / tokens for name, total in state.metric_sums.items()}
        update["loss"] = state.loss_sum / tokens
        update["lr"] = adam_params.learning_rate
        update["wd"] = adam_params.weight_decay
        update["global_valid_toks"] = state.num_loss_tokens
        update["global_valid_seqs"] = self._sum_over_replicas(state.num_local_loss_seqs)
        update["update_successful"] = step["update_successful"]
        update["grad_norm"] = step["grad_norm"]
        return update

    def _sum_over_replicas(self, count: int) -> int:
        """Add up a per-replica count over the data-parallel replicas."""
        total = torch.tensor([count], device="cuda")
        torch.distributed.all_reduce(total, group=self.engine.process_groups.dp)
        return int(total.item())

    def _train_result(self, updates: list[dict[str, float]]) -> dict[str, Any]:
        """The worker's training result for a list of updates."""
        metrics: dict[str, list[float]] = defaultdict(list)
        for update in updates:
            for name, value in update.items():
                metrics[name].append(value)
        grad_norms = metrics.pop("grad_norm")
        losses = metrics["loss"]
        # The metrics are already totals over all ranks, and the driver adds up what the
        # replicas report, so only the first replica reports them.
        is_first_replica = self.engine.parallel_info["dp_rank"] == 0
        return {
            "global_loss": torch.tensor(sum(losses) / len(losses)),
            "grad_norm": torch.tensor(grad_norms),
            "all_mb_metrics": dict(metrics) if is_first_replica else {},
        }

    # ---- model residency ----
    # The colocated generation engine shares this GPU, so only what the next call needs
    # stays on it. Moving an engine that is already in place is a no-op.

    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        """Bring both engines to the GPU for scoring."""
        del keep_train_buffers
        self.engine.cuda()
        if self.reference_engine is not None:
            self.reference_engine.cuda()

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        """Keep only the policy engine on the GPU."""
        if self.reference_engine is not None:
            self.reference_engine.cpu()
        self.engine.cuda()

    def offload_before_refit(self) -> None:
        """Keep only the policy engine on the GPU: the export reads its weights there."""
        if self.reference_engine is not None:
            self.reference_engine.cpu()
        self.engine.cuda()

    def offload_after_refit(self) -> None:
        """Move both engines to host memory to make room for generation."""
        self.engine.cpu()
        if self.reference_engine is not None:
            self.reference_engine.cpu()
        torch.cuda.empty_cache()

    # ---- weight transfer to the generation engine ----

    def prepare_refit_info(self) -> dict[str, tuple[torch.Size, torch.dtype]]:
        """Shape and dtype of every weight a refit sends, by HF name."""
        return {
            name: (tensor.shape, tensor.dtype)
            for name, tensor in self.engine.export(streaming=True)
        }

    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Stream the policy's HF weights to the generation process on this GPU."""
        if kv_scales is not None:
            # TODO: append the FP8 KV-cache scales to the stream.
            raise NotImplementedError(
                "FP8 KV-cache scales are not supported by the TrainingEngine policy yet."
            )
        self.maybe_init_zmq()
        stream_weights_via_ipc_zmq_impl(
            params_generator=self.engine.export(streaming=True),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self,
        kv_scales: Optional[dict[str, float]] = None,
        refit_timeout_s: Optional[float] = None,
        *,
        buffer_size_bytes: Optional[int] = None,
        num_buffers: Optional[int] = None,
        use_legacy_packing: bool = False,
    ) -> None:
        """Broadcast the policy's HF weights to generation workers on other GPUs.

        Every training rank runs the export and joins the broadcast; rank 0 is the source.
        The watchdog aborts a broadcast that a dead generation rank would block forever.
        """
        if kv_scales is not None:
            # TODO: append the FP8 KV-cache scales to the stream.
            raise NotImplementedError(
                "FP8 KV-cache scales are not supported by the TrainingEngine policy yet."
            )
        # The export reads the weights on the GPU.
        self.engine.cuda()
        with RefitAbortWatchdog(self.model_update_group, refit_timeout_s) as guard:
            packed_broadcast_producer(
                iterator=self.engine.export(streaming=True),
                group=self.model_update_group,
                src=0,
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=buffer_size_bytes,
                num_buffers=num_buffers,
                use_legacy_packing=use_legacy_packing,
            )
        if guard.fired:
            raise RefitAborted(
                f"refit broadcast exceeded {refit_timeout_s}s and was aborted; "
                "a generation rank most likely stopped participating"
            )

    # ---- checkpointing ----

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        """Save weights, optimizer state and the schedule position under ``weights_path``.

        Blocks until the checkpoint is on disk. The tokenizer is not saved.
        """
        del tokenizer_path
        if optimizer_path is None:
            # TODO: a weights-only engine checkpoint.
            raise NotImplementedError(
                "The TrainingEngine policy always saves the optimizer state; set "
                "checkpointing.save_optimizer=true."
            )
        # Generation may have left the engine in host memory; the save reads it on the GPU.
        was_offloaded = self.engine.device == "cpu"
        self.engine.cuda()
        self.engine.save_state(
            str(weights_path), user_state={"train_calls": self._train_calls}
        )
        if was_offloaded:
            self.engine.cpu()


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("training_engine_policy_worker")
)  # pragma: no cover
class TrainingEnginePolicyWorker(TrainingEnginePolicyWorkerImpl):
    pass

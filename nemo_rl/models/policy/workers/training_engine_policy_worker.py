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
from typing import Any, Optional

import megatron.bridge.engine.recipes as engine_recipes
import ray
import torch
from megatron.bridge.engine import AdamParams, TrainingEngine
from megatron.core.transformer.moe.router import Router

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.numa_utils import bind_to_gpu_numa
from nemo_rl.models.megatron.setup import setup_distributed
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
)
from nemo_rl.models.policy.training_engine_adapter.data import (
    padded_scores,
    scoring_datums,
    training_datums,
)
from nemo_rl.models.policy.training_engine_adapter.losses import NeMoRLTrainingEngine
from nemo_rl.models.policy.utils import (
    get_runtime_env_for_policy_worker,
    stream_weights_via_ipc_zmq_impl,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker

# Engine topology key for each axis of the policy's sharding annotations.
_TOPOLOGY_AXES = {
    "tensor_parallel": "tp_size",
    "context_parallel": "cp_size",
    "pipeline_parallel": "pp_size",
    "data_parallel": "dp_size",
}


def _freeze_moe_routers(model: list[Any]) -> list[Any]:
    """Pre-wrap hook: stop gradient updates to every MoE router."""
    for chunk in model:
        for module in chunk.modules():
            if isinstance(module, Router):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
    return model


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
class TrainingEnginePolicyWorkerImpl(AbstractPolicyWorker):
    """Owns a trainable policy engine and a frozen reference engine on one GPU.

    Every worker receives the same global batch for every call. The engines pack it,
    assign it to data-parallel replicas and split it across context-parallel ranks, so
    all workers must enter each call together. Results are identical on every rank;
    rank 0 returns them and the others return None.
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
        del tokenizer, processor, kwargs
        if weights_path is not None or optimizer_path is not None:
            # TODO: resume through engine.load_state.
            raise NotImplementedError(
                "Resuming a TrainingEngine policy from a checkpoint is not implemented yet."
            )
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
        # Triton's kernel cache defaults to the home directory, which is shared storage:
        # ranks compiling the same kernel at once race on its files. One local cache per rank.
        os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton-cache-rank{self.rank}")
        # The engine attaches to this process group instead of creating its own.
        setup_distributed(config)

        engine_config = config["training_engine"]
        recipe = getattr(engine_recipes, engine_config["recipe"])(
            seq_length=engine_config["seq_length"], **engine_config["recipe_kwargs"]
        )
        engine_recipes.apply_recipe_overrides(recipe, engine_config["overrides"])
        checkpoint_path = engine_config["checkpoint_path"]
        hooks = [_freeze_moe_routers] if megatron_cfg["freeze_moe_router"] else []
        self.engine = NeMoRLTrainingEngine.from_recipe(
            recipe,
            checkpoint_path=checkpoint_path,
            hooks=hooks,
            spawn_async_save_worker=False,
        )
        self._check_topology(worker_sharding_annotations)
        self.reference_engine: Optional[TrainingEngine] = None
        if init_reference_model:
            self.reference_engine = TrainingEngine.from_recipe(
                recipe, checkpoint_path=checkpoint_path, trainable=False
            )
        # train() calls so far; the learning-rate schedule advances once per call.
        self._train_calls = 0

    def _check_topology(self, annotations: NamedSharding) -> None:
        """The driver lays workers out from the policy config; the engine from its recipe."""
        info = self.engine.parallel_info
        for axis, key in _TOPOLOGY_AXES.items():
            if annotations.get_axis_size(axis) != info[key]:
                raise ValueError(
                    f"The policy config gives {axis} size "
                    f"{annotations.get_axis_size(axis)}, but the engine recipe "
                    f"resolves to {info[key]}. Set policy.megatron_cfg to the "
                    "recipe's topology."
                )

    # ---- scoring ----

    def _score(
        self, engine: TrainingEngine, data: BatchedDataDict[Any]
    ) -> Optional[torch.Tensor]:
        """Target logprobs as ``[B, S]`` on rank 0."""
        output = engine.forward_only(scoring_datums(data), gather_output=True)
        if self.rank != 0:
            return None
        return padded_scores(output.logprobs, data["input_ids"].shape[1])

    # 'data' is keyword-only like the other policy workers (a Ray argument-passing quirk).
    def get_logprobs(
        self, *, data: BatchedDataDict[Any]
    ) -> Optional[BatchedDataDict[LogprobOutputSpec]]:
        """Score the global batch with the policy.

        Returns:
          On rank 0, a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length],
          where position i holds the logprob of input token i and the first token's is 0.
        """
        logprobs = self._score(self.engine, data)
        if logprobs is None:
            return None
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs)

    def get_reference_policy_logprobs(
        self, *, data: BatchedDataDict[Any]
    ) -> Optional[BatchedDataDict[ReferenceLogprobOutputSpec]]:
        """Score the global batch with the frozen reference model.

        Returns: as get_logprobs, under the key "reference_logprobs".
        """
        if self.reference_engine is None:
            raise RuntimeError(
                "The reference model was not built (init_reference_model=False)."
            )
        logprobs = self._score(self.reference_engine, data)
        if logprobs is None:
            return None
        return BatchedDataDict[ReferenceLogprobOutputSpec](reference_logprobs=logprobs)

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
        *,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        gbs: int,
    ) -> Optional[dict[str, Any]]:
        """Run one optimizer update per ``gbs`` samples of the global batch.

        Returns:
          On rank 0: "global_loss" (mean loss per loss token, averaged over the updates),
          "grad_norm", and "all_mb_metrics" with one entry per update for each metric.
          Token-level metrics are per-token means of the update.
        """
        if data.size % gbs != 0:
            raise ValueError(
                f"The batch has {data.size} samples, not a multiple of the global "
                f"batch size {gbs}."
            )
        adam_params = self._adam_params()
        metrics: dict[str, list[float]] = defaultdict(list)
        grad_norms = []
        for start in range(0, data.size, gbs):
            update = data.slice(start, start + gbs)
            output = self.engine.forward_backward(
                training_datums(update),
                loss_fn="clipped_pg_loss",
                loss_kwargs={"loss_fn": loss_fn},
            )
            step = self.engine.optim_step(adam_params)
            tokens = max(output.num_loss_tokens, 1)
            metrics["loss"].append(output.loss_sum / tokens)
            for name, total in output.metrics.items():
                metrics[name].append(total / tokens)
            metrics["lr"].append(adam_params.learning_rate)
            metrics["wd"].append(adam_params.weight_decay)
            metrics["global_valid_toks"].append(output.num_loss_tokens)
            loss_mask = update["token_mask"] * update["sample_mask"].unsqueeze(-1)
            metrics["global_valid_seqs"].append(
                int(loss_mask[:, 1:].bool().any(dim=-1).sum())
            )
            metrics["update_successful"].append(step["update_successful"])
            grad_norms.append(step["grad_norm"])
        self._train_calls += 1
        if self.rank != 0:
            return None
        losses = metrics["loss"]
        return {
            "global_loss": torch.tensor(sum(losses) / len(losses)),
            "grad_norm": torch.tensor(grad_norms),
            "all_mb_metrics": dict(metrics),
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

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Save a training checkpoint."""
        # TODO: engine.save_state plus the train() call count, and the matching resume.
        raise NotImplementedError(
            "Checkpointing a TrainingEngine policy is not implemented yet; run with "
            "checkpointing.enabled=false."
        )


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("training_engine_policy_worker")
)  # pragma: no cover
class TrainingEnginePolicyWorker(TrainingEnginePolicyWorkerImpl):
    pass

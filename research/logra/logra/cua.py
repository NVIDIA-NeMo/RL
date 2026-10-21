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

"""CUA-specific task baselines and finite-difference KL step control.

These helpers are independent of the rollout backend. The worker supplies a
log-probability iterator over fixed probe contexts; the controller evaluates the
same contexts before a trial update, during the trial, and after the real update.
"""

import copy
import math
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

import torch
import torch.distributed as dist
from pydantic import BaseModel, Field
from torch import Tensor

from logra.compression import local_weight_rows
from logra.optimizer import LoGRAOptimizer

if TYPE_CHECKING:
    from logra.worker import LoGRAWorkerImpl
    from nemo_rl.algorithms.loss.interfaces import LossFunction
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
    from nemo_rl.models.automodel.data import ProcessedMicrobatch
    from nemo_rl.models.automodel.router_replay import ReplayGate


class CuaConfig(BaseModel, extra="forbid"):
    """Defaults mirror the historical CUA implementation; override in the recipe."""

    kl_budget: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 5e-4
    # None keeps the original constant budget; otherwise cosine anneal by merge count.
    kl_budget_final: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    kl_budget_steps: Annotated[int, Field(gt=0)] = 600
    max_update_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 6e-5
    alpha_max: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 3.0
    probe_sequences: Annotated[int, Field(gt=0)] = 2
    probe_tokens: Annotated[int, Field(gt=0)] = 2048
    check_realized_kl: bool = True
    baseline_prior: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 2.0
    baseline_ema: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.1
    baseline_by_template: bool = True
    length_normalization: bool = True
    length_normalization_reference: Annotated[float, Field(gt=0)] = 1024.0
    routing_replay: bool = True


def kl_budget_at(config: CuaConfig, completed_updates: int) -> float:
    """Budget for the next merge, using the optimizer's checkpointed update count."""
    if completed_updates < 0:
        raise ValueError("Completed updates must be nonnegative")
    if config.kl_budget_final is None:
        return config.kl_budget
    fraction = min(completed_updates / config.kl_budget_steps, 1.0)
    return (
        config.kl_budget_final
        + (config.kl_budget - config.kl_budget_final)
        * (1 + math.cos(math.pi * fraction))
        / 2
    )


@dataclass
class TaskBaseline:
    """Subtract each task/template's success rate from previous visits only."""

    config: CuaConfig
    counts: dict[tuple[str, str], tuple[float, int]] = field(
        default_factory=dict, init=False
    )
    global_rate: float = field(default=0.0, init=False)
    template_rates: dict[str, float] = field(default_factory=dict, init=False)

    def advantages(
        self,
        task_ids: Sequence[str],
        templates: Sequence[str],
        rewards: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """One entry per training segment; infrastructure failures never count as visits."""
        if rewards.ndim != 1 or valid.shape != rewards.shape:
            raise ValueError(
                "Rewards and validity must be one-dimensional episode arrays"
            )
        if len(task_ids) != len(templates) or len(task_ids) != rewards.numel():
            raise ValueError("Episode identities and rewards have different lengths")
        valid = valid.bool()
        if not torch.isfinite(rewards[valid]).all():
            raise ValueError("Valid episodes must have finite rewards")
        successes = (rewards >= 1.0).float()
        keys = [
            (task, template if self.config.baseline_by_template else "")
            for task, template in zip(task_ids, templates, strict=True)
        ]
        prior = self.config.baseline_prior
        probabilities = []
        for key in keys:
            wins, visits = self.counts.get(key, (0.0, 0))
            centre = (
                self.template_rates.get(key[1], self.global_rate)
                if self.config.baseline_by_template
                else self.global_rate
            )
            probabilities.append((wins + prior * centre) / (visits + prior))
        # Compute all advantages before counting any episode in this batch.
        result = successes - torch.tensor(probabilities, device=rewards.device)
        result = torch.where(valid, result, 0.0)
        groups: dict[str, list[float]] = {}
        for key, success, keep in zip(
            keys, successes.tolist(), valid.tolist(), strict=True
        ):
            if keep:
                wins, visits = self.counts.get(key, (0.0, 0))
                self.counts[key] = (wins + success, visits + 1)
                if self.config.baseline_by_template:
                    groups.setdefault(key[1], []).append(success)
        if valid.any():
            beta = self.config.baseline_ema
            self.global_rate = (1 - beta) * self.global_rate + beta * successes[
                valid
            ].mean().item()
            for template, values in groups.items():
                # The historical implementation initializes new template centres
                # from the updated global EMA, after forming this batch's advantages.
                old = self.template_rates.get(template, self.global_rate)
                self.template_rates[template] = (1 - beta) * old + beta * sum(
                    values
                ) / len(values)
        return result

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "counts": self.counts,
                "global_rate": self.global_rate,
                "template_rates": self.template_rates,
            }
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        restored = copy.deepcopy(state)
        self.counts = restored["counts"]
        self.global_rate = restored["global_rate"]
        self.template_rates = restored["template_rates"]


def _sum_across_ranks(value: Tensor, process_group: dist.ProcessGroup | None) -> Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, group=process_group)
    return value


@torch.no_grad()
def fd_kl_step(
    optimizer: LoGRAOptimizer,
    config: CuaConfig,
    *,
    gradient_norm: float,
    evaluate_logprobs: Callable[[], Iterator[Tensor]],
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, float]:
    """Apply historical target-norm SGD followed by FD predicted-KL scaling.

    ``evaluate_logprobs`` yields [positions, vocabulary] log probabilities on the
    same contexts in the same order on each invocation. It must also reshard any
    FSDP weights before yielding. Probes never mutate the FP32 optimizer masters.
    The quadratic prediction is a scale-selection rule, not an exact KL bound.
    """
    if optimizer.config.optimizer != "sgd" or not optimizer.ready:
        raise ValueError("FD KL control requires synchronized SGD sketches")
    if optimizer.param_groups[0]["weight_decay"] or any(
        group["params"] for group in optimizer.native.param_groups
    ):
        raise ValueError(
            "CUA FD control requires frozen non-target parameters and zero decay"
        )
    lr = optimizer.param_groups[0]["lr"]
    if not math.isfinite(gradient_norm) or gradient_norm < 0 or lr <= 0:
        raise ValueError(
            "FD control requires a finite nonnegative gradient norm and positive LR"
        )
    budget = kl_budget_at(config, optimizer.update_count)
    if gradient_norm == 0:
        optimizer.step(scale=0.0)
        return {
            "kl_alpha": 0.0,
            "kl_predicted": 0.0,
            "kl_budget": budget,
            "zero_gradient": 1.0,
        }

    device = optimizer.layers[0].sketch.device
    weight_norm2 = torch.zeros((), device=device, dtype=torch.float64)
    for layer in optimizer.layers:
        local, _ = local_weight_rows(layer.weight)
        for start in range(0, local.shape[0], 256):
            weight_norm2 += local[start : start + 256].double().square().sum()
    weight_norm = _sum_across_ranks(weight_norm2, process_group).sqrt().item()
    proposal_scale = config.max_update_ratio * weight_norm / (lr * gradient_norm)
    references = [value.detach().to("cpu", copy=True) for value in evaluate_logprobs()]

    def divergence() -> tuple[float, int]:
        totals = torch.zeros(2, device=device, dtype=torch.float64)
        for reference, current in zip(references, evaluate_logprobs(), strict=True):
            if reference.shape != current.shape or current.ndim != 2:
                raise ValueError("FD probe contexts changed between evaluations")
            reference = reference.to(current.device)
            totals[0] += (
                (reference.exp() * (reference - current))
                .sum(dtype=torch.float64)
                .to(device)
            )
            totals[1] += current.shape[0]
        totals = _sum_across_ranks(totals, process_group)
        count = int(totals[1].item())
        if count == 0:
            raise ValueError("No valid generated-token positions for the FD probe")
        return totals[0].item() / count, count

    with optimizer.trial_step(scale=proposal_scale):
        proposal_kl, count = divergence()
    if not math.isfinite(proposal_kl) or proposal_kl <= 0:
        raise FloatingPointError(f"FD probe returned invalid KL: {proposal_kl}")
    alpha = min(config.alpha_max, math.sqrt(budget / proposal_kl))
    optimizer.step(scale=proposal_scale * alpha)
    metrics = {
        "kl_proposal": proposal_kl,
        "kl_predicted": proposal_kl * alpha**2,
        "kl_budget": budget,
        "kl_alpha": alpha,
        "kl_probe_tokens": float(count),
        "update_relative_norm": config.max_update_ratio * alpha,
    }
    if config.check_realized_kl:
        realized_kl, _ = divergence()
        if not math.isfinite(realized_kl):
            raise FloatingPointError("The realized KL is nonfinite after the update")
        metrics["kl_realized"] = realized_kl
    return metrics


@dataclass
class CuaUpdate:
    """Add routing replay and FD probes around the native AutoModel train loop."""

    worker: "LoGRAWorkerImpl"
    config: CuaConfig
    probes: list["ProcessedMicrobatch"] = field(default_factory=list, init=False)
    gates: list["ReplayGate"] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.config.routing_replay:
            return
        from nemo_rl.models.automodel.router_replay import router_replay_gates

        for module in self.worker.model.modules():
            if getattr(module, "bias_update_factor", 0) != 0:
                raise ValueError("The CUA recipe requires frozen router bias")
        # Reuse the native worker's handles rather than installing a second set;
        # native logprob scoring sees these same routing handles.
        self.gates = router_replay_gates(self.worker.model)
        if not self.gates:
            raise ValueError("Routing replay requested, but no AutoModel gates found")

    @contextmanager
    def replay(self, batch: "ProcessedMicrobatch") -> Iterator[None]:
        if not self.config.routing_replay:
            yield
            return
        from nemo_rl.models.automodel.router_replay import microbatch_replay_context

        with microbatch_replay_context(self.gates, batch):
            yield

    @contextmanager
    def training_batches(
        self, batches: Iterator["ProcessedMicrobatch"]
    ) -> Iterator[Iterator["ProcessedMicrobatch"]]:
        self.probes.clear()

        def replayed() -> Generator["ProcessedMicrobatch", None, None]:
            for batch in batches:
                if len(self.probes) < self.config.probe_sequences:
                    self.probes.append(batch)
                # The consumer performs BOTH forward and backward before
                # requesting the next item, so checkpoint recomputation replays
                # the same routes. closing() also restores routing on exceptions.
                with self.replay(batch):
                    yield batch

        with closing(replayed()) as iterator:
            yield iterator

    @torch.no_grad()
    def logprobs(self) -> Iterator[Tensor]:
        from torch.distributed.fsdp import FSDPModule

        from nemo_rl.models.automodel.train import extract_logits, prepare_model_forward

        worker = self.worker
        model = worker.model
        was_training = model.training
        model.eval()
        per_sequence = max(1, self.config.probe_tokens // len(self.probes))
        try:
            for batch in self.probes:
                if batch.original_batch_size != 1:
                    raise ValueError("CUA FD probes require microbatch size one")
                # Position i predicts token i+1; only generated, valid tokens
                # count. Still forward empty/dummy rows to match EP collectives.
                mask = batch.data_dict["token_mask"][0, 1:].bool()
                mask = mask & batch.data_dict["sample_mask"][0].bool()
                positions = mask.nonzero().flatten()
                stride = max(1, positions.numel() // per_sequence)
                positions = positions[::stride][:per_sequence]
                prepared = prepare_model_forward(
                    model,
                    batch.processed_inputs,
                    device_mesh=worker.device_mesh,
                    cp_size=1,
                    padding_token_id=worker.tokenizer.pad_token_id or 0,
                    is_reward_model=False,
                    allow_flash_attn_args=worker.allow_flash_attn_args,
                )
                kwargs = dict(prepared.model_batch)
                # Omni's native forward projects only these positions through
                # the vocabulary head, avoiding full-context vocabulary logits.
                kwargs["logits_to_keep"] = (
                    positions if positions.numel() else positions.new_zeros(1)
                )
                try:
                    with worker._autocast_context():
                        output = model(**kwargs)
                        logits = extract_logits(model, output)[0, : positions.numel()]
                        temperature = worker.sampling_params.temperature
                        values = (logits.float() / temperature).log_softmax(-1)
                    del output, logits
                finally:
                    # Trial writes and the committed update operate on shards.
                    child: torch.nn.Module | FSDPModule
                    for child in model.modules():
                        if isinstance(child, FSDPModule):
                            child.reshard()
                yield values
        finally:
            model.train(was_training)

    def step(self, optimizer: LoGRAOptimizer, gradient_norm: float) -> dict[str, float]:
        if not self.probes:
            raise RuntimeError(
                "No training microbatches were retained for the FD probe"
            )
        try:
            metrics = fd_kl_step(
                optimizer,
                self.config,
                gradient_norm=gradient_norm,
                evaluate_logprobs=self.logprobs,
                process_group=self.worker.dp_mesh.get_group(),
            )
        finally:
            self.probes.clear()
        return metrics


def make_cua_loss(
    config: CuaConfig, loss_config: "ClippedPGLossConfig"
) -> "LossFunction":
    """Reuse native REINFORCE/TIS with current-policy weights and per-row scaling.

    CUA recomputes the importance ratio at each update. No separate old-policy
    pass is needed, and the ratio is detached before weighting the loss.
    """
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    if (
        not loss_config.disable_ppo_ratio
        or loss_config.force_on_policy_ratio
        or not loss_config.token_level_loss
        or loss_config.reference_policy_kl_penalty != 0
        or loss_config.ratio_clip_c is not None
        or loss_config.use_cispo
        or loss_config.positive_example_nll_weight != 0
        or not loss_config.use_importance_sampling_correction
        or loss_config.truncated_importance_sampling_type != "tis"
    ):
        raise ValueError(
            "CUA requires token-level REINFORCE with clipped importance weights and no auxiliary loss"
        )

    class CuaLoss(ClippedPGLossFn):
        def __call__(self, next_token_logprobs=None, data=None, **kwargs):
            if next_token_logprobs is None or data is None:
                raise ValueError(
                    "CUA loss requires current log probabilities and rollout data"
                )
            prepared = BatchedDataDict(dict(data))
            prepared["prev_logprobs"] = torch.nn.functional.pad(
                next_token_logprobs.detach(), (1, 0)
            )
            if config.length_normalization:
                lengths = data["token_mask"][:, 1:].sum(-1, keepdim=True).clamp_min(1)
                prepared["advantages"] = data["advantages"] * (
                    config.length_normalization_reference / lengths
                )
            return super().__call__(
                next_token_logprobs=next_token_logprobs, data=prepared, **kwargs
            )

    return CuaLoss(loss_config)

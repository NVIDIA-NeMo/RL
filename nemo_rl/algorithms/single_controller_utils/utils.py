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

"""Helpers used by SingleControllerActor."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.metric_utils import calculate_single_metric

# Reduction rules for all_mb_metrics. Mirror grpo.py / grpo_sync.py.
_MB_METRIC_MIN: frozenset[str] = frozenset(
    {"probs_ratio_min", "probs_ratio_clamped_min"}
)
_MB_METRIC_MAX: frozenset[str] = frozenset(
    {"probs_ratio_max", "probs_ratio_clamped_max"}
)
_MB_METRIC_MEAN: frozenset[str] = frozenset(
    {
        "lr",
        "wd",
        "reward",
        "global_valid_seqs",
        "global_valid_toks",
        "mean_prompt_length",
    }
)


def aggregate_step_metrics(train_result: dict[str, Any]) -> dict[str, Any]:
    """Reduce per-microbatch metric lists into step-level scalars.

    Args:
        train_result: Output of TQPolicy.finish_train_step.

    Returns:
        Flat dict of step-level scalars ready for logging.
    """
    metrics: dict[str, Any] = {}
    loss = train_result.get("loss")
    if isinstance(loss, torch.Tensor):
        metrics["loss"] = loss.detach().mean().item()
    elif loss is not None:
        metrics["loss"] = float(loss)
    grad_norm = train_result.get("grad_norm")
    if isinstance(grad_norm, torch.Tensor):
        metrics["grad_norm"] = grad_norm.detach().mean().item()
    elif grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm)
    if "total_flops" in train_result:
        metrics["total_flops"] = float(train_result["total_flops"])
    if "num_ranks" in train_result:
        metrics["num_ranks"] = int(train_result["num_ranks"])

    # moe/mtp share the same reduction rules as all_mb_metrics in grpo.py.
    mb: dict[str, list[Any]] = {}
    if "moe_metrics" in train_result:
        mb.update({f"moe/{k}": v for k, v in train_result["moe_metrics"].items()})
    if "mtp_metrics" in train_result:
        mb.update({f"mtp/{k}": v for k, v in train_result["mtp_metrics"].items()})
    mb.update(train_result.get("all_mb_metrics", {}))

    for k, v in mb.items():
        if k in _MB_METRIC_MIN:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.min(valid)) if valid else -1.0
        elif k in _MB_METRIC_MAX:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.max(valid)) if valid else -1.0
        elif k in _MB_METRIC_MEAN:
            metrics[k] = float(np.mean(v))
        else:
            metrics[k] = float(np.sum(v))
    return metrics


def _metric_values(value: Any) -> list[Any]:
    """Normalize a metric payload so results can be concatenated by key."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _collect_reducible_metrics(
    train_result: dict[str, Any],
) -> dict[str, list[Any]]:
    """Collect one finish result's metrics that share GRPO reductions."""
    mb: dict[str, list[Any]] = {}
    if "moe_metrics" in train_result:
        mb.update(
            {
                f"moe/{k}": _metric_values(v)
                for k, v in train_result["moe_metrics"].items()
            }
        )
    if "mtp_metrics" in train_result:
        mb.update(
            {
                f"mtp/{k}": _metric_values(v)
                for k, v in train_result["mtp_metrics"].items()
            }
        )
    mb.update(
        {
            k: _metric_values(v)
            for k, v in train_result.get("all_mb_metrics", {}).items()
        }
    )
    return mb


def _reduce_mb_metrics(mb: dict[str, list[Any]]) -> dict[str, float]:
    """Apply the legacy GRPO per-microbatch reduction rules."""
    metrics: dict[str, float] = {}
    for k, v in mb.items():
        if k in _MB_METRIC_MIN:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.min(valid)) if valid else -1.0
        elif k in _MB_METRIC_MAX:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.max(valid)) if valid else -1.0
        elif k in _MB_METRIC_MEAN:
            metrics[k] = float(np.mean(v))
        else:
            metrics[k] = float(np.sum(v))
    return metrics


def aggregate_step_metrics_multi_minibatch(
    train_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce an AReaL RL step's optimizer results into logging scalars.

    Per-microbatch metric lists are concatenated before applying the existing
    GRPO reductions; summed keys are then rescaled by 1/M because each finish
    result's lists are already normalized within that minibatch (a result's
    list sums to its minibatch mean). Metrics that describe one optimizer
    result retain their natural cross-result semantics: loss and grad norm are
    averaged, FLOPs and valid counts are summed, and the final optimizer
    result supplies LR/WD.

    Args:
        train_results: Ordered ``finish_train_step`` outputs, one per AReaL
            training minibatch.

    Returns:
        Flat dict of RL-step scalars ready for logging.

    Raises:
        ValueError: If no optimizer results are provided.
    """
    if not train_results:
        raise ValueError(
            "aggregate_step_metrics_multi_minibatch requires at least one train "
            "result"
        )
    if len(train_results) == 1:
        return aggregate_step_metrics(train_results[0])

    per_result = [aggregate_step_metrics(result) for result in train_results]
    merged_mb: dict[str, list[Any]] = {}
    for result in train_results:
        for key, values in _collect_reducible_metrics(result).items():
            merged_mb.setdefault(key, []).extend(values)
    metrics: dict[str, Any] = _reduce_mb_metrics(merged_mb)

    # Each finish result's non-min/max lists are normalized within that
    # minibatch (the worker rescales by its own 1/N, so a result's list sums
    # to the minibatch MEAN — mirroring the sync path where the worker
    # pre-divides by num_global_batches). Summing the concatenation therefore
    # yields M× the per-step value; rescale summed keys back to the mean over
    # minibatches. Min/max/mean rules need no correction, and the step-level
    # keys below (valid counts, lr/wd) are overwritten explicitly anyway.
    for key in merged_mb:
        if (
            key not in _MB_METRIC_MIN
            and key not in _MB_METRIC_MAX
            and key not in _MB_METRIC_MEAN
        ):
            metrics[key] /= len(train_results)

    # These fields describe a whole optimizer result, not an individual inner
    # microbatch. Reduce them across optimizer-step boundaries explicitly.
    for key in ("loss", "grad_norm"):
        values = [result[key] for result in per_result if key in result]
        if values:
            metrics[key] = float(np.mean(values))

    # A finish result repeats its global count for each inner microbatch. The
    # single-result reducer removes that duplication; the RL step then needs the
    # sum across its separate optimizer steps for correct throughput metrics.
    for key in ("total_flops", "global_valid_seqs", "global_valid_toks"):
        values = [result[key] for result in per_result if key in result]
        if values:
            metrics[key] = float(np.sum(values))

    num_ranks = [result["num_ranks"] for result in per_result if "num_ranks" in result]
    if num_ranks:
        metrics["num_ranks"] = num_ranks[-1]

    # LR/WD are repeated within a finish result and change between optimizer
    # steps. Log the values used by the final optimizer step.
    for key in ("lr", "wd"):
        if key in per_result[-1]:
            metrics[key] = per_result[-1][key]
        else:
            metrics.pop(key, None)
    return metrics


def reduce_advantage_pump_metrics(
    rewards: list[torch.Tensor],
    masked_advantages: list[torch.Tensor],
    sequence_lengths: list[int],
    gen_tokens_per_sample: list[float] | None = None,
    staleness: list[int] | None = None,
) -> dict[str, Any]:
    """Reduce per-step accumulators from _advantage_pump into step scalars.

    Args:
        rewards: One tensor per advantage_pump call; each row a sample reward.
        masked_advantages: Token-masked advantages, one tensor per call.
        sequence_lengths: All input_lengths trained on this step.
        gen_tokens_per_sample: Per-row generated-token counts (token_mask sums),
            when the controller collects them. Emitted under the SAME metric
            names as the legacy NeMo-Gym rollout path
            (``gen_tokens_per_sample/*``, ``total_tokens_per_sample/*``,
            ``mean_gen_tokens_per_sample``) so wandb charts line up across the
            legacy and SC paths.
        staleness: Per-sample weight-version lag (trainer version at
            consumption minus the sample's generation version) for every
            sample trained on this step.

    Returns:
        Dict with reward, advantages/{mean,max,min}, total_num_tokens, and —
        when ``gen_tokens_per_sample`` is provided — the legacy-named length
        metrics above, and — when ``staleness`` is provided —
        staleness/{mean,max,min}.
    """
    out: dict[str, float] = {}
    if rewards:
        out["reward"] = float(torch.cat([r.flatten() for r in rewards]).mean())
    if masked_advantages:
        cat = torch.cat([a.flatten() for a in masked_advantages])
        if cat.numel() > 0:
            out["advantages/mean"] = float(cat.mean())
            out["advantages/max"] = float(cat.max())
            out["advantages/min"] = float(cat.min())
        else:
            out["advantages/mean"] = 0.0
            out["advantages/max"] = 0.0
            out["advantages/min"] = 0.0
    if sequence_lengths:
        out["total_num_tokens"] = float(sum(sequence_lengths))
    if gen_tokens_per_sample:
        out.update(
            calculate_single_metric(
                gen_tokens_per_sample,
                len(gen_tokens_per_sample),
                "gen_tokens_per_sample",
            )
        )
        # Alias kept for parity with the legacy path's downstream logging.
        out["mean_gen_tokens_per_sample"] = out["gen_tokens_per_sample/mean"]
        if sequence_lengths:
            out.update(
                calculate_single_metric(
                    sequence_lengths,
                    len(sequence_lengths),
                    "total_tokens_per_sample",
                )
            )
    if staleness:
        out["staleness/mean"] = float(np.mean(staleness))
        out["staleness/max"] = float(max(staleness))
        out["staleness/min"] = float(min(staleness))
    return out


def tensor_field(data: TensorDict, field_name: str) -> torch.Tensor:
    """Read a tensor column from a TensorDict, depadding if nested.

    Args:
        data: TensorDict returned by the data plane.
        field_name: Column name to fetch.

    Returns:
        Dense tensor (nested columns are padded with zeros).
    """
    value = data[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor field {field_name!r}; got {type(value)}")
    if value.is_nested:
        return torch.nested.to_padded_tensor(value, padding=0)
    return value


def squeeze_trailing_unit_dim(value: torch.Tensor) -> torch.Tensor:
    """Drop a trailing dim of size 1 if present.

    Args:
        value: Input tensor.

    Returns:
        Tensor without the trailing unit dim.
    """
    if value.dim() >= 2 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def fields_for_put(meta: KVBatchMeta, fields: dict[str, torch.Tensor]) -> TensorDict:
    """Pack tensors for DataPlane put, re-nesting jagged rows when needed.

    Args:
        meta: Batch meta whose sequence_lengths drive the nesting.
        fields: Field name to dense tensor.

    Returns:
        TensorDict shaped for dp_client.put_samples.
    """
    packed: dict[str, torch.Tensor] = {}
    if meta.sequence_lengths is None:
        for field_name, value in fields.items():
            packed[field_name] = value.detach().contiguous()
        # pyrefly: ignore[bad-argument-type]
        return TensorDict(packed, batch_size=[meta.size])

    lengths = torch.tensor(meta.sequence_lengths, dtype=torch.long)
    for field_name, value in fields.items():
        if value.dim() >= 2 and value.shape[1] == int(lengths.max().item()):
            rows = [
                value[i, : int(lengths[i].item())].detach().contiguous()
                for i in range(meta.size)
            ]
            packed[field_name] = torch.nested.as_nested_tensor(
                rows,
                layout=torch.jagged,
            )
        else:
            packed[field_name] = value.detach().contiguous()
    # pyrefly: ignore[bad-argument-type]
    return TensorDict(packed, batch_size=[meta.size])

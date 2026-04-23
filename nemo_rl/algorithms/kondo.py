from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import torch
from typing_extensions import NotRequired

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

KondoMode = Literal[
    "off",
    "dense_reference",
    "stochastic_response_rows",
]
KondoPriorityMode = Literal[
    "delight",
    "split_dual_tempered_delight",
    "advantage",
    "abs_advantage",
    "surprisal",
    "uniform",
]


class KondoConfig(TypedDict):
    enabled: bool
    mode: NotRequired[KondoMode]
    target_backward_token_fraction: NotRequired[float]
    priority_mode: NotRequired[KondoPriorityMode]
    surprisal_temperature: NotRequired[float]
    gate_temperature: NotRequired[float]
    positive_keep_floor: NotRequired[float]
    negative_keep_floor: NotRequired[float]
    min_keep_probability: NotRequired[float]
    bypass_on_nonfinite: NotRequired[bool]


@dataclass(frozen=True)
class KondoScreeningResult:
    loss_token_mask: torch.Tensor
    loss_normalizer: torch.Tensor
    kl_normalizer: torch.Tensor
    valid_token_mask: torch.Tensor
    screen_mask: torch.Tensor
    reference_token_gate: torch.Tensor
    priority: torch.Tensor
    row_cost: torch.Tensor
    row_response_utility: torch.Tensor
    row_response_density: torch.Tensor
    row_reference_tokens: torch.Tensor
    row_abs_priority: torch.Tensor
    row_sampling_priority: torch.Tensor
    total_valid_tokens: int
    total_reference_tokens: int
    screenable_token_count: int
    nonfinite_screen_token_count: int
    rows_total: int
    should_bypass: bool


def resolve_kondo_config(raw_cfg: dict[str, Any] | None) -> KondoConfig:
    cfg: KondoConfig = {
        "enabled": False,
        "mode": "stochastic_response_rows",
        "target_backward_token_fraction": 0.7,
        "priority_mode": "split_dual_tempered_delight",
        "surprisal_temperature": 0.1,
        "gate_temperature": 1.0,
        "positive_keep_floor": 0.95,
        "negative_keep_floor": 0.25,
        "min_keep_probability": 0.05,
        "bypass_on_nonfinite": True,
    }
    if raw_cfg is not None:
        cfg.update(raw_cfg)

    mode = cfg["mode"]
    if mode not in {"off", "dense_reference", "stochastic_response_rows"}:
        raise ValueError(f"Unsupported Kondo mode: {mode}")

    fraction = cfg["target_backward_token_fraction"]
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            "grpo.kondo.target_backward_token_fraction must be in (0, 1]"
        )

    priority_mode = cfg["priority_mode"]
    if priority_mode not in {
        "delight",
        "split_dual_tempered_delight",
        "advantage",
        "abs_advantage",
        "surprisal",
        "uniform",
    }:
        raise ValueError(f"Unsupported Kondo priority mode: {priority_mode}")

    surprisal_temperature = cfg["surprisal_temperature"]
    if surprisal_temperature <= 0:
        raise ValueError("grpo.kondo.surprisal_temperature must be positive")

    gate_temperature = cfg["gate_temperature"]
    if gate_temperature <= 0:
        raise ValueError("grpo.kondo.gate_temperature must be positive")

    positive_keep_floor = cfg["positive_keep_floor"]
    if not (0.0 <= positive_keep_floor <= 1.0):
        raise ValueError("grpo.kondo.positive_keep_floor must be in [0, 1]")

    negative_keep_floor = cfg["negative_keep_floor"]
    if not (0.0 <= negative_keep_floor <= 1.0):
        raise ValueError("grpo.kondo.negative_keep_floor must be in [0, 1]")

    min_keep_probability = cfg["min_keep_probability"]
    if not (0.0 < min_keep_probability <= 1.0):
        raise ValueError("grpo.kondo.min_keep_probability must be in (0, 1]")

    if (
        priority_mode == "split_dual_tempered_delight"
        and mode not in {"dense_reference", "stochastic_response_rows"}
    ):
        raise ValueError(
            "grpo.kondo.priority_mode=split_dual_tempered_delight is currently only "
            "supported with grpo.kondo.mode in {dense_reference, stochastic_response_rows}"
        )

    return cfg


def kondo_mode_metric(mode: KondoMode) -> float:
    if mode == "off":
        return 0.0
    if mode == "dense_reference":
        return 1.0
    if mode == "stochastic_response_rows":
        return 2.0
    raise ValueError(f"Unsupported Kondo mode: {mode}")


def disabled_kondo_metrics(mode: KondoMode = "off") -> dict[str, float]:
    return {
        "kondo_enabled": 0.0,
        "kondo_mode": kondo_mode_metric(mode),
        "kondo_target_backward_token_fraction": 0.0,
        "kondo_recall_floor": 0.0,
        "kondo_recall_floor_satisfied": 0.0,
        "kondo_bypass_step": 0.0,
        "kondo_rows_total": 0.0,
        "kondo_rows_kept": 0.0,
        "kondo_selected_token_recall": 0.0,
        "kondo_selected_token_fraction": 0.0,
        "kondo_actual_backward_token_fraction": 0.0,
        "kondo_kept_row_token_fraction": 0.0,
        "kondo_nonfinite_screen_token_count": 0.0,
        "kondo_row_keep_probability_mean": 0.0,
        "kondo_row_ht_weight_mean": 0.0,
    }


def compute_token_priority(
    *,
    advantages: torch.Tensor,
    surprisal: torch.Tensor,
    priority_mode: KondoPriorityMode,
) -> torch.Tensor:
    if priority_mode == "delight":
        return advantages * surprisal
    if priority_mode == "advantage":
        return advantages
    if priority_mode == "abs_advantage":
        return advantages.abs()
    if priority_mode == "surprisal":
        return surprisal
    if priority_mode == "uniform":
        return torch.ones_like(surprisal)
    raise ValueError(f"Unsupported Kondo priority mode: {priority_mode}")


def _solve_sigmoid_bias_for_target_mean(
    logits: torch.Tensor,
    target_mean: float,
) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.empty_like(logits, dtype=torch.float32)
    if target_mean <= 0.0:
        return torch.zeros_like(logits, dtype=torch.float32)
    if target_mean >= 1.0:
        return torch.ones_like(logits, dtype=torch.float32)

    low = -80.0
    high = 80.0
    for _ in range(60):
        mid = (low + high) / 2.0
        probs = torch.sigmoid(logits + mid)
        mean_prob = float(probs.mean().item())
        if mean_prob < target_mean:
            low = mid
        else:
            high = mid
    return torch.sigmoid(logits + high).to(dtype=torch.float32)


def _build_split_dual_tempered_dense_priority_and_gate(
    *,
    advantages: torch.Tensor,
    surprisal: torch.Tensor,
    screen_mask: torch.Tensor,
    surprisal_temperature: float,
    gate_temperature: float,
    prompt_group_std: torch.Tensor | None,
    positive_keep_floor: float,
    negative_keep_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    gate = torch.zeros_like(advantages, dtype=torch.float32)
    tempered_surprisal = torch.log1p(
        surprisal.clamp(min=0) / surprisal_temperature
    )
    signed_delight = advantages * tempered_surprisal
    priority = advantages.abs() * tempered_surprisal

    positive_mask = screen_mask & (advantages > 1e-9)
    negative_mask = screen_mask & (advantages < -1e-9)
    if prompt_group_std is not None:
        zero_std_rows = (prompt_group_std <= 1e-6).unsqueeze(-1)
        positive_mask = positive_mask & ~zero_std_rows
        negative_mask = negative_mask & ~zero_std_rows

    positive_logits = signed_delight[positive_mask] / gate_temperature
    negative_logits = signed_delight[negative_mask] / gate_temperature

    if positive_logits.numel() > 0:
        gate[positive_mask] = _solve_sigmoid_bias_for_target_mean(
            positive_logits,
            positive_keep_floor,
        ).to(dtype=gate.dtype)
    if negative_logits.numel() > 0:
        gate[negative_mask] = _solve_sigmoid_bias_for_target_mean(
            negative_logits,
            negative_keep_floor,
        ).to(dtype=gate.dtype)

    return priority, gate, math.ceil(float(gate.sum().item()))


def compute_required_row_multiple(
    dp_size: int,
    train_micro_batch_size: int,
    use_dynamic_batching: bool,
    use_sequence_packing: bool,
) -> int:
    if dp_size < 1:
        raise ValueError("dp_size must be at least 1")
    if use_dynamic_batching or use_sequence_packing:
        return dp_size
    return max(dp_size * train_micro_batch_size, dp_size)


def _build_exact_topk_token_gate(
    *,
    priority: torch.Tensor,
    screen_mask: torch.Tensor,
    target_fraction: float,
) -> tuple[torch.Tensor, int]:
    gate = torch.zeros_like(priority, dtype=torch.float32)
    candidate_indices = screen_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    if candidate_indices.numel() == 0:
        return gate, 0

    candidate_scores = priority.reshape(-1)[candidate_indices]
    k_target = max(1, math.ceil(target_fraction * candidate_indices.numel()))
    _, topk_positions = torch.topk(
        candidate_scores,
        k=k_target,
        largest=True,
        sorted=False,
    )
    selected_indices = candidate_indices[topk_positions]
    gate.reshape(-1)[selected_indices] = 1.0
    return gate, int(k_target)


def screen_kondo_tokens(
    train_data: BatchedDataDict[Any],
    cfg: KondoConfig,
) -> KondoScreeningResult | None:
    token_mask = train_data["token_mask"][:, 1:]
    valid_token_mask = token_mask > 0
    row_cost = valid_token_mask.sum(dim=-1).to(torch.int64)
    rows_total = int(train_data["sample_mask"].shape[0])
    total_valid_tokens = int(row_cost.sum().item())

    prev_logprobs = train_data["prev_logprobs"][:, 1:]
    advantages = train_data["advantages"][:, 1:]

    finite_screen_mask = (
        valid_token_mask
        & torch.isfinite(prev_logprobs)
        & torch.isfinite(advantages)
    )
    nonfinite_screen_token_count = int(
        (valid_token_mask & ~finite_screen_mask).sum().item()
    )
    screenable_token_count = int(finite_screen_mask.sum().item())

    surprisal = torch.zeros_like(prev_logprobs)
    priority = torch.zeros_like(prev_logprobs)
    reference_token_gate = torch.zeros_like(prev_logprobs, dtype=torch.float32)

    if screenable_token_count > 0:
        surprisal[finite_screen_mask] = (-prev_logprobs.detach())[finite_screen_mask]
        if cfg["priority_mode"] == "split_dual_tempered_delight":
            tempered_surprisal = torch.log1p(
                surprisal.clamp(min=0) / cfg["surprisal_temperature"]
            )
            if cfg["mode"] == "stochastic_response_rows":
                # Use a dense teacher built from detached actor magnitude, then
                # sample prompt groups from the resulting teacher coverage.
                priority = advantages.detach().abs() * tempered_surprisal
                reference_token_gate, _ = _build_exact_topk_token_gate(
                    priority=priority,
                    screen_mask=finite_screen_mask,
                    target_fraction=cfg["target_backward_token_fraction"],
                )
            else:
                prompt_group_std = train_data.get("prompt_group_std")
                priority, reference_token_gate, _ = (
                    _build_split_dual_tempered_dense_priority_and_gate(
                        advantages=advantages.detach(),
                        surprisal=surprisal,
                        screen_mask=finite_screen_mask,
                        surprisal_temperature=cfg["surprisal_temperature"],
                        gate_temperature=cfg["gate_temperature"],
                        prompt_group_std=(
                            prompt_group_std.to(dtype=torch.float32)
                            if prompt_group_std is not None
                            else None
                        ),
                        positive_keep_floor=cfg["positive_keep_floor"],
                        negative_keep_floor=cfg["negative_keep_floor"],
                    )
                )
        else:
            priority = compute_token_priority(
                advantages=advantages.detach(),
                surprisal=surprisal,
                priority_mode=cfg["priority_mode"],
            )
            reference_token_gate, _ = _build_exact_topk_token_gate(
                priority=priority,
                screen_mask=finite_screen_mask,
                target_fraction=cfg["target_backward_token_fraction"],
            )

    loss_token_mask = torch.zeros_like(
        train_data["token_mask"],
        dtype=train_data["token_mask"].dtype,
    )
    loss_token_mask[:, 1:] = reference_token_gate.to(dtype=loss_token_mask.dtype)

    batch_size = train_data["sample_mask"].shape[0]
    if cfg["priority_mode"] == "split_dual_tempered_delight":
        pg_normalizer_value = max(float(reference_token_gate.sum().item()), 1.0)
        kl_normalizer_value = float(total_valid_tokens)
    else:
        pg_normalizer_value = float(total_valid_tokens)
        kl_normalizer_value = float(total_valid_tokens)

    loss_normalizer = torch.full(
        (batch_size,),
        pg_normalizer_value,
        dtype=torch.float32,
    )
    kl_normalizer = torch.full(
        (batch_size,),
        kl_normalizer_value,
        dtype=torch.float32,
    )

    row_response_utility = (priority * finite_screen_mask).sum(dim=-1)
    row_reference_tokens = reference_token_gate.sum(dim=-1).to(torch.float32)
    row_abs_priority = (reference_token_gate * priority.abs()).sum(dim=-1)
    row_sampling_priority = row_abs_priority.clone()
    if (
        cfg["priority_mode"] == "split_dual_tempered_delight"
        and cfg["mode"] == "dense_reference"
    ):
        row_sampling_priority = row_response_utility.clamp(min=0)
    row_response_density = row_response_utility / row_cost.clamp(min=1).to(
        row_response_utility.dtype
    )
    total_reference_tokens = math.ceil(float(row_reference_tokens.sum().item()))

    return KondoScreeningResult(
        loss_token_mask=loss_token_mask,
        loss_normalizer=loss_normalizer,
        kl_normalizer=kl_normalizer,
        valid_token_mask=valid_token_mask,
        screen_mask=finite_screen_mask,
        reference_token_gate=reference_token_gate,
        priority=priority,
        row_cost=row_cost,
        row_response_utility=row_response_utility,
        row_response_density=row_response_density,
        row_reference_tokens=row_reference_tokens,
        row_abs_priority=row_abs_priority,
        row_sampling_priority=row_sampling_priority,
        total_valid_tokens=total_valid_tokens,
        total_reference_tokens=total_reference_tokens,
        screenable_token_count=screenable_token_count,
        nonfinite_screen_token_count=nonfinite_screen_token_count,
        rows_total=rows_total,
        should_bypass=(
            screenable_token_count == 0
            or (cfg["bypass_on_nonfinite"] and nonfinite_screen_token_count > 0)
        ),
    )


def _solve_stochastic_row_keep_probabilities(
    *,
    row_signal: torch.Tensor,
    row_cost: torch.Tensor,
    target_token_fraction: float,
    total_valid_tokens: int,
    min_keep_probability: float,
) -> torch.Tensor:
    valid_rows = row_cost > 0
    keep_prob = torch.zeros_like(row_signal, dtype=torch.float32)
    if not valid_rows.any():
        return keep_prob

    cost = row_cost[valid_rows].to(dtype=torch.float32)
    score = row_signal[valid_rows].to(dtype=torch.float32) / torch.sqrt(
        cost.clamp(min=1.0)
    )
    target_budget = float(target_token_fraction * total_valid_tokens)
    minimum_budget = float((cost * min_keep_probability).sum().item())
    if minimum_budget > target_budget + 1e-6:
        raise ValueError(
            "stochastic_response_rows min_keep_probability is too high for the "
            "requested target_backward_token_fraction"
        )

    if float(score.max().item()) <= 0.0:
        keep_prob[valid_rows] = min_keep_probability
        return keep_prob

    def expected_cost(scale: float) -> float:
        probs = torch.clamp(score * scale, min=min_keep_probability, max=1.0)
        return float((probs * cost).sum().item())

    low = 0.0
    high = 1.0
    while expected_cost(high) < target_budget and high < 1e6:
        high *= 2.0

    for _ in range(50):
        mid = (low + high) / 2.0
        if expected_cost(mid) < target_budget:
            low = mid
        else:
            high = mid

    keep_prob[valid_rows] = torch.clamp(
        score * high,
        min=min_keep_probability,
        max=1.0,
    )
    return keep_prob


def _solve_stochastic_row_keep_probabilities_by_group(
    *,
    row_signal: torch.Tensor,
    row_cost: torch.Tensor,
    prompt_group_id: torch.Tensor,
    target_token_fraction: float,
    total_valid_tokens: int,
    min_keep_probability: float,
) -> torch.Tensor:
    keep_prob = torch.zeros_like(row_signal, dtype=torch.float32)
    valid_rows = row_cost > 0
    if not valid_rows.any():
        return keep_prob

    group_signal = []
    group_cost = []
    group_masks = []
    for group in torch.unique(prompt_group_id[valid_rows]):
        group_rows = valid_rows & (prompt_group_id == group)
        if not group_rows.any():
            continue
        group_masks.append(group_rows)
        group_signal.append(row_signal[group_rows].sum())
        group_cost.append(row_cost[group_rows].sum())

    if not group_masks:
        return keep_prob

    solved_group_keep_prob = _solve_stochastic_row_keep_probabilities(
        row_signal=torch.stack(group_signal).to(dtype=torch.float32),
        row_cost=torch.stack(group_cost).to(dtype=torch.int64),
        target_token_fraction=target_token_fraction,
        total_valid_tokens=total_valid_tokens,
        min_keep_probability=min_keep_probability,
    )
    for group_rows, group_keep_prob in zip(group_masks, solved_group_keep_prob):
        keep_prob[group_rows] = group_keep_prob

    return keep_prob


def _sample_stochastic_groups(
    *,
    keep_prob: torch.Tensor,
    row_cost: torch.Tensor,
    prompt_group_id: torch.Tensor,
) -> torch.Tensor:
    valid_rows = row_cost > 0
    if not valid_rows.any():
        return torch.empty(0, dtype=torch.long)

    selected = torch.zeros_like(valid_rows, dtype=torch.bool)
    for group in torch.unique(prompt_group_id[valid_rows]):
        group_rows = valid_rows & (prompt_group_id == group)
        if not group_rows.any():
            continue
        group_keep_prob = keep_prob[group_rows][0]
        if torch.rand((), device=keep_prob.device) < group_keep_prob:
            selected |= group_rows

    return selected.nonzero(as_tuple=False).squeeze(-1).to(dtype=torch.long)


def _sample_stochastic_rows(
    *,
    keep_prob: torch.Tensor,
    row_cost: torch.Tensor,
) -> torch.Tensor:
    valid_rows = row_cost > 0
    if not valid_rows.any():
        return torch.empty(0, dtype=torch.long)
    draws = torch.rand_like(keep_prob)
    selected = valid_rows & (draws < keep_prob)
    return selected.nonzero(as_tuple=False).squeeze(-1).to(dtype=torch.long)


def apply_kondo_mode(
    *,
    train_data: BatchedDataDict[Any],
    screening: KondoScreeningResult | None,
    cfg: KondoConfig,
    row_multiple: int,
) -> tuple[BatchedDataDict[Any], dict[str, float]]:
    metrics = disabled_kondo_metrics(cfg["mode"])
    metrics["kondo_enabled"] = 1.0
    metrics["kondo_mode"] = kondo_mode_metric(cfg["mode"])
    metrics["kondo_target_backward_token_fraction"] = cfg[
        "target_backward_token_fraction"
    ]

    if screening is None:
        metrics["kondo_bypass_step"] = 1.0
        return train_data, metrics

    metrics["kondo_rows_total"] = float(screening.rows_total)
    metrics["kondo_nonfinite_screen_token_count"] = float(
        screening.nonfinite_screen_token_count
    )

    if screening.should_bypass:
        metrics["kondo_bypass_step"] = 1.0
        return train_data, metrics

    reference_selected_tokens = max(screening.total_reference_tokens, 1)

    if cfg["mode"] == "dense_reference":
        train_data["loss_token_mask"] = screening.loss_token_mask
        train_data["loss_normalizer"] = screening.loss_normalizer
        train_data["kl_normalizer"] = screening.kl_normalizer
        row_selected = (screening.row_cost > 0).to(dtype=torch.float32)
        train_data["kondo_row_selected"] = row_selected
        metrics["kondo_rows_kept"] = float(screening.rows_total)
        metrics["kondo_selected_token_recall"] = (
            1.0 if reference_selected_tokens > 0 else 0.0
        )
        metrics["kondo_selected_token_fraction"] = float(
            screening.total_reference_tokens / max(screening.total_valid_tokens, 1)
        )
        metrics["kondo_actual_backward_token_fraction"] = 1.0
        metrics["kondo_kept_row_token_fraction"] = 1.0
        return train_data, metrics

    if cfg["mode"] != "stochastic_response_rows":
        raise ValueError(f"Unsupported Kondo mode: {cfg['mode']}")

    if row_multiple != 1:
        raise ValueError(
            "stochastic_response_rows currently requires row_multiple == 1. "
            "Use a single-DP sync configuration or add an exact aligned sampler."
        )

    if float(screening.row_sampling_priority.max().item()) <= 0.0:
        # When the entire batch has zero actor utility, stochastic row dropping
        # only removes KL/stability updates. Fail open to dense GRPO for that step.
        row_selected = (screening.row_cost > 0).to(dtype=torch.float32)
        row_keep_prob = torch.zeros_like(
            train_data["sample_mask"], dtype=torch.float32
        )
        row_keep_prob[row_selected > 0] = 1.0
        train_data["kondo_row_selected"] = row_selected
        train_data["kondo_row_keep_prob"] = row_keep_prob
        train_data["kondo_row_ht_weight"] = row_selected
        metrics["kondo_bypass_step"] = 1.0
        metrics["kondo_rows_kept"] = float(row_selected.sum().item())
        metrics["kondo_selected_token_recall"] = 1.0
        metrics["kondo_selected_token_fraction"] = float(
            screening.total_reference_tokens / max(screening.total_valid_tokens, 1)
        )
        metrics["kondo_actual_backward_token_fraction"] = 1.0
        metrics["kondo_kept_row_token_fraction"] = 1.0
        return train_data, metrics

    prompt_group_id = train_data.get("prompt_group_id")
    if prompt_group_id is not None:
        row_keep_prob = _solve_stochastic_row_keep_probabilities_by_group(
            row_signal=screening.row_sampling_priority,
            row_cost=screening.row_cost,
            prompt_group_id=prompt_group_id.to(dtype=torch.int64),
            target_token_fraction=cfg["target_backward_token_fraction"],
            total_valid_tokens=screening.total_valid_tokens,
            min_keep_probability=cfg["min_keep_probability"],
        )
        selected_rows = _sample_stochastic_groups(
            keep_prob=row_keep_prob,
            row_cost=screening.row_cost,
            prompt_group_id=prompt_group_id.to(dtype=torch.int64),
        )
    else:
        row_keep_prob = _solve_stochastic_row_keep_probabilities(
            row_signal=screening.row_sampling_priority,
            row_cost=screening.row_cost,
            target_token_fraction=cfg["target_backward_token_fraction"],
            total_valid_tokens=screening.total_valid_tokens,
            min_keep_probability=cfg["min_keep_probability"],
        )
        selected_rows = _sample_stochastic_rows(
            keep_prob=row_keep_prob,
            row_cost=screening.row_cost,
        )
    if selected_rows.numel() == 0:
        metrics["kondo_bypass_step"] = 1.0
        return train_data, metrics

    policy_train_data = train_data.select_indices(selected_rows)
    full_batch_normalizer = torch.full(
        (selected_rows.numel(),),
        float(screening.total_valid_tokens),
        dtype=torch.float32,
    )
    policy_train_data["loss_normalizer"] = full_batch_normalizer
    policy_train_data["kl_normalizer"] = full_batch_normalizer.clone()
    selected_keep_prob = row_keep_prob[selected_rows].clamp(
        min=cfg["min_keep_probability"]
    )
    policy_train_data["sample_loss_weight"] = 1.0 / selected_keep_prob

    row_selected = torch.zeros_like(train_data["sample_mask"], dtype=torch.float32)
    row_selected[selected_rows] = 1.0
    train_data["kondo_row_selected"] = row_selected
    train_data["kondo_row_keep_prob"] = row_keep_prob.to(dtype=torch.float32)
    row_ht_weight = torch.zeros_like(train_data["sample_mask"], dtype=torch.float32)
    row_ht_weight[selected_rows] = (1.0 / selected_keep_prob).to(
        dtype=row_ht_weight.dtype
    )
    train_data["kondo_row_ht_weight"] = row_ht_weight

    kept_valid_tokens = screening.row_cost[selected_rows].sum().item()
    kept_reference_tokens = screening.row_reference_tokens[selected_rows].sum().item()
    metrics["kondo_rows_kept"] = float(selected_rows.numel())
    metrics["kondo_selected_token_recall"] = float(
        kept_reference_tokens / reference_selected_tokens
    )
    metrics["kondo_selected_token_fraction"] = float(
        kept_reference_tokens / max(screening.total_valid_tokens, 1)
    )
    metrics["kondo_actual_backward_token_fraction"] = float(
        kept_valid_tokens / max(screening.total_valid_tokens, 1)
    )
    metrics["kondo_kept_row_token_fraction"] = float(
        kept_valid_tokens / max(screening.total_valid_tokens, 1)
    )
    metrics["kondo_recall_floor_satisfied"] = 1.0
    valid_keep_prob = row_keep_prob[screening.row_cost > 0]
    metrics["kondo_row_keep_probability_mean"] = float(valid_keep_prob.mean().item())
    metrics["kondo_row_ht_weight_mean"] = float(
        (1.0 / selected_keep_prob).mean().item()
    )
    return policy_train_data, metrics

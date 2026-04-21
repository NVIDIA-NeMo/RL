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
    "block_dense_cover",
    "routed",
    "response_dense_rows",
    "response_dense_rows_v2",
    "response_block_rows",
    "response_routed",
    "oracle_dense_rows",
    "oracle_block_rows",
    "oracle_full_rows",
]
KondoPriorityMode = Literal[
    "delight",
    "advantage",
    "abs_advantage",
    "surprisal",
    "uniform",
]


class KondoConfig(TypedDict):
    enabled: bool
    mode: NotRequired[KondoMode]
    target_backward_token_fraction: NotRequired[float]
    recall_floor: NotRequired[float]
    min_selected_token_recall: NotRequired[float]
    priority_mode: NotRequired[KondoPriorityMode]
    block_size: NotRequired[int]
    min_selected_rows: NotRequired[int]
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
    row_utility: torch.Tensor
    row_response_utility: torch.Tensor
    row_response_density: torch.Tensor
    row_reference_tokens: torch.Tensor
    row_abs_priority: torch.Tensor
    total_valid_tokens: int
    total_reference_tokens: int
    screenable_token_count: int
    nonfinite_screen_token_count: int
    rows_total: int
    should_bypass: bool


def resolve_kondo_config(raw_cfg: dict[str, Any] | None) -> KondoConfig:
    cfg: KondoConfig = {
        "enabled": False,
        "mode": "response_dense_rows_v2",
        "target_backward_token_fraction": 0.5,
        "recall_floor": 1.0,
        "priority_mode": "delight",
        "block_size": 8,
        "min_selected_rows": 1,
        "bypass_on_nonfinite": True,
    }
    if raw_cfg is not None:
        cfg.update(raw_cfg)
    if "recall_floor" not in cfg and "min_selected_token_recall" in cfg:
        cfg["recall_floor"] = cfg["min_selected_token_recall"]

    mode = cfg["mode"]
    if mode not in {
        "off",
        "dense_reference",
        "block_dense_cover",
        "routed",
        "response_dense_rows",
        "response_dense_rows_v2",
        "response_block_rows",
        "response_routed",
        "oracle_dense_rows",
        "oracle_block_rows",
        "oracle_full_rows",
    }:
        raise ValueError(f"Unsupported Kondo mode: {mode}")
    fraction = cfg["target_backward_token_fraction"]
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            "grpo.kondo.target_backward_token_fraction must be in (0, 1]"
        )
    if mode == "response_dense_rows_v2":
        if "recall_floor" not in cfg:
            cfg["recall_floor"] = 1.0
    else:
        cfg.setdefault("recall_floor", 0.0)
    recall_floor = cfg["recall_floor"]
    if not (0.0 <= recall_floor <= 1.0):
        raise ValueError("grpo.kondo.recall_floor must be in [0, 1]")
    if mode == "response_dense_rows_v2" and recall_floor <= 0.0:
        raise ValueError("grpo.kondo.recall_floor must be in (0, 1] for v2")
    if cfg["min_selected_rows"] < 1:
        raise ValueError("grpo.kondo.min_selected_rows must be at least 1")
    if cfg["block_size"] < 1:
        raise ValueError("grpo.kondo.block_size must be at least 1")
    priority_mode = cfg["priority_mode"]
    if priority_mode not in {
        "delight",
        "advantage",
        "abs_advantage",
        "surprisal",
        "uniform",
    }:
        raise ValueError(f"Unsupported Kondo priority mode: {priority_mode}")
    return cfg


def kondo_mode_metric(mode: KondoMode) -> float:
    if mode == "off":
        return 0.0
    if mode == "dense_reference":
        return 1.0
    if mode == "block_dense_cover":
        return 1.5
    if mode == "routed":
        return 2.0
    if mode == "response_dense_rows":
        return 2.5
    if mode == "response_dense_rows_v2":
        return 2.6
    if mode == "response_block_rows":
        return 2.75
    if mode == "response_routed":
        return 3.0
    if mode == "oracle_dense_rows":
        return 4.0
    if mode == "oracle_block_rows":
        return 4.5
    if mode == "oracle_full_rows":
        return 5.0
    raise ValueError(f"Unsupported Kondo mode: {mode}")


def disabled_kondo_metrics(mode: KondoMode = "off") -> dict[str, float]:
    return {
        "kondo_enabled": 0.0,
        "kondo_mode": kondo_mode_metric(mode),
        "kondo_target_backward_token_fraction": 0.0,
        "kondo_recall_floor": 0.0,
        "kondo_recall_floor_satisfied": 0.0,
        "kondo_rows_kept": 0.0,
        "kondo_rows_total": 0.0,
        "kondo_selected_token_recall": 0.0,
        "kondo_selected_token_fraction": 0.0,
        "kondo_actual_backward_token_fraction": 0.0,
        "kondo_kept_row_token_fraction": 0.0,
        "kondo_nonfinite_screen_token_count": 0.0,
        "kondo_block_size": 0.0,
        "kondo_bypass_step": 0.0,
    }


def compute_token_priority(
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
    priority: torch.Tensor,
    screen_mask: torch.Tensor,
    target_fraction: float,
) -> tuple[torch.Tensor, int]:
    gate = torch.zeros_like(priority, dtype=torch.float32)
    flat_mask = screen_mask.reshape(-1)
    candidate_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)
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


def _cover_reference_tokens_with_blocks(
    reference_token_gate: torch.Tensor,
    cover_token_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    covered = torch.zeros_like(reference_token_gate, dtype=torch.float32)
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    for row_idx in range(reference_token_gate.shape[0]):
        cover_positions = cover_token_mask[row_idx].nonzero(as_tuple=False).squeeze(-1)
        if cover_positions.numel() == 0:
            continue

        selected_ordinals = (
            reference_token_gate[row_idx, cover_positions] > 0
        ).nonzero(as_tuple=False).squeeze(-1)
        if selected_ordinals.numel() == 0:
            continue

        block_ids = torch.unique(selected_ordinals // block_size)
        for block_id in block_ids.tolist():
            start = block_id * block_size
            end = min(start + block_size, cover_positions.numel())
            covered[row_idx, cover_positions[start:end]] = 1.0

    return covered


def screen_kondo_tokens(
    train_data: BatchedDataDict[Any],
    cfg: KondoConfig,
) -> KondoScreeningResult | None:
    token_mask = train_data["token_mask"][:, 1:]
    sample_mask = train_data["sample_mask"].unsqueeze(-1)
    advantages = train_data["advantages"][:, 1:]
    prev_logprobs = train_data["prev_logprobs"][:, 1:]

    valid_token_mask = (token_mask * sample_mask).bool()
    total_valid_tokens = int(valid_token_mask.sum().item())
    row_cost = valid_token_mask.sum(dim=-1).to(torch.int64)
    rows_total = int((row_cost > 0).sum().item())
    if total_valid_tokens == 0 or rows_total == 0:
        return None

    finite_screen_mask = (
        valid_token_mask
        & torch.isfinite(prev_logprobs)
        & torch.isfinite(advantages)
    )
    nonfinite_screen_token_count = total_valid_tokens - int(
        finite_screen_mask.sum().item()
    )
    screenable_token_count = int(finite_screen_mask.sum().item())
    surprisal = torch.zeros_like(prev_logprobs)
    priority = torch.zeros_like(prev_logprobs)
    reference_token_gate = torch.zeros_like(prev_logprobs, dtype=torch.float32)
    if screenable_token_count > 0:
        surprisal[finite_screen_mask] = (-prev_logprobs.detach())[finite_screen_mask]
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
    row_utility = (reference_token_gate * priority).sum(dim=-1)
    row_response_utility = (priority * finite_screen_mask).sum(dim=-1)

    loss_token_mask = torch.zeros_like(
        train_data["token_mask"], dtype=train_data["token_mask"].dtype
    )
    loss_token_mask[:, 1:] = reference_token_gate.to(dtype=loss_token_mask.dtype)
    batch_size = train_data["sample_mask"].shape[0]
    loss_normalizer = torch.full(
        (batch_size,),
        float(total_valid_tokens),
        dtype=torch.float32,
    )
    kl_normalizer = loss_normalizer.clone()
    row_reference_tokens = reference_token_gate.sum(dim=-1).to(torch.int64)
    row_abs_priority = (reference_token_gate * priority.abs()).sum(dim=-1)
    row_response_density = row_response_utility / row_cost.clamp(min=1).to(
        row_response_utility.dtype
    )
    total_reference_tokens = int(row_reference_tokens.sum().item())

    return KondoScreeningResult(
        loss_token_mask=loss_token_mask,
        loss_normalizer=loss_normalizer,
        kl_normalizer=kl_normalizer,
        valid_token_mask=valid_token_mask,
        screen_mask=finite_screen_mask,
        reference_token_gate=reference_token_gate,
        priority=priority,
        row_cost=row_cost,
        row_utility=row_utility,
        row_response_utility=row_response_utility,
        row_response_density=row_response_density,
        row_reference_tokens=row_reference_tokens,
        row_abs_priority=row_abs_priority,
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


def _select_routed_rows(
    screening: KondoScreeningResult,
    cfg: KondoConfig,
    row_multiple: int,
    row_utility_override: torch.Tensor | None = None,
    row_token_override: torch.Tensor | None = None,
) -> torch.Tensor:
    if row_token_override is None:
        row_token_override = screening.row_cost
    positive_cost_rows = (
        (screening.row_cost > 0) & (row_token_override > 0)
    ).nonzero(as_tuple=False).squeeze(-1).tolist()
    filler_rows = (screening.row_cost == 0).nonzero(as_tuple=False).squeeze(-1).tolist()
    if not positive_cost_rows and not filler_rows:
        return torch.empty(0, dtype=torch.long)

    def sort_key(idx: int) -> tuple[float, float, int]:
        if row_utility_override is not None:
            row_utility = float(row_utility_override[idx].item())
            row_density = row_utility / max(float(screening.row_cost[idx].item()), 1.0)
        elif cfg["mode"] in {
            "response_routed",
            "response_dense_rows",
            "response_block_rows",
        }:
            row_density = float(screening.row_response_density[idx].item())
            row_utility = float(screening.row_response_utility[idx].item())
        else:
            row_cost = screening.row_cost[idx].item()
            row_utility = screening.row_utility[idx].item()
            row_density = row_utility / max(float(row_cost), 1.0)
        return (-row_density, -row_utility, idx)

    ranked_rows = sorted(positive_cost_rows, key=sort_key)
    target_budget = max(
        1,
        math.ceil(
            cfg["target_backward_token_fraction"] * screening.total_valid_tokens
        ),
    )
    selected_rows: list[int] = []
    cumulative_cost = 0
    min_rows = min(cfg["min_selected_rows"], len(ranked_rows))
    for idx in ranked_rows:
        selected_rows.append(idx)
        cumulative_cost += int(screening.row_cost[idx].item())
        enough_rows = len(selected_rows) >= min_rows
        enough_tokens = cumulative_cost >= target_budget
        aligned = len(selected_rows) % row_multiple == 0
        if enough_rows and enough_tokens and aligned:
            break

    if len(selected_rows) % row_multiple != 0:
        for idx in filler_rows:
            if idx in selected_rows:
                continue
            selected_rows.append(idx)
            if len(selected_rows) % row_multiple == 0:
                break

    if len(selected_rows) % row_multiple != 0:
        for idx in ranked_rows:
            if idx in selected_rows:
                continue
            selected_rows.append(idx)
            if len(selected_rows) % row_multiple == 0:
                break

    if len(selected_rows) % row_multiple != 0:
        raise ValueError(
            "Kondo routed row selection could not satisfy the required row multiple. "
            "Increase the batch size or reduce the routing aggressiveness."
        )

    return torch.tensor(selected_rows, dtype=torch.long)


def _select_response_dense_rows_v2(
    screening: KondoScreeningResult,
    cfg: KondoConfig,
    row_multiple: int,
) -> torch.Tensor:
    teacher_rows = (
        (screening.row_reference_tokens > 0) & (screening.row_cost > 0)
    ).nonzero(as_tuple=False).squeeze(-1)
    if teacher_rows.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    def teacher_sort_key(idx: int) -> tuple[float, float, float, int]:
        row_cost = max(float(screening.row_cost[idx].item()), 1.0)
        teacher_tokens = float(screening.row_reference_tokens[idx].item())
        return (
            -(teacher_tokens / row_cost),
            -teacher_tokens,
            -float(screening.row_abs_priority[idx].item()),
            idx,
        )

    ranked_teacher_rows = sorted(teacher_rows.tolist(), key=teacher_sort_key)
    target_teacher_tokens = max(
        1,
        math.ceil(cfg["recall_floor"] * screening.total_reference_tokens),
    )
    selected_rows: list[int] = []
    covered_teacher_tokens = 0
    min_rows = min(cfg["min_selected_rows"], len(ranked_teacher_rows))
    for idx in ranked_teacher_rows:
        selected_rows.append(idx)
        covered_teacher_tokens += int(screening.row_reference_tokens[idx].item())
        enough_rows = len(selected_rows) >= min_rows
        enough_recall = covered_teacher_tokens >= target_teacher_tokens
        if enough_rows and enough_recall:
            break

    if len(selected_rows) % row_multiple != 0:
        selected_set = set(selected_rows)
        filler_rows = [
            idx
            for idx in range(screening.row_cost.shape[0])
            if idx not in selected_set and int(screening.row_reference_tokens[idx].item()) == 0
        ]
        filler_rows.sort(key=lambda idx: (int(screening.row_cost[idx].item()), idx))
        for idx in filler_rows:
            selected_rows.append(idx)
            if len(selected_rows) % row_multiple == 0:
                break

    if len(selected_rows) % row_multiple != 0:
        selected_set = set(selected_rows)
        filler_teacher_rows = [
            idx for idx in ranked_teacher_rows if idx not in selected_set
        ]
        filler_teacher_rows.sort(
            key=lambda idx: (int(screening.row_cost[idx].item()), idx)
        )
        for idx in filler_teacher_rows:
            selected_rows.append(idx)
            if len(selected_rows) % row_multiple == 0:
                break

    if len(selected_rows) % row_multiple != 0:
        raise ValueError(
            "response_dense_rows_v2 could not satisfy the required row multiple. "
            "Increase the batch size or reduce the alignment constraint."
        )

    return torch.tensor(selected_rows, dtype=torch.long)


def _select_oracle_dense_rows(
    screening: KondoScreeningResult,
    row_multiple: int,
) -> torch.Tensor:
    return _select_oracle_mask_rows(
        mask_rows=screening.row_reference_tokens,
        row_cost=screening.row_cost,
        row_multiple=row_multiple,
    )


def _select_oracle_mask_rows(
    mask_rows: torch.Tensor,
    row_cost: torch.Tensor,
    row_multiple: int,
) -> torch.Tensor:
    selected_rows = (mask_rows > 0).nonzero(as_tuple=False).squeeze(-1)
    if selected_rows.numel() == 0:
        return selected_rows.to(dtype=torch.long)

    if selected_rows.numel() % row_multiple == 0:
        return selected_rows.to(dtype=torch.long)

    filler_rows = (
        (row_cost == 0) & (mask_rows == 0)
    ).nonzero(as_tuple=False).squeeze(-1)
    if filler_rows.numel() > 0:
        needed = row_multiple - (selected_rows.numel() % row_multiple)
        filler_rows = filler_rows[:needed]
        if filler_rows.numel() == needed:
            return torch.cat(
                [selected_rows.to(dtype=torch.long), filler_rows.to(dtype=torch.long)]
            )

    raise ValueError(
        "oracle_dense_rows could not satisfy the required row multiple without "
        "adding non-reference rows. Reduce DP alignment constraints or use "
        "dense_reference for this configuration."
    )


def apply_kondo_mode(
    train_data: BatchedDataDict[Any],
    screening: KondoScreeningResult | None,
    cfg: KondoConfig,
    row_multiple: int,
) -> tuple[BatchedDataDict[Any], dict[str, float]]:
    metrics = {
        "kondo_enabled": 1.0,
        "kondo_mode": kondo_mode_metric(cfg["mode"]),
        "kondo_target_backward_token_fraction": cfg[
            "target_backward_token_fraction"
        ],
        "kondo_recall_floor": cfg["recall_floor"],
        "kondo_recall_floor_satisfied": 0.0,
        "kondo_rows_kept": 0.0,
        "kondo_rows_total": 0.0,
        "kondo_selected_token_recall": 0.0,
        "kondo_selected_token_fraction": 0.0,
        "kondo_actual_backward_token_fraction": 0.0,
        "kondo_kept_row_token_fraction": 0.0,
        "kondo_nonfinite_screen_token_count": 0.0,
        "kondo_block_size": 0.0,
        "kondo_bypass_step": 0.0,
    }
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

    reference_selected_tokens = float(screening.total_reference_tokens)
    block_reference_token_gate = None
    block_loss_token_mask = None
    block_row_tokens = None
    if cfg["mode"] in {
        "block_dense_cover",
        "response_block_rows",
        "oracle_block_rows",
    }:
        block_reference_token_gate = _cover_reference_tokens_with_blocks(
            reference_token_gate=screening.reference_token_gate,
            cover_token_mask=screening.screen_mask,
            block_size=cfg["block_size"],
        )
        block_loss_token_mask = torch.zeros_like(
            train_data["token_mask"], dtype=train_data["token_mask"].dtype
        )
        block_loss_token_mask[:, 1:] = block_reference_token_gate.to(
            dtype=block_loss_token_mask.dtype
        )
        block_row_tokens = block_reference_token_gate.sum(dim=-1).to(torch.int64)
        block_row_utility = (block_reference_token_gate * screening.priority).sum(dim=-1)
        metrics["kondo_block_size"] = float(cfg["block_size"])
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
            reference_selected_tokens / max(screening.total_valid_tokens, 1)
        )
        metrics["kondo_actual_backward_token_fraction"] = 1.0
        metrics["kondo_kept_row_token_fraction"] = 1.0
        return train_data, metrics

    if cfg["mode"] == "block_dense_cover":
        assert block_loss_token_mask is not None
        assert block_reference_token_gate is not None
        block_selected_tokens = float(block_reference_token_gate.sum().item())
        train_data["loss_token_mask"] = block_loss_token_mask
        train_data["loss_normalizer"] = screening.loss_normalizer
        train_data["kl_normalizer"] = screening.kl_normalizer
        row_selected = (screening.row_cost > 0).to(dtype=torch.float32)
        train_data["kondo_row_selected"] = row_selected
        metrics["kondo_rows_kept"] = float(screening.rows_total)
        metrics["kondo_selected_token_recall"] = (
            1.0 if reference_selected_tokens > 0 else 0.0
        )
        metrics["kondo_selected_token_fraction"] = float(
            block_selected_tokens / max(screening.total_valid_tokens, 1)
        )
        metrics["kondo_actual_backward_token_fraction"] = 1.0
        metrics["kondo_kept_row_token_fraction"] = 1.0
        return train_data, metrics

    if cfg["mode"] in {
        "oracle_dense_rows",
        "oracle_block_rows",
        "oracle_full_rows",
    }:
        if cfg["mode"] == "oracle_dense_rows":
            selected_rows = _select_oracle_dense_rows(
                screening=screening,
                row_multiple=row_multiple,
            )
        elif cfg["mode"] == "oracle_block_rows":
            assert block_row_tokens is not None
            selected_rows = _select_oracle_mask_rows(
                mask_rows=block_row_tokens,
                row_cost=screening.row_cost,
                row_multiple=row_multiple,
            )
        else:
            selected_rows = _select_oracle_dense_rows(
                screening=screening,
                row_multiple=row_multiple,
            )
        if selected_rows.numel() == 0:
            metrics["kondo_bypass_step"] = 1.0
            return train_data, metrics

        policy_train_data = train_data.select_indices(selected_rows)
        if cfg["mode"] == "oracle_dense_rows":
            policy_train_data["loss_token_mask"] = screening.loss_token_mask[
                selected_rows
            ]
            policy_train_data["loss_normalizer"] = screening.loss_normalizer[
                selected_rows
            ]
            policy_train_data["kl_normalizer"] = screening.kl_normalizer[
                selected_rows
            ]
        elif cfg["mode"] == "oracle_block_rows":
            assert block_loss_token_mask is not None
            assert block_reference_token_gate is not None
            assert block_row_tokens is not None
            policy_train_data["loss_token_mask"] = block_loss_token_mask[selected_rows]
            policy_train_data["loss_normalizer"] = screening.loss_normalizer[
                selected_rows
            ]
            policy_train_data["kl_normalizer"] = screening.kl_normalizer[
                selected_rows
            ]
        else:
            policy_train_data["loss_normalizer"] = screening.loss_normalizer[
                selected_rows
            ]
            policy_train_data["kl_normalizer"] = screening.kl_normalizer[
                selected_rows
            ]

        row_selected = torch.zeros_like(train_data["sample_mask"], dtype=torch.float32)
        row_selected[selected_rows] = 1.0
        train_data["kondo_row_selected"] = row_selected

        kept_valid_tokens = screening.row_cost[selected_rows].sum().item()
        kept_reference_tokens = screening.row_reference_tokens[selected_rows].sum().item()
        metrics["kondo_rows_kept"] = float(selected_rows.numel())
        if cfg["mode"] == "oracle_block_rows":
            kept_block_tokens = block_row_tokens[selected_rows].sum().item()
            metrics["kondo_selected_token_recall"] = (
                1.0 if reference_selected_tokens > 0 else 0.0
            )
            metrics["kondo_selected_token_fraction"] = float(
                kept_block_tokens / max(screening.total_valid_tokens, 1)
            )
            metrics["kondo_actual_backward_token_fraction"] = float(
                kept_valid_tokens / max(screening.total_valid_tokens, 1)
            )
        else:
            metrics["kondo_selected_token_recall"] = (
                float(kept_reference_tokens / reference_selected_tokens)
                if reference_selected_tokens > 0
                else 0.0
            )
            metrics["kondo_selected_token_fraction"] = float(
                kept_reference_tokens / max(screening.total_valid_tokens, 1)
            )
            if cfg["mode"] == "oracle_dense_rows":
                metrics["kondo_actual_backward_token_fraction"] = float(
                    kept_valid_tokens / max(screening.total_valid_tokens, 1)
                )
            else:
                metrics["kondo_actual_backward_token_fraction"] = float(
                    kept_valid_tokens / max(screening.total_valid_tokens, 1)
                )
        metrics["kondo_kept_row_token_fraction"] = float(
            kept_valid_tokens / max(screening.total_valid_tokens, 1)
        )
        metrics["kondo_recall_floor_satisfied"] = float(
            metrics["kondo_selected_token_recall"] + 1e-9 >= cfg["recall_floor"]
        )
        return policy_train_data, metrics

    if cfg["mode"] == "response_dense_rows_v2":
        selected_rows = _select_response_dense_rows_v2(
            screening=screening,
            cfg=cfg,
            row_multiple=row_multiple,
        )
    else:
        selected_rows = _select_routed_rows(
            screening=screening,
            cfg=cfg,
            row_multiple=row_multiple,
            row_utility_override=(
                block_row_utility if cfg["mode"] == "response_block_rows" else None
            ),
            row_token_override=(
                block_row_tokens if cfg["mode"] == "response_block_rows" else None
            ),
        )
    if selected_rows.numel() == 0:
        metrics["kondo_bypass_step"] = 1.0
        return train_data, metrics
    if cfg["mode"] in {"routed", "response_dense_rows", "response_dense_rows_v2"}:
        train_data["loss_token_mask"] = screening.loss_token_mask
        train_data["loss_normalizer"] = screening.loss_normalizer
        train_data["kl_normalizer"] = screening.kl_normalizer
    elif cfg["mode"] == "response_block_rows":
        assert block_loss_token_mask is not None
        train_data["loss_token_mask"] = block_loss_token_mask
        train_data["loss_normalizer"] = screening.loss_normalizer
        train_data["kl_normalizer"] = screening.kl_normalizer
    policy_train_data = train_data.select_indices(selected_rows)
    if cfg["mode"] == "response_routed":
        # Keep the objective scale matched to the original full batch even
        # though we only backprop through the compacted kept rows.
        policy_train_data["loss_normalizer"] = screening.loss_normalizer[
            selected_rows
        ]
        policy_train_data["kl_normalizer"] = screening.kl_normalizer[selected_rows]
    row_selected = torch.zeros_like(train_data["sample_mask"], dtype=torch.float32)
    row_selected[selected_rows] = 1.0
    train_data["kondo_row_selected"] = row_selected

    kept_valid_tokens = screening.row_cost[selected_rows].sum().item()
    kept_reference_tokens = screening.row_reference_tokens[selected_rows].sum().item()
    metrics["kondo_rows_kept"] = float(selected_rows.numel())
    if cfg["mode"] == "response_routed":
        metrics["kondo_selected_token_recall"] = (
            float(kept_reference_tokens / reference_selected_tokens)
            if reference_selected_tokens > 0
            else 0.0
        )
        metrics["kondo_selected_token_fraction"] = float(
            kept_reference_tokens / max(screening.total_valid_tokens, 1)
        )
        metrics["kondo_actual_backward_token_fraction"] = float(
            kept_valid_tokens / max(screening.total_valid_tokens, 1)
        )
    elif cfg["mode"] == "response_block_rows":
        assert block_row_tokens is not None
        kept_block_tokens = block_row_tokens[selected_rows].sum().item()
        metrics["kondo_selected_token_recall"] = (
            float(kept_reference_tokens / reference_selected_tokens)
            if reference_selected_tokens > 0
            else 0.0
        )
        metrics["kondo_selected_token_fraction"] = float(
            kept_block_tokens / max(screening.total_valid_tokens, 1)
        )
        metrics["kondo_actual_backward_token_fraction"] = float(
            kept_valid_tokens / max(screening.total_valid_tokens, 1)
        )
    else:
        metrics["kondo_selected_token_recall"] = (
            float(kept_reference_tokens / reference_selected_tokens)
            if reference_selected_tokens > 0
            else 0.0
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
    metrics["kondo_recall_floor_satisfied"] = float(
        metrics["kondo_selected_token_recall"] + 1e-9 >= cfg["recall_floor"]
    )
    return policy_train_data, metrics

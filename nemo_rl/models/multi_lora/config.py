"""Multi-LoRA adapter and config dataclasses.

See docs/multi-lora-implementation-plan.md §1.2. These are plain, frozen
dataclasses — no NeMo-RL or torch imports here so they stay cheap to load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MultiLoRAAdapter:
    """A single LoRA adapter slot in a multi-adapter run."""

    name: str
    lora_cfg: dict          # NeMo-RL lora_cfg block, verbatim
    optimizer: dict         # NeMo-RL optimizer block, verbatim
    data: dict              # NeMo-RL data block, verbatim
    weight: float = 1.0
    nemo_gym: str | None = None  # reserved for Phase 3 (RL)
    pin: Literal["gpu", "cpu", "auto"] = "auto"


@dataclass(frozen=True)
class MultiLoRAConfig:
    """Top-level multi-LoRA config block."""

    enabled: bool
    schedule: Literal["round_robin", "weighted", "progress_aware"] = "round_robin"
    global_batch_size: int = 32
    # Number of consecutive steps the trainer should stay on a single adapter
    # (or group, in concurrent mode) before re-querying the scheduler. Reduces
    # weight-swap pressure (sequential mode) or group churn (concurrent mode).
    # K=1 (default) = re-query every step.
    steps_per_adapter: int = 1
    # Where inactive-adapter state lives. "cpu" offloads (saves GPU memory,
    # adds a copy per swap); "cuda" keeps every adapter resident on GPU
    # (zero swap latency, ~tens of MB per adapter at rank=16). Default
    # "cpu" stays safe for big models / many adapters.
    storage_device: Literal["cpu", "cuda"] = "cpu"
    # Execution mode. "sequential" (default) = Phase A weight-swap loop. "concurrent"
    # = Phase B token-packed forward (one base pass serves all GPU-pinned adapters).
    execution_mode: Literal["sequential", "concurrent"] = "sequential"
    # HBM budget for the memory planner (concurrent mode only). ``None`` = auto-detect
    # from torch.cuda.mem_get_info at runtime.
    hbm_budget_gb: float | None = None
    # Per-adapter micro batch size for the NeMo-RL multi-adapter SFT path
    # (round-robin packer in :mod:`nemo_rl.models.multi_lora.data`). The
    # global packed batch is ``len(adapters) * batch_size_per_adapter`` rows.
    batch_size_per_adapter: int = 1
    adapters: list[MultiLoRAAdapter] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "MultiLoRAConfig":
        adapters = [MultiLoRAAdapter(**a) for a in d.get("adapters", [])]
        return cls(
            enabled=d.get("enabled", False),
            schedule=d.get("schedule", "round_robin"),
            global_batch_size=d.get("global_batch_size", 32),
            steps_per_adapter=d.get("steps_per_adapter", 1),
            storage_device=d.get("storage_device", "cpu"),
            execution_mode=d.get("execution_mode", "sequential"),
            hbm_budget_gb=d.get("hbm_budget_gb", None),
            batch_size_per_adapter=d.get("batch_size_per_adapter", 1),
            adapters=adapters,
        )

    def validate(self) -> None:
        if len(self.adapters) < 1:
            raise ValueError("multi_lora.adapters must have at least 1 entry")
        names = [a.name for a in self.adapters]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate adapter names: {names}")
        if self.storage_device not in ("cpu", "cuda"):
            raise ValueError(
                f"multi_lora.storage_device must be 'cpu' or 'cuda', got "
                f"{self.storage_device!r}"
            )
        if self.steps_per_adapter < 1:
            raise ValueError(
                f"multi_lora.steps_per_adapter must be >= 1, got "
                f"{self.steps_per_adapter}"
            )
        if self.execution_mode not in ("sequential", "concurrent"):
            raise ValueError(
                f"multi_lora.execution_mode must be 'sequential' | 'concurrent', "
                f"got {self.execution_mode!r}"
            )
        if self.hbm_budget_gb is not None and self.hbm_budget_gb <= 0:
            raise ValueError(
                f"multi_lora.hbm_budget_gb must be positive when set, got "
                f"{self.hbm_budget_gb}"
            )
        if self.batch_size_per_adapter < 1:
            raise ValueError(
                f"multi_lora.batch_size_per_adapter must be >= 1, got "
                f"{self.batch_size_per_adapter}"
            )
        for a in self.adapters:
            if a.pin not in ("gpu", "cpu", "auto"):
                raise ValueError(
                    f"adapter {a.name!r} pin must be 'gpu' | 'cpu' | 'auto', "
                    f"got {a.pin!r}"
                )

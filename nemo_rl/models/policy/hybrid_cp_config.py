"""Configuration for Hybrid Context Parallelism (HCP) feature."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class HybridCPConfig:
    """Configuration for Hybrid Context Parallelism.

    HCP allows each sample to use a different CP size based on its sequence length,
    enabling better load balancing across DP×CP ranks.

    Args:
        enabled: Whether to enable HCP
        max_seqlen_per_dp_cp_rank: Maximum sequence length per DP×CP rank.
            Used to determine CP size for each sample: cp_size = ceil(seq_len / max_seqlen_per_dp_cp_rank)
            Default: None (will be calculated as max_seq_len / global_cp_size)
        scheduling_strategy: Strategy for balancing workload ("dp" or "pp")
            - "dp": Data parallel strategy (default)
            - "pp": Pipeline parallel strategy (not yet supported)
        balance_slack: Balance slack parameter for scheduler (default: 0.05 = 5%)
        eps_bucket: Epsilon target for bucket balance (default: 0.10 = 10%)
    """

    enabled: bool = False
    max_seqlen_per_dp_cp_rank: Optional[int] = None
    scheduling_strategy: Literal["dp", "pp"] = "dp"
    balance_slack: float = 0.05
    eps_bucket: float = 0.10

    def __post_init__(self):
        """Validate configuration."""
        if self.enabled:
            if self.scheduling_strategy not in ["dp", "pp"]:
                raise ValueError(
                    f"scheduling_strategy must be 'dp' or 'pp', got {self.scheduling_strategy}"
                )
            if self.scheduling_strategy == "pp":
                raise NotImplementedError("Pipeline parallel strategy is not yet supported for HCP")

            if self.balance_slack < 0 or self.balance_slack > 1:
                raise ValueError(f"balance_slack must be between 0 and 1, got {self.balance_slack}")

            if self.eps_bucket < 0 or self.eps_bucket > 1:
                raise ValueError(f"eps_bucket must be between 0 and 1, got {self.eps_bucket}")

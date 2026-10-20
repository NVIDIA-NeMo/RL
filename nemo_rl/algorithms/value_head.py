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
"""The PPO critic's output head: scalar (MSE) or HL-Gauss categorical (cross-entropy).

HL-Gauss (Imani & White 2018; Farebrother et al. 2024, "Stop Regressing"; for LLM PPO
critics, Zhou et al. 2026, "Start Classifying: Categorical Critics for LLM Reinforcement
Learning", arXiv 2608.02181): the head outputs ``num_bins`` logits over equal-width bins
on ``[v_min, v_max]`` and is trained by cross-entropy against the return smoothed by a
Gaussian of width ``sigma`` (``HLGaussValueLossFn``). The critic's value is the
expectation over the bin centers, so GAE, the actor and every other value consumer see
one scalar per token, as with the MSE head.

Kept free of model/worker imports: the loss, the workers and the config checks all use it.
"""

from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import torch
import yaml
from pydantic import BaseModel, Field, model_validator

# Key of the head block in the value config: ``value.head``.
CONFIG_KEY = "head"


class ValueHeadConfig(BaseModel, extra="forbid"):
    """``value.head``: the critic's output head. Absent means the scalar head."""

    # "scalar": one output per token, trained by MSE (MseValueLossFn).
    # "hl_gauss": num_bins logits per token, trained by cross-entropy (HLGaussValueLossFn).
    type: Literal["scalar", "hl_gauss"] = "scalar"
    # HL-Gauss support: num_bins equal-width bins on [v_min, v_max]. Returns are clamped to
    # it, so leave a margin around the reachable range; the paper uses 101 bins on
    # [-0.1, 1.1] for 0/1 rewards with gamma = lambda = 1.
    num_bins: int = Field(default=101, ge=2)
    v_min: float = -0.1
    v_max: float = 1.1
    # Width of the Gaussian that smooths each return over the bins. None: 0.75 bin widths
    # (the paper's 0.009 on the default support; it beat 1 and 2 bin widths).
    sigma: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_support(self) -> "ValueHeadConfig":
        if not self.v_min < self.v_max:
            raise ValueError(
                f"value.head needs v_min < v_max, got v_min={self.v_min}, "
                f"v_max={self.v_max}"
            )
        return self

    @property
    def categorical(self) -> bool:
        """Whether the head predicts a distribution over bins."""
        return self.type == "hl_gauss"

    @property
    def output_size(self) -> int:
        """Outputs per token: 1 for the scalar head, num_bins for HL-Gauss."""
        return self.num_bins if self.categorical else 1

    @property
    def bin_width(self) -> float:
        return (self.v_max - self.v_min) / self.num_bins

    @property
    def smoothing(self) -> float:
        """The Gaussian width actually used (sigma, or 0.75 bin widths)."""
        return self.sigma if self.sigma is not None else 0.75 * self.bin_width

    def describe(self) -> str:
        if not self.categorical:
            return "scalar"
        return (
            f"hl_gauss({self.num_bins} bins on [{self.v_min}, {self.v_max}], "
            f"sigma={self.smoothing:g})"
        )

    def weights_signature(self) -> tuple:
        """What trained head weights depend on. sigma only shapes the targets."""
        if not self.categorical:
            return ("scalar",)
        return ("hl_gauss", self.num_bins, self.v_min, self.v_max)


def resolve_value_head(value_config: Optional[Mapping[str, Any]]) -> ValueHeadConfig:
    """The parsed ``value.head`` block; the scalar head when it is absent."""
    if value_config is None or value_config.get(CONFIG_KEY) is None:
        return ValueHeadConfig()
    return ValueHeadConfig.model_validate(value_config[CONFIG_KEY])


def bin_centers(
    head: ValueHeadConfig, device: Optional[torch.device | str] = None
) -> torch.Tensor:
    """fp32 ``[num_bins]``: z_i = v_min + (i + 1/2) * bin_width."""
    index = torch.arange(head.num_bins, device=device, dtype=torch.float32)
    return head.v_min + (index + 0.5) * head.bin_width


def decode_values(head_output: torch.Tensor, head: ValueHeadConfig) -> torch.Tensor:
    """Per-token values ``[...]`` from the head output ``[..., head.output_size]``.

    The scalar head's output is the value; HL-Gauss returns the expectation of the
    softmax over the bin centers, in fp32.
    """
    if not head.categorical:
        return head_output.squeeze(-1)
    probs = torch.softmax(head_output.float(), dim=-1)
    return (probs * bin_centers(head, device=head_output.device)).sum(dim=-1)


def hl_gauss_targets(returns: torch.Tensor, head: ValueHeadConfig) -> torch.Tensor:
    """fp32 ``[..., num_bins]`` targets: the mass of N(return, sigma^2) in each bin.

    Returns are clamped to [v_min, v_max] first, then the Gaussian's mass is renormalized
    over the support (paper eq.: q_i ∝ Φ((z_i + Δ/2 - y)/σ) - Φ((z_i - Δ/2 - y)/σ)).
    """
    y = returns.float().clamp(head.v_min, head.v_max).unsqueeze(-1)
    edges = head.v_min + head.bin_width * torch.arange(
        head.num_bins + 1, device=returns.device, dtype=torch.float32
    )
    cdf = torch.special.ndtr((edges - y) / head.smoothing)
    # Far above the return both CDFs round to 1 in fp32; keep their difference >= 0.
    mass = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def check_warm_start_value_head(
    warm_start: str | Path, value_config: Optional[Mapping[str, Any]]
) -> None:
    """Refuse a critic seed whose head means something else than ``value.head``.

    A scalar seed cannot start an HL-Gauss critic (or the reverse), and an HL-Gauss seed
    only fits the same bins: the head weights are tied to them. Reads the seed's saved
    ``config.yaml``; a seed without one is not checked.
    """
    config_path = Path(warm_start) / "config.yaml"
    if not config_path.exists():
        return
    with open(config_path) as f:
        seed_config = yaml.safe_load(f) or {}
    seed = resolve_value_head(seed_config.get("value"))
    ours = resolve_value_head(value_config)
    if seed.weights_signature() != ours.weights_signature():
        raise ValueError(
            f"warm_start_value_checkpoint={str(warm_start)!r} has a {seed.describe()} "
            f"critic head but value.head is {ours.describe()}. The head weights only "
            "fit the head they were trained as: pretrain a critic with this value.head "
            "or drop the warm start."
        )

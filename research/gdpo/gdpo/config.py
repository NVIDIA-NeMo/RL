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
"""Configuration for masked diffusion language model policies."""

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field

_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(ge=1)]
_Probability = Annotated[float, Field(ge=0.0, le=1.0)]
_NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class SdmcLikelihoodConfig(BaseModel, extra="allow"):
    """Configuration for GDPO's SDMC likelihood estimator.

    Enabling this switches the policy from autoregressive next-token log
    probabilities to a sequence-level ELBO estimated with the Semi-deterministic
    Monte Carlo (SDMC) scheme of https://arxiv.org/abs/2510.08554: the ELBO's
    time integral is approximated by a fixed quadrature rule over the mask ratio
    ``t``, with a small number of Monte Carlo mask draws per quadrature point.

    The number of model forward passes per likelihood evaluation is
    ``len(quadrature points) * mc_samples``, so ``quadrature`` and ``mc_samples``
    are the primary compute/variance knobs. The paper finds ``gauss-2`` or
    ``gauss-3`` with ``mc_samples=1`` sufficient.
    """

    type: Literal["sdmc"] = "sdmc"
    quadrature: Literal[
        "gauss-1", "gauss-2", "gauss-3", "gauss-4", "gauss-5", "simpson", "mc"
    ] = "gauss-2"
    """Integration rule over the mask ratio t. One of ``gauss-1`` .. ``gauss-5``
    (deterministic Gauss-Legendre, the SDMC scheme), ``simpson``, or ``mc``
    (double Monte Carlo -- the higher-variance baseline, kept for ablations)."""

    mc_samples: _PositiveInt = 1
    """Monte Carlo mask draws per quadrature point (``K`` in the paper)."""

    p_mask_prompt: _Probability = 0.0
    """Probability of masking each prompt token, as a regularizer.

    Leave at 0.0 to reproduce GDPO. The option is inherited from the
    d1/diffu-GRPO trainer GDPO was built on. Corrupted prompt positions
    condition the model but are never scored by the SDMC estimator."""


class MaskedDiffusionConfig(BaseModel, extra="allow"):
    """Policy-side configuration for masked diffusion language models."""

    enabled: bool = False
    """Whether the policy is a masked diffusion LM."""

    mask_id: Optional[_NonNegativeInt] = None
    """Token id of the ``[MASK]`` token.

    Model-specific, and a wrong value silently corrupts the likelihood rather
    than failing, so there is no static default. Leave unset to read it from the
    model's own config (LLaDA publishes ``mask_token_id``)."""

    shift_targets: bool = False
    """Whether position ``i`` scores token ``i+1`` rather than token ``i``."""

    likelihood: SdmcLikelihoodConfig = Field(default_factory=SdmcLikelihoodConfig)
    """Sequence-likelihood estimator configuration."""


class DenoiseConfig(BaseModel, extra="allow"):
    """Generation-side block denoising configuration."""

    type: Literal["block"] = "block"
    block_length: _PositiveInt = 32
    """Block size for block-wise iterative denoising during generation."""

    diffusion_steps: _PositiveInt = 64
    """Total denoising steps per generated sequence."""

    cfg_scale: _NonNegativeFloat = 0.0
    """Unsupervised classifier-free guidance scale. 0.0 disables guidance and
    halves the number of forward passes."""


# Attributes a masked diffusion model may publish its mask token id under.
_MASK_ID_ATTRS = ("mask_token_id", "mask_id")


def resolve_mask_id(cfg: MaskedDiffusionConfig, model_config: Any) -> int:
    """Resolves the mask token id from the config, falling back to the model.

    An explicit ``cfg.mask_id`` wins so a model with a mislabeled config can be
    corrected, but the model's own value is preferred over asking the user to
    retype a constant they cannot verify.

    Args:
        cfg: The masked-diffusion policy configuration.
        model_config: The Hugging Face model config to read ``mask_token_id``
            from when ``cfg.mask_id`` is unset.

    Returns:
        The mask token id to substitute at masked positions.

    Raises:
        ValueError: If neither the config nor the model supplies a mask token id.
    """
    if cfg.mask_id is not None:
        return cfg.mask_id

    for attr in _MASK_ID_ATTRS:
        value = getattr(model_config, attr, None)
        if value is not None:
            return int(value)

    raise ValueError(
        "policy.masked_diffusion.enabled is true but no mask token id is available: "
        f"policy.masked_diffusion.mask_id is unset and the model config exposes none of "
        f"{_MASK_ID_ATTRS}. Set policy.masked_diffusion.mask_id explicitly (126336 for "
        "LLaDA-8B, 156895 for LLaDA2.0)."
    )

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
"""Configuration for masked diffusion language model (dLLM) policies."""

from typing import Any, Optional

from pydantic import BaseModel


class DllmConfig(BaseModel, extra="allow"):
    """Policy-side configuration for masked diffusion language models.

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

    enabled: bool = False
    """Whether the policy is a masked diffusion LM."""

    mask_id: Optional[int] = None
    """Token id of the ``[MASK]`` token.

    Model-specific, and a wrong value silently corrupts the likelihood rather
    than failing, so there is no static default. Leave unset to read it from the
    model's own config (LLaDA publishes ``mask_token_id``); set it explicitly
    only for a model that does not. :func:`resolve_mask_id` performs that
    lookup, and raises if neither source supplies one."""

    quadrature: str = "gauss-2"
    """Integration rule over the mask ratio t. One of ``gauss-1`` .. ``gauss-5``
    (deterministic Gauss-Legendre, the SDMC scheme), ``simpson``, or ``mc``
    (double Monte Carlo -- the higher-variance baseline, kept for ablations)."""

    mc_samples: int = 1
    """Monte Carlo mask draws per quadrature point (``K`` in the paper)."""

    p_mask_prompt: float = 0.0
    """Probability of masking each prompt token, as a regularizer.

    Leave at 0.0 to reproduce GDPO: its SDMC estimator masks only completion
    positions and zeroes the prompt region outright, so ``p_mask_prompt`` is
    inert there even though the key appears in the published configs -- it is
    inherited from the d1/diffu-GRPO trainer GDPO was built on, whose one-shot
    estimator does use it. Corrupted prompt positions are never scored here."""

    shift_targets: bool = False
    """Whether position ``i`` scores token ``i+1`` (autoregressive) rather than
    token ``i``. False for masked diffusion LMs, which are trained
    position-aligned -- equivalent to dFactory's ``same_token_labels=true``,
    which is what the released LLaDA2.0 configs use."""

    block_length: int = 32
    """Block size for block-wise iterative denoising during generation."""

    diffusion_steps: int = 64
    """Total denoising steps per generated sequence."""

    cfg_scale: float = 0.0
    """Unsupervised classifier-free guidance scale. 0.0 disables guidance and
    halves the number of forward passes."""


# Attributes a masked diffusion model may publish its mask token id under.
_MASK_ID_ATTRS = ("mask_token_id", "mask_id")


def resolve_mask_id(cfg: DllmConfig, model_config: Any) -> int:
    """Resolves the mask token id from the config, falling back to the model.

    An explicit ``cfg.mask_id`` wins so a model with a mislabeled config can be
    corrected, but the model's own value is preferred over asking the user to
    retype a constant they cannot verify.

    Args:
        cfg: The dLLM policy configuration.
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
        "policy.dllm.enabled is true but no mask token id is available: "
        f"policy.dllm.mask_id is unset and the model config exposes none of "
        f"{_MASK_ID_ATTRS}. Set policy.dllm.mask_id explicitly (126336 for "
        "LLaDA-8B, 156895 for LLaDA2.0)."
    )

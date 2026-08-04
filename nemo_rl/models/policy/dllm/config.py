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

    mask_id: int
    """Token id of the ``[MASK]`` token. Model-specific and required: 126336 for
    LLaDA-8B, 156895 for LLaDA2.0. A wrong value silently corrupts the
    likelihood, so there is deliberately no default."""

    quadrature: str = "gauss-2"
    """Integration rule over the mask ratio t. One of ``gauss-1`` .. ``gauss-5``
    (deterministic Gauss-Legendre, the SDMC scheme), ``simpson``, or ``mc``
    (double Monte Carlo -- the higher-variance baseline, kept for ablations)."""

    mc_samples: int = 1
    """Monte Carlo mask draws per quadrature point (``K`` in the paper)."""

    p_mask_prompt: float = 0.0
    """Probability of masking each prompt token. 0.0 keeps the prompt fully
    intact, matching the dLLM SFT objective. The d1/diffu-GRPO line of work uses
    a nonzero value as a regularizer."""

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

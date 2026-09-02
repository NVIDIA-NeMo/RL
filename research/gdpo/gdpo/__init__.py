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
"""Masked diffusion language model (dLLM) support for RL training.

Implements the sequence-level ELBO likelihood used by GDPO, "Improving
Reasoning for Diffusion Language Models via Group Diffusion Policy
Optimization", ICLR 2026 (https://openreview.net/forum?id=JaqvespRBP,
https://arxiv.org/abs/2510.08554). Not to be confused with
``grpo.adv_estimator.name = "gdpo"``, which is the unrelated multi-reward
estimator from https://arxiv.org/abs/2601.05242.
"""

from gdpo.config import (
    DenoiseConfig,
    MaskedDiffusionConfig,
    SdmcLikelihoodConfig,
    resolve_mask_id,
)
from gdpo.elbo import (
    MaskPoint,
    SdmcElboEstimator,
    accumulate_elbo_logprobs,
    get_quadrature,
    make_dllm_mask_seeds,
)
from gdpo.setup import (
    denoise_config_from_generation,
    masked_diffusion_config_from_policy,
    validate_gdpo_config,
)

__all__ = [
    "DenoiseConfig",
    "denoise_config_from_generation",
    "MaskedDiffusionConfig",
    "masked_diffusion_config_from_policy",
    "MaskPoint",
    "resolve_mask_id",
    "SdmcElboEstimator",
    "SdmcLikelihoodConfig",
    "accumulate_elbo_logprobs",
    "get_quadrature",
    "make_dllm_mask_seeds",
    "validate_gdpo_config",
]

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
"""DSpark/DFlash/EAGLE3 draft co-training on the DTensor-v2 backend.

Thin extension layer over ``nemo_automodel.components.speculative`` (3rdparty
Automodel r0.6.0): byte-identical building blocks are imported from Automodel;
only the RL-co-training deltas live in this package (see each module's header
for what it extends and why). The draft model predicts a block of tokens per
anchor position in one parallel pass, cross-attending target-model hidden
states captured from the policy's own training forward.
"""

from nemo_automodel.components.attention.dflash_mask import (  # noqa: F401
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from nemo_automodel.components.speculative.dspark._sampling import (  # noqa: F401
    sample_tokens,
)
from nemo_automodel.components.speculative.dspark.config import (  # noqa: F401
    build_draft_config,
)

from nemo_rl.models.automodel.draft.common import DSparkForwardOutput
from nemo_rl.models.automodel.draft.draft_qwen3 import Qwen3DSparkModel
from nemo_rl.models.automodel.draft.loss import compute_dspark_loss

__all__ = [
    "DSparkForwardOutput",
    "Qwen3DSparkModel",
    "build_draft_config",
    "compute_dspark_loss",
    "create_dflash_block_mask",
    "create_dflash_sdpa_mask",
    "sample_tokens",
]

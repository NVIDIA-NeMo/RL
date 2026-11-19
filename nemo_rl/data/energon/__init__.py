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

from typing import Any

from nemo_rl.data.energon.config import (
    EnergonCookerConfig,
    EnergonLoaderConfig,
    EnergonSourceConfig,
    EnergonTaskEncoderConfig,
)


def build_energon_sft_loader(**kwargs: Any) -> Any:
    """Build one rank-aware SFT loader with an optional Energon dependency."""
    # Deferred so importing the default Hugging Face SFT path does not require Energon.
    from nemo_rl.data.energon.sft_dataloader import (
        build_energon_sft_loader as _build,
    )

    return _build(**kwargs)


__all__ = [
    "EnergonCookerConfig",
    "EnergonLoaderConfig",
    "EnergonSourceConfig",
    "EnergonTaskEncoderConfig",
    "build_energon_sft_loader",
]

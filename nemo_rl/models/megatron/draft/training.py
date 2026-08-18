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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule

from nemo_rl.models.megatron.draft.utils import build_draft_model
from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

if TYPE_CHECKING:
    from megatron.bridge.models.model_provider import ModelProviderMixin


class DraftSpeculator(Protocol):
    """Model-build operation selected by a configured draft speculator."""

    config: Eagle3DraftConfig

    def build_model(
        self,
        *,
        model_provider: ModelProviderMixin,
        pg_collection: ProcessGroupCollection,
        policy_model_chunk: MegatronModule,
    ) -> MegatronModule | None:
        """Build the draft model for the policy chunk that owns it."""


@dataclass(frozen=True)
class Eagle3Speculator:
    """Current EAGLE-3 draft speculator."""

    config: Eagle3DraftConfig

    def build_model(
        self,
        *,
        model_provider: ModelProviderMixin,
        pg_collection: ProcessGroupCollection,
        policy_model_chunk: MegatronModule,
    ) -> MegatronModule | None:
        """Build the EAGLE-3 draft model with the existing implementation."""
        return build_draft_model(
            model_provider=model_provider,
            draft_config=self.config,
            pg_collection=pg_collection,
            policy_model_chunk=policy_model_chunk,
        )


_SPECULATOR_FACTORIES: dict[str, type[Eagle3Speculator]] = {
    "eagle3": Eagle3Speculator,
}


def resolve_draft_speculator(
    config: Eagle3DraftConfig | None,
) -> DraftSpeculator | None:
    """Resolve an enabled draft configuration to its speculator."""
    if config is None or not config.enabled:
        return None
    return cast(
        DraftSpeculator,
        _SPECULATOR_FACTORIES[config.speculator_type](config),
    )

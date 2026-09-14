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

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, TypeAlias

from megatron.energon import Cooker, CrudeSample, DefaultTaskEncoder

from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    EncodedSFTSample,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

SFTCooker: TypeAlias = (
    Cooker[CanonicalSFTSample] | Callable[[CrudeSample], CanonicalSFTSample]
)


class BaseSFTTaskEncoder(
    DefaultTaskEncoder[
        CanonicalSFTSample,
        EncodedSFTSample,
        BatchedDataDict[Any],
        BatchedDataDict[Any],
    ],
    ABC,
):
    """Common SFT lifecycle shared by the Energon task encoders."""

    sample_schema: ClassVar[str]

    def __init__(
        self,
        *,
        cooker_functions: Sequence[SFTCooker],
    ) -> None:
        super().__init__()
        self.cookers = tuple(
            cooker if isinstance(cooker, Cooker) else Cooker(cooker)
            for cooker in cooker_functions
        )

    @abstractmethod
    def preencode_sample(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        """Encode one sample before optional pack selection."""

    @abstractmethod
    def postencode_sample(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        """Finish the selected sample before physical packing."""

    @abstractmethod
    def batch(self, samples: list[EncodedSFTSample]) -> BatchedDataDict[Any]:
        """Combine encoded samples into one minibatch."""

    @abstractmethod
    def encode_batch(self, batch: BatchedDataDict[Any]) -> BatchedDataDict[Any]:
        """Finish one minibatch before the loader emits it."""


__all__ = ["BaseSFTTaskEncoder", "SFTCooker"]

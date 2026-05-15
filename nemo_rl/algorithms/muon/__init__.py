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
"""Muon optimizer adapter for the NeMo-RL DTensor backend."""

from nemo_rl.algorithms.muon.builder import build_dtensor_muon
from nemo_rl.algorithms.muon.chained import ChainedTorchOptimizer
from nemo_rl.algorithms.muon.dtensor_muon import DTensorMuon

__all__ = ["DTensorMuon", "ChainedTorchOptimizer", "build_dtensor_muon"]

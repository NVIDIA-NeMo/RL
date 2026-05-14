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
"""Top-level Flextron configuration consumed by the Megatron policy."""

from typing import TypedDict


class FlexRouterConfig(TypedDict):
    """Configuration for one deterministic Flextron submodel route."""

    # MLP/MoE intermediate dimension for this submodel route.
    # Set an int to apply the same dimension to all eligible layers, or a
    # list[int] with one value per main hybrid layer before setup projects it
    # to the layer types handled by the Flextron MLP hooks.
    mlp_int_list: int | list[int]
    # Hidden-state dimension for this submodel route.
    # Set an int to apply the same dimension to all eligible layers, or a
    # list[int] with one value per main hybrid layer before setup projects it
    # to the layer types handled by the Flextron hidden-state hooks.
    emb_int_list: int | list[int]


class FlextronConfig(TypedDict):
    """Top-level Flextron block consumed by the Megatron policy.

    YAML key: ``policy.flextron``.
    - ``flex_routers``: list of router specs. An empty list disables Flextron.
    - ``flextron_sampling_rates``: length ``1 + len(flex_routers)``. Index 0 is
      the un-pruned base model; entries 1..N map one-to-one to ``flex_routers``.
    - ``on_policy``: when True, generation runs under each sample's assigned
      router (existing behavior). When False, every sample is generated under
      router 0 (the base model) while logprobs and training keep their assigned
      routers — this is the off-policy regime.
    """

    flex_routers: list[FlexRouterConfig]
    flextron_sampling_rates: list[float]
    on_policy: bool

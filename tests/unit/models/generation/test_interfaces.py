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

"""Default behaviour of the optional hooks declared on GenerationInterface."""

import pytest

from nemo_rl.models.generation.interfaces import GenerationInterface


class _MinimalGeneration(GenerationInterface):
    """Satisfies the abstract methods only; inherits every optional hook."""

    def init_collective(self, ip, port, world_size, *, train_world_size):
        return []

    def generate(self, data, greedy):
        raise AssertionError("not exercised")

    def prepare_for_generation(self, *args, **kwargs):
        return True

    def finish_generation(self, *args, **kwargs):
        return True

    def shutdown(self):
        return True


def test_setup_token_capture_default_names_the_backend():
    """A backend without token capture rejects setup by name, not AttributeError."""
    with pytest.raises(NotImplementedError, match="_MinimalGeneration"):
        _MinimalGeneration().setup_token_capture({}, "staging")


def test_set_rollout_weight_version_default_names_the_backend():
    """The per-step version rotation is rejected the same way as setup."""
    with pytest.raises(NotImplementedError, match="_MinimalGeneration"):
        _MinimalGeneration().set_rollout_weight_version(1)

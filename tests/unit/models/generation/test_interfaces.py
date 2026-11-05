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
import torch

from nemo_rl.data.captured_media import MediaColumnSpec
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
    agreed_media_columns,
)


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


def test_agreed_media_columns_returns_the_one_spec_or_none():
    """Text-only workers and ranks without capture report nothing; the one
    spec the media workers pinned comes back, else ``None``."""
    spec = MediaColumnSpec(pixel_dtype=torch.bfloat16, patch_size=16)
    assert agreed_media_columns([None, spec, None]) == spec
    assert agreed_media_columns([None, None]) is None
    assert agreed_media_columns([]) is None


def test_agreed_media_columns_rejects_disagreeing_workers():
    """Two dtypes or widths on one column would be a TransferQueue schema
    conflict found mid-run; it is refused at setup instead."""
    with pytest.raises(RuntimeError, match="different media column specs"):
        agreed_media_columns(
            [
                MediaColumnSpec(pixel_dtype=torch.bfloat16, patch_size=16),
                MediaColumnSpec(pixel_dtype=torch.float32, patch_size=16),
            ]
        )

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Resolution of the per-backend sizing block.

``data_plane`` carries one block per backend (``simple:`` / ``mooncake_cpu:``)
and only the selected one is read, falling back to that backend's defaults when
absent. Getting this wrong would silently run a job at the wrong RDMA segment
size or with the staging pool off, neither of which fails loudly.
"""

from __future__ import annotations

import pydantic
import pytest
from pydantic import TypeAdapter

from nemo_rl.data_plane.interfaces import (
    DataPlaneConfig,
    MooncakeCpuConfig,
    SimpleStorageConfig,
    backend_config,
)

_BASE = {
    "enabled": True,
    "impl": "transfer_queue",
    "claim_meta_poll_interval_s": 0.5,
}


def _cfg(backend: str, **extra) -> dict:
    return {**_BASE, "backend": backend, **extra}


def test_nested_block_is_used() -> None:
    cfg = _cfg(
        "mooncake_cpu",
        mooncake_cpu={"global_segment_size": 111, "reuse_registered_buffers": False},
    )
    resolved = backend_config(cfg)
    assert isinstance(resolved, MooncakeCpuConfig)
    assert resolved.global_segment_size == 111
    assert resolved.reuse_registered_buffers is False


def test_absent_block_falls_back_to_model_defaults() -> None:
    """The point of the nesting: a config need not mention a backend it isn't using.

    Pins the literals rather than comparing against MooncakeCpuConfig()'s own
    attributes — that would hold for any value the class default was changed
    to and couldn't catch a regression of the sizing itself.
    """
    resolved = backend_config(_cfg("mooncake_cpu"))
    assert resolved.global_segment_size == 68719476736  # 64 GiB per client process
    assert resolved.local_buffer_size == 4294967296  # 4 GiB per client process
    # The opt-out flag defaults on, so omitting it must not disable the pool.
    assert resolved.reuse_registered_buffers is True


def test_accepts_an_already_coerced_model() -> None:
    """Configs arriving via pydantic have the block coerced to a model already."""
    cfg = _cfg("mooncake_cpu", mooncake_cpu=MooncakeCpuConfig(global_segment_size=555))
    assert backend_config(cfg).global_segment_size == 555


def test_partial_nested_block_keeps_other_defaults() -> None:
    cfg = _cfg("mooncake_cpu", mooncake_cpu={"local_buffer_size": 7})
    resolved = backend_config(cfg)
    assert resolved.local_buffer_size == 7
    assert resolved.global_segment_size == MooncakeCpuConfig().global_segment_size


def test_simple_backend_nested_block_is_used() -> None:
    cfg = _cfg("simple", simple={"storage_capacity": 7, "num_storage_units": 3})
    resolved = backend_config(cfg)
    assert isinstance(resolved, SimpleStorageConfig)
    assert resolved.storage_capacity == 7
    assert resolved.num_storage_units == 3


def test_simple_backend_num_storage_units_has_no_default() -> None:
    """No static default is correct across cluster sizes (see the field's
    docstring), so an absent/incomplete simple: block must raise rather than
    silently run at a node count the config never chose."""
    with pytest.raises(pydantic.ValidationError, match="num_storage_units"):
        backend_config(_cfg("simple"))
    with pytest.raises(pydantic.ValidationError, match="num_storage_units"):
        backend_config(_cfg("simple", simple={"storage_capacity": 7}))


def test_only_the_selected_backend_is_read() -> None:
    """A mooncake block must not leak into a simple run, or vice versa."""
    cfg = _cfg(
        "simple",
        simple={"storage_capacity": 5, "num_storage_units": 1},
        mooncake_cpu={"global_segment_size": 999},
    )
    resolved = backend_config(cfg)
    assert isinstance(resolved, SimpleStorageConfig)
    assert not hasattr(resolved, "global_segment_size")


def test_schema_validates_without_any_backend_block() -> None:
    """Regression guard: a required backend key is what broke SingleController CI.

    ``data_plane`` built from scratch — not inherited from the exemplar — must
    validate, otherwise MasterConfig fails before training starts.
    """
    TypeAdapter(DataPlaneConfig).validate_python(_cfg("simple"))

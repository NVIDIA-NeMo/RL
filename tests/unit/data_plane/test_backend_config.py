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
and only the selected one is read. Three shapes have to keep working at once:
the nested block, the pre-nesting flat keys that shipped on main, and neither
(meaning "this backend's defaults"). Getting the precedence wrong would
silently run a job at the wrong RDMA segment size or with the staging pool
off, neither of which fails loudly.
"""

from __future__ import annotations

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
    """The point of the nesting: a config need not mention a backend it isn't using."""
    resolved = backend_config(_cfg("mooncake_cpu"))
    assert resolved.global_segment_size == MooncakeCpuConfig().global_segment_size
    # The opt-out flag defaults on, so omitting it must not disable the pool.
    assert resolved.reuse_registered_buffers is True


def test_legacy_flat_key_raises_rather_than_being_honoured() -> None:
    """Accepting the old spelling in place would run the job on the wrong value.

    A user recipe inherits the exemplar, which supplies the nested block, so a
    surviving flat key would always lose the merge — silently. Erroring is the
    only outcome that cannot start a job at a sizing the user did not choose.
    """
    cfg = _cfg("mooncake_cpu", global_segment_size=222)
    with pytest.raises(ValueError, match="moved under data_plane.mooncake_cpu"):
        backend_config(cfg)


def test_legacy_flat_key_raises_even_beside_a_nested_block() -> None:
    """This is the case that used to lose silently."""
    cfg = _cfg(
        "mooncake_cpu",
        global_segment_size=333,
        mooncake_cpu={"global_segment_size": 444},
    )
    with pytest.raises(ValueError, match="moved under"):
        backend_config(cfg)


def test_accepts_an_already_coerced_model() -> None:
    """Configs arriving via pydantic have the block coerced to a model already."""
    cfg = _cfg("mooncake_cpu", mooncake_cpu=MooncakeCpuConfig(global_segment_size=555))
    assert backend_config(cfg).global_segment_size == 555


def test_partial_nested_block_keeps_other_defaults() -> None:
    cfg = _cfg("mooncake_cpu", mooncake_cpu={"local_buffer_size": 7})
    resolved = backend_config(cfg)
    assert resolved.local_buffer_size == 7
    assert resolved.global_segment_size == MooncakeCpuConfig().global_segment_size


@pytest.mark.parametrize(
    "cfg, expected",
    [
        (_cfg("simple", simple={"storage_capacity": 7}), 7),
        (_cfg("simple"), SimpleStorageConfig().storage_capacity),
    ],
    ids=["nested", "absent"],
)
def test_simple_backend_resolves_the_same_way(cfg, expected) -> None:
    resolved = backend_config(cfg)
    assert isinstance(resolved, SimpleStorageConfig)
    assert resolved.storage_capacity == expected


def test_only_the_selected_backend_is_read() -> None:
    """A mooncake block must not leak into a simple run, or vice versa."""
    cfg = _cfg(
        "simple",
        simple={"storage_capacity": 5},
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

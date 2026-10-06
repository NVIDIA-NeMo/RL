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
"""Shared helpers for the NIXL data-plane tests."""

from __future__ import annotations


def make_tq_conf(
    *,
    num_units: int = 2,
    slab_bytes: int = 64 << 20,
    staging_bytes: int = 32 << 20,
    read_pin_s: float = 2.0,
    namespace: str = "nrl-nixl-test-ns",
):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "controller": {"polling_mode": True},
            "backend": {
                "storage_backend": "NixlStore",
                "NixlStore": {
                    "namespace": namespace,
                    "directory_name": "BlobDirectory",
                    "num_storage_units": num_units,
                    "unit_slab_bytes": slab_bytes,
                    "client_staging_bytes": staging_bytes,
                    "read_pin_s": read_pin_s,
                    "nixl": {"backend_name": "UCX", "backend_init_params": {}},
                },
            },
        }
    )


def dense(t):
    """Pinned TQ returns every non-scalar field as a jagged NestedTensor; tests
    stack uniform rows densely."""
    import torch

    if isinstance(t, torch.Tensor) and t.is_nested:
        return torch.stack(list(t.unbind()))
    return t

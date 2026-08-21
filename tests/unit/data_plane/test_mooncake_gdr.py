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
"""GDR configuration plumbing on top of the RDMA-only Mooncake backend."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

import nemo_rl.data_plane as data_plane_module
from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin


def _gdr_cfg(*, staging_mb: int = 384) -> dict:
    return {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "mooncake_cpu",
        "claim_meta_poll_interval_s": 0.5,
        "mooncake_cpu": {
            "use_gdr": True,
            "gdr_staging_buffer_mb": staging_mb,
        },
    }


def test_init_tq_forwards_nested_gdr_config_and_keeps_rdma(
    tmp_path, monkeypatch
) -> None:
    """GDR changes destination memory, not #2935's transport selection."""
    mooncake_dir = tmp_path / "mooncake"
    mooncake_dir.mkdir()
    mooncake_init = mooncake_dir / "__init__.py"
    mooncake_init.touch()
    (mooncake_dir / "mooncake_master").touch()

    mooncake_module = ModuleType("mooncake")
    mooncake_module.__file__ = str(mooncake_init)
    mooncake_module.__path__ = [str(mooncake_dir)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", ModuleType("mooncake.store"))

    monkeypatch.setattr(tq_adapter.os, "environ", dict(tq_adapter.os.environ))
    monkeypatch.setattr(tq_adapter, "local_node_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(
        tq_adapter,
        "rdma_devices",
        lambda: "mlx5_0,mlx5_1,mlx5_2,mlx5_4",
    )
    captured: dict = {}
    monkeypatch.setattr(
        tq_adapter.tq,
        "init",
        lambda *, conf: captured.update(conf=conf),
    )

    tq_adapter._init_tq(_gdr_cfg(staging_mb=384))

    store_cfg = captured["conf"]["backend"]["MooncakeStore"]
    assert store_cfg["protocol"] == "rdma"
    assert store_cfg["device_name"] == "mlx5_0,mlx5_1,mlx5_2,mlx5_4"
    assert store_cfg["use_gdr"] is True
    assert store_cfg["gdr_staging_buffer_mb"] == 384


class _ReceiverWorker(TQWorkerMixin):
    pass


def test_cpu_only_client_may_attach_with_gdr_config(monkeypatch) -> None:
    """The CUDA guard is receiver-specific; a CPU-only producer remains valid."""
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(tq_adapter, "_connect_existing", lambda: None)
    monkeypatch.setattr(tq_adapter, "local_node_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(tq_adapter, "_patch_mooncake_register_check", lambda: None)
    # Positional on this stack: the CPU-RDMA work passes staging_buffer_size,
    # where the first GDR integration passed n_slots/max_bytes as keywords.
    monkeypatch.setattr(
        tq_adapter, "_patch_mooncake_staging_buffers", lambda *_, **__: None
    )
    monkeypatch.setattr(tq_adapter.os, "environ", dict(tq_adapter.os.environ))

    client = tq_adapter.TQDataPlaneClient(_gdr_cfg(), bootstrap=False)

    assert client._closed is False


def test_gdr_receiver_requires_cuda_initialized(monkeypatch) -> None:
    """A policy receiver must establish its CUDA context before TQ attaches."""
    worker = _ReceiverWorker()
    worker._dp_client = None
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        data_plane_module,
        "build_data_plane_client",
        lambda *args, **kwargs: pytest.fail("client built before CUDA guard"),
    )

    with pytest.raises(RuntimeError, match="CUDA must be initialized"):
        worker.setup_data_plane(_gdr_cfg())

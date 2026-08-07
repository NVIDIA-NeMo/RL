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

import pytest

from nemo_rl.data_plane.adapters.transfer_queue import _mooncake_transport_config


def test_mooncake_cpu_keeps_tcp_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MC_MOONCAKE_PROTOCOL", raising=False)

    assert _mooncake_transport_config(use_gdr=False) == {"protocol": "tcp"}


def test_mooncake_gdr_uses_rdma_with_client_local_device_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MC_MOONCAKE_PROTOCOL", raising=False)
    monkeypatch.delenv("MC_MOONCAKE_DEVICE", raising=False)

    assert _mooncake_transport_config(use_gdr=True) == {
        "protocol": "rdma",
        "device_name": "",
    }

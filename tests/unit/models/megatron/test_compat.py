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

from enum import Enum
from types import SimpleNamespace

from nemo_rl.models.megatron import compat


class _Scope(Enum):
    FULL = "full"


def test_resolve_inference_cuda_graph_scope_uses_enum(monkeypatch):
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda _name: SimpleNamespace(InferenceCudaGraphScope=_Scope),
    )

    assert compat.resolve_inference_cuda_graph_scope("FULL") is _Scope.FULL


def test_resolve_inference_cuda_graph_scope_supports_legacy_megatron(monkeypatch):
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda _name: SimpleNamespace(),
    )

    assert compat.resolve_inference_cuda_graph_scope("full") == "full"

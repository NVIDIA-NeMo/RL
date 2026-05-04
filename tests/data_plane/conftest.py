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
"""Shared fixtures for data-plane tests.

Deliberately slim. The parent ``tests/unit/conftest.py`` drags in
``mlflow``, ``torch.distributed``, ``init_ray`` etc. — none of which are
needed for data-plane Tier 1 tests. Per the test plan §11 we keep our
conftest local and minimal so unit tests run in a slim venv (torch +
tensordict + pytest only).
"""

from __future__ import annotations

import pathlib

import pytest


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    """Absolute path to the repo root (computed from this file's location)."""
    return pathlib.Path(__file__).resolve().parents[2]

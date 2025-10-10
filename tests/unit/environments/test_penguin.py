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
import pytest

try:
    from penguin import config_types  # noqa: F401

    PENGUIN_INSTALLED = True
except ImportError:
    penguin = None
    PENGUIN_INSTALLED = False


@pytest.mark.skipif(
    not PENGUIN_INSTALLED,
    reason="Skipping Penguin test since Penguin is not installed!",
)
def test_penguin_stub_module():
    print(f"Penguin test successfully run! Penguin config_types module: {config_types}")

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
"""The server-type keys NeMo RL mirrors must be the ones Gym nests under a config entry.

``GYM_SERVER_TYPE_KEYS`` is a copy of Gym's private
``nemo_gym.discovery._SERVER_GROUP_KEYS``. ``NemoGym.list_entries()`` walks
the copy to decide which config entries start a server, so a key Gym registers
and the copy lacks hides every entry of that type from the shard router's
duplicate check. This test fails as soon as the two lists differ.
"""

import pytest

from nemo_rl.environments.nemo_gym import GYM_SERVER_TYPE_KEYS

nemo_gym_discovery = pytest.importorskip("nemo_gym.discovery")

pytestmark = pytest.mark.nemo_gym


def test_server_type_keys_match_gyms_own_list():
    assert set(GYM_SERVER_TYPE_KEYS) == set(nemo_gym_discovery._SERVER_GROUP_KEYS)

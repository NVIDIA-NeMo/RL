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
"""The server-type keys NeMo RL mirrors must be the ones Gym starts servers for.

``GYM_SERVER_TYPE_KEYS`` restates the ``SERVER_TYPE`` of every member of Gym's
``nemo_gym.config_types.ServerInstanceConfig`` union, the schema Gym validates
a config entry against before it starts the entry's server. ``NemoGym.list_entries()``
walks the copy to decide which config entries start a server, so a key Gym
registers and the copy lacks hides every entry of that type from the shard
router's duplicate check. This test fails as soon as the two lists differ.

The Gym import is a module-level ``importorskip`` on the module the unit-test
conftest itself requires in the Gym lane, so a missing Gym skips the file in
the default lane while a renamed Gym symbol fails here instead of skipping.
"""

from typing import get_args

import pytest

from nemo_rl.environments.nemo_gym import GYM_SERVER_TYPE_KEYS

config_types = pytest.importorskip("nemo_gym.config_types")

pytestmark = pytest.mark.nemo_gym


def test_server_type_keys_match_gyms_server_instance_schema():
    gym_keys = {cls.SERVER_TYPE for cls in get_args(config_types.ServerInstanceConfig)}
    assert set(GYM_SERVER_TYPE_KEYS) == gym_keys

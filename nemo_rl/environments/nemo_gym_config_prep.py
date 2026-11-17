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

"""What NeMo RL changes in a NeMo Gym config before handing it to NeMo Gym.

Two readers:

* ``nemo_rl.environments.nemo_gym`` uses these when it starts NeMo Gym for training.
* ``examples/nemo_gym/prefetch_venvs.py`` uses them for the NeMo Gym dry run that
  builds the server venvs.

KEEP THIS MODULE STDLIB-ONLY. ``docker/Dockerfile`` copies it in before the Gym venv
prefetch, where the rest of the ``nemo_rl`` package is not in the image yet.
"""

import os
import subprocess

# Keys under env.nemo_gym that NeMo RL reads itself and does not forward to NeMo Gym.
NEMO_RL_ONLY_GYM_KEYS = ("effort_levels", "pad_dynamic_image_shapes")


def get_nemo_gym_uv_cache_dir() -> str | None:
    """Return the uv cache directory inside a container, or None outside one.

    Inside a container (NRL_CONTAINER=1), returns the uv cache location so Gym
    stores its caches in the expected shared path. Returns None outside a
    container, meaning the caller should omit this arg and let Gym create the
    cache locally (the default when you may not be able to write to /opt).
    """
    if not os.environ.get("NRL_CONTAINER"):
        return None
    return subprocess.check_output(["uv", "cache", "dir"]).decode().strip()


def get_nemo_gym_venv_dir() -> str | None:
    """Return the NeMo Gym venv directory from NEMO_GYM_VENV_DIR, or None.

    Returns the value of NEMO_GYM_VENV_DIR if set, otherwise None. When None
    the caller should omit this arg and let Gym create venvs locally (the
    default when a container is not used since you may not be able to write
    to /opt).
    """
    return os.environ.get("NEMO_GYM_VENV_DIR")

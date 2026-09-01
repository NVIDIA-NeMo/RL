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

from nemo_rl.utils.venv_prefetch_manifest import MANIFEST_PATH, render_manifest


def test_manifest_matches_actor_registry():
    """docker/venv_prefetch_manifest.tsv must stay in lockstep with the registry."""
    assert MANIFEST_PATH.read_text() == render_manifest(), (
        "docker/venv_prefetch_manifest.tsv is stale relative to "
        "ACTOR_ENVIRONMENT_REGISTRY; regenerate it with "
        "`uv run python -m nemo_rl.utils.venv_prefetch_manifest`"
    )

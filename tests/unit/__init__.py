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


"""
Auto-loading remote_select plugin here:
- Ensures the plugin is discovered without extra CLI flags or global config.
- Loads early in pytest’s startup so ``pytest_load_initial_conftests`` can
  rewrite args before other plugins (e.g., testmon) prune collection.
- Scopes behavior to unit tests only (does not affect functional tests).
- Avoids a top-level ``conftest.py`` that would apply repo-wide.
"""

pytest_plugins = ["tests.unit._plugins.remote_select"]

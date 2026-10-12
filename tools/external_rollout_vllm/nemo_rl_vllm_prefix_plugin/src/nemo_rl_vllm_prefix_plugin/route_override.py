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

"""Route replacement helpers that do not depend on vLLM imports."""

from typing import Any


CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def remove_existing_chat_completion_routes(app: Any) -> int:
    """Remove POST chat-completion routes before installing the plugin route.

    FastAPI dispatches the first matching route. vLLM installs endpoint plugins
    after its stock OpenAI routes, so merely registering a duplicate path leaves
    the stock handler active and silently drops ``required_prefix_token_ids``.
    """
    routes = app.router.routes
    matching_routes = [
        route
        for route in list(routes)
        if getattr(route, "path", None) == CHAT_COMPLETIONS_PATH
        and "POST" in (getattr(route, "methods", None) or set())
    ]
    if not matching_routes:
        raise RuntimeError(
            "nemo_rl_prefix_api could not find vLLM's stock "
            f"POST {CHAT_COMPLETIONS_PATH} route to replace"
        )
    for route in matching_routes:
        routes.remove(route)
    return len(matching_routes)

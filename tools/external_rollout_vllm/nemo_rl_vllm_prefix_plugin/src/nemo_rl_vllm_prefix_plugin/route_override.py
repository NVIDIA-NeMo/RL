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

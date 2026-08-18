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

"""Config resolution and the NeMo-Gym agent registry for the central agent loop."""

from typing import Any, Optional

# NeMo-Gym agent implementations whose loop the central agent reproduces. The
# implementation name is the inner key of a `responses_api_agents` block, i.e.
# the directory under Gym's `responses_api_agents/`.
SUPPORTED_AGENT_IMPLEMENTATIONS = ("simple_agent", "non_executing_simple_agent")

AGENT_SERVER_TYPE_KEY = "responses_api_agents"
CENTRAL_AGENT_KEY = "central_agent"

DEFAULT_CENTRAL_AGENT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "rollout_tree": {"type": "linear"},
    "tool_calls": {"mode": "serial", "min_returns": 1},
    # Steering calls the agent's own resources server; only the path is configurable.
    "steering_message": {"enabled": False, "url_path": "/steering"},
    "termination": {
        "max_turns": 20,
        "max_malformed_tool_calls": None,
        "stop_on_no_tool_calls": True,
        "stop_on_incomplete_details": True,
    },
}


def _to_plain(value: Any) -> Any:
    """Convert an OmegaConf container to plain Python; pass anything else through."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:  # pragma: no cover - omegaconf is a hard NeMo-RL dep
        return value
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def merge_central_agent_config(
    base: Optional[dict[str, Any]], override: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Deep-merge ``override`` over ``base``. Either may be None."""
    merged = dict(_to_plain(base) or {})
    for key, value in (_to_plain(override) or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_central_agent_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_central_agent_config(cfg: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Fill ``cfg`` in over the defaults. A None config resolves to disabled."""
    return merge_central_agent_config(DEFAULT_CENTRAL_AGENT_CONFIG, cfg)


def pop_agent_registry(
    global_config_dict: Any, central_agent_config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Remove every agent-server block from the merged Gym config and index it.

    The central agent replaces the agent servers, so their blocks are deleted
    before Gym spawns processes: one uvicorn process and one uv venv build per
    agent instance are saved. Server cross-references were already validated when
    the config was parsed, so deleting these blocks cannot break ref resolution
    (nothing references an agent; data rows name one via ``agent_ref``).

    Returns:
        {agent instance name: {implementation, resources_server, model_server,
        central_agent}}, where ``central_agent`` is the base config with the
        agent's own optional ``central_agent`` block merged over it.
    """
    registry: dict[str, dict[str, Any]] = {}
    for top_key in list(global_config_dict.keys()):
        block = global_config_dict[top_key]
        if not hasattr(block, "keys") or AGENT_SERVER_TYPE_KEY not in block:
            continue

        agent_type_block = block[AGENT_SERVER_TYPE_KEY]
        implementation = next(iter(agent_type_block))
        inner = agent_type_block[implementation]
        if implementation not in SUPPORTED_AGENT_IMPLEMENTATIONS:
            raise ValueError(
                f"env.nemo_gym.central_agent is enabled, but agent instance "
                f"'{top_key}' uses the '{implementation}' implementation, whose loop "
                f"the central agent does not reproduce. Supported implementations: "
                f"{list(SUPPORTED_AGENT_IMPLEMENTATIONS)}. Disable central_agent to run "
                f"this agent through its own NeMo-Gym server."
            )

        registry[top_key] = {
            "name": top_key,
            "implementation": implementation,
            "resources_server": _to_plain(inner["resources_server"])["name"],
            "model_server": _to_plain(inner["model_server"])["name"],
            "central_agent": merge_central_agent_config(
                central_agent_config, inner.get(CENTRAL_AGENT_KEY)
            ),
        }
        del global_config_dict[top_key]

    if not registry:
        raise ValueError(
            "env.nemo_gym.central_agent is enabled, but the merged NeMo-Gym config "
            f"defines no '{AGENT_SERVER_TYPE_KEY}' instance to take over."
        )
    return registry

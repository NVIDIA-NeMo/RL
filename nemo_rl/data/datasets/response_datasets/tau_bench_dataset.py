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
"""TauBench dataset for nemo-rl GRPO training.

Each sample contains:
  - messages: [system_message, initial_user_message]
  - extra_env_info: {task_index, episode_id: null, step_count: 0}
  - task_name: "tau_bench"

The system message is built from the environment's tool definitions, domain
wiki, and policy rules, exactly matching what TauBenchEnvironment initialises
at training time.
"""

from typing import Any

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


_TOOL_CALL_INSTRUCTIONS = """\
To call a tool, wrap a JSON object in <tool_call> tags:

    <tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>

Call one tool at a time and wait for the result before proceeding. \
To send a final message to the customer (ending the interaction), \
respond with plain text only — no <tool_call> tags."""


def _format_tool(tool_info: dict[str, Any]) -> str:
    lines = [f"### {tool_info['name']}"]
    if "description" in tool_info:
        lines.append(f"Description: {tool_info['description']}")
    params = tool_info.get("parameters", {})
    if isinstance(params, dict):
        props = params.get("properties", {})
        required = set(params.get("required", []))
        if props:
            lines.append("Parameters:")
            for name, schema in props.items():
                req = " (required)" if name in required else " (optional)"
                ptype = schema.get("type", "any")
                desc = schema.get("description", "")
                lines.append(f"  - {name} ({ptype}{req}): {desc}")
    return "\n".join(lines)


def _build_system_prompt(env: Any, env_name: str) -> str:
    tool_infos = []
    for t in env.tools or []:
        info = t.get_info() if hasattr(t, "get_info") else t
        tool_infos.append(info)

    tools_section = "\n\n".join(_format_tool(t) for t in tool_infos)
    rules_section = "\n".join(f"- {r}" for r in (env.rules or []))
    wiki = (env.wiki or "").strip()

    return "\n\n".join(
        filter(
            None,
            [
                f"You are a customer service agent for a {env_name} company. "
                "Complete the customer's request using the tools below.",
                f"## Tool Usage\n\n{_TOOL_CALL_INSTRUCTIONS}",
                f"## Available Tools\n\n{tools_section}" if tools_section else None,
                f"## Domain Knowledge\n\n{wiki}" if wiki else None,
                f"## Policies\n\n{rules_section}" if rules_section else None,
            ],
        )
    )


class TauBenchDataset(RawDataset):
    """Dataset built from a tau-bench environment.

    Calls tau_bench.envs.get_env() at construction time and iterates every
    task in the requested split to produce (system_prompt, initial_user_message)
    pairs.  No offline data-preparation step is needed.

    Args:
        tau_bench_env_name: tau-bench domain — "retail" or "airline".
        split: Dataset split — "train", "dev", or "test" (default: "train").
        repeat: Number of times to repeat the dataset (default: 1).
    """

    def __init__(
        self,
        tau_bench_env_name: str,
        split: str = "train",
        repeat: int = 1,
        **kwargs: Any,
    ) -> None:
        from tau_bench.envs import get_env

        self.task_name = "tau_bench"

        # user_strategy and user_model are placeholders; no user simulation
        # occurs during data loading — only during training rollouts.
        env = get_env(
            env_name=tau_bench_env_name,
            user_strategy="llm",
            user_model="placeholder",
            task_split=split,
        )

        system_prompt = _build_system_prompt(env, tau_bench_env_name)

        records = []
        for task_index in range(len(env.tasks)):
            reset_response = env.reset(task_index=task_index)
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(reset_response.observation)},
                    ],
                    "extra_env_info": {
                        "task_index": task_index,
                        "episode_id": None,
                        "step_count": 0,
                    },
                    "task_name": "tau_bench",
                }
            )

        self.dataset = Dataset.from_list(records)

        if repeat > 1:
            self.dataset = self.dataset.repeat(repeat)

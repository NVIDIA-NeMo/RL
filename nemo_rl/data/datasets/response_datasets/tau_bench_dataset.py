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

import ray
from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

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
    for t in env.tools_info or []:
        # tau_bench wraps each tool spec as {"type": "function", "function": {...}}
        info = t["function"] if "function" in t else t
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


def _build_records(tau_bench_env_name: str, split: str) -> list[dict[str, Any]]:
    """Build the list of dataset records from a tau-bench environment.

    This function is invoked via Ray remote with PY_EXECUTABLES.TAU_BENCH so
    that tau_bench is only required in the TAU_BENCH Ray environment, not in
    the driver process.
    """
    from tau_bench.envs import get_env

    # Use the "human" strategy so no LLM provider is required at data-load
    # time. We read task.instruction directly — env.reset() is intentionally
    # NOT called here because HumanUserSimulationEnv.reset() blocks on
    # input(), which would hang the process for every task.
    try:
        env = get_env(
            env_name=tau_bench_env_name,
            user_strategy="human",
            user_model="",
            task_split=split,
        )
    except IndexError:
        # The requested split has no tasks (e.g. the data file does not exist).
        # Return an empty list so the caller can handle it gracefully.
        print(
            f"[WARNING] tau-bench split '{split}' has no tasks for env "
            f"'{tau_bench_env_name}'. Returning empty dataset."
        )
        return []

    system_prompt = _build_system_prompt(env, tau_bench_env_name)

    # The actual customer opening message comes from env.reset() (via the LLM
    # user simulator) at training time.  TauBenchWorker.execute() returns it as
    # the first observation (a "pre-step") before any agent action is processed.
    # The placeholder below just gives the model a neutral starting point for
    # the wasted generation on that pre-step turn; the real conversation begins
    # once the agent sees the customer's actual first message.
    # Keep the placeholder as uninformative as possible so it doesn't prime the
    # model with false expectations before it sees the real customer message.
    placeholder_user_msg = "Hi"
    return [
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": placeholder_user_msg},
            ],
            "extra_env_info": {
                "task_index": task_index,
                "episode_id": None,
                "step_count": 0,
            },
            "task_name": "tau_bench",
        }
        for task_index, task in enumerate(env.tasks)
    ]


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
        self.task_name = "tau_bench"

        # Run record construction in the TAU_BENCH Ray environment so that
        # tau_bench is not required in the driver process.
        records = ray.get(
            ray.remote(_build_records)
            .options(runtime_env={"py_executable": PY_EXECUTABLES.TAU_BENCH})
            .remote(tau_bench_env_name, split)
        )

        self.dataset = Dataset.from_list(records)

        if repeat > 1:
            self.dataset = self.dataset.repeat(repeat)

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

import json
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

# Map Python type names used in xLAM to JSON Schema types.
_XLAM_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "List": "array",
    "Dict": "object",
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}


def _convert_xlam_tools_to_openai(xlam_tools: list[dict]) -> list[dict]:
    """Convert xLAM tool definitions to OpenAI function-calling format.

    xLAM uses flat ``{name, description, parameters: {param_name: {type, description, default}}}``
    while ``apply_chat_template(tools=...)`` expects OpenAI format:
    ``{type: "function", function: {name, description, parameters: <JSON Schema>}}``.
    """
    openai_tools = []
    for tool in xlam_tools:
        properties = {}
        required = []
        for param_name, param_spec in (tool.get("parameters") or {}).items():
            prop: dict[str, Any] = {
                "description": param_spec.get("description", ""),
            }
            raw_type = param_spec.get("type", "string")
            prop["type"] = _XLAM_TYPE_MAP.get(raw_type, "string")
            if "default" not in param_spec:
                required.append(param_name)
            properties[param_name] = prop

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        **({"required": required} if required else {}),
                    },
                },
            }
        )
    return openai_tools


class XLAMFunctionCallingDataset(RawDataset):
    """Wrapper around the Salesforce/xlam-function-calling-60k dataset.

    Each sample contains a user query, tool definitions, and gold function calls.
    The dataset converts xLAM tool schemas to OpenAI format for use with
    ``tokenizer.apply_chat_template(tools=...)``.
    """

    def __init__(
        self,
        split: str = "train",
        split_validation_size: float = 0.05,
        seed: int = 42,
        **kwargs,
    ):
        self.task_name = "xlam_function_calling"
        self.dataset = load_dataset("Salesforce/xlam-function-calling-60k", split=split)
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        xlam_tools = json.loads(data["tools"])
        openai_tools = _convert_xlam_tools_to_openai(xlam_tools)
        return {
            "query": data["query"],
            "tools": json.dumps(openai_tools),
            "gold_answer": data["answers"],
            "task_name": self.task_name,
        }

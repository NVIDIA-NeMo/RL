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

"""Generate native-chat data for the cross-tokenizer functional test."""

import argparse
import json
from pathlib import Path


def write_dataset(path: Path) -> None:
    """Write eight conversations covering reasoning, tools, and turn selection."""
    with path.open("w", encoding="utf-8") as output:
        for value in range(8):
            first_answer = {
                "role": "assistant",
                "reasoning_content": f"Add one to {value} to obtain {value + 1}.",
                "content": str(value + 1),
            }
            if value == 1:
                reasoning = first_answer.pop("reasoning_content")
                first_answer["content"] = (
                    f"<think>\n{reasoning}\n</think>\n\n{value + 1}"
                )
            final_answer = {
                "role": "assistant",
                "reasoning_content": (
                    f"The previous result was {value + 1}. "
                    f"Adding two gives {value + 3}."
                ),
                "content": str(value + 3),
            }
            row = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"What is {value} plus one?"},
                    first_answer,
                    {"role": "user", "content": "Now add two to that result."},
                    final_answer,
                ],
                "message_loss_mask": [0, 0, int(value % 2 == 0), 0, 1],
            }
            if value == 2:
                final_answer["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": {"a": value + 1, "b": 2},
                        },
                    }
                ]
                row["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                    }
                ]
            output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_dataset(args.output)

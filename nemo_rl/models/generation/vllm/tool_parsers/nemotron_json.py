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

import json
import re
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = init_logger(__name__)


@ToolParserManager.register_module("nemotron_json")
class NemotronJSONToolParser(ToolParser):
    """Nemotron Nano v2 non-streaming tool parser for vLLM."""

    def __init__(self, tokenizer: Any, tools: Any | None = None) -> None:
        try:
            super().__init__(tokenizer, tools)
        except TypeError:
            super().__init__(tokenizer)
        self.tool_call_start_token = "<TOOLCALL>"
        self.tool_call_end_token = "</TOOLCALL>"
        self.tool_call_regex = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_call_matches = self.tool_call_regex.findall(model_output)
        if not tool_call_matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        try:
            str_tool_calls = tool_call_matches[0].strip()
            if not str_tool_calls.startswith("["):
                str_tool_calls = "[" + str_tool_calls
            if not str_tool_calls.endswith("]"):
                str_tool_calls += "]"

            tool_calls = []
            for tool_call in json.loads(str_tool_calls):
                try:
                    arguments = tool_call["arguments"]
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=tool_call["name"],
                                arguments=arguments,
                            ),
                        )
                    )
                except (KeyError, TypeError):
                    continue

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            content = model_output[: model_output.rfind(self.tool_call_start_token)]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content or None,
            )
        except (json.JSONDecodeError, TypeError):
            logger.debug("Failed to parse Nemotron tool call.", exc_info=True)
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        raise NotImplementedError("Tool calling is not supported in streaming mode.")

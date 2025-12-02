"""
Nemotron JSON tool parser for vLLM, adapted from
`nvidia/NVIDIA-Nemotron-Nano-9B-v2` on Hugging Face:
`https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/main/nemotron_toolcall_parser_no_streaming.py`.

The original file is licensed under the NVIDIA Open Model License /
Apache-2.0-equivalent terms; this variant is trimmed to the pieces needed
for NeMo RL non-streaming tool calling.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Union

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("nemotron_json")
class NemotronJSONToolParser(ToolParser):
    """
    Simple tool parser for Nemotron-Nano v2 models using <TOOLCALL>...</TOOLCALL>
    JSON blocks in the assistant output.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<TOOLCALL>"
        self.tool_call_end_token: str = "</TOOLCALL>"

        self.tool_call_regex = re.compile(
            r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Non-streaming extraction: look for a single <TOOLCALL>...</TOOLCALL>
        block containing a JSON list of tool calls.
        """
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        try:
            # Grab the JSON substring inside the TOOLCALL tags.
            str_tool_calls = self.tool_call_regex.findall(model_output)[0].strip()
            if not str_tool_calls.startswith("["):
                str_tool_calls = "[" + str_tool_calls
            if not str_tool_calls.endswith("]"):
                str_tool_calls = str_tool_calls + "]"

            json_tool_calls = json.loads(str_tool_calls)
            tool_calls: list[ToolCall] = []
            for tool_call in json_tool_calls:
                try:
                    args = tool_call.get("arguments")
                    if isinstance(args, dict):
                        args_str = json.dumps(args, ensure_ascii=False)
                    else:
                        args_str = args

                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=tool_call["name"],
                                arguments=args_str,
                            ),
                        )
                    )
                except Exception:
                    # Skip malformed tool call entries rather than failing hard.
                    continue

            content = model_output[: model_output.rfind(self.tool_call_start_token)]

            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception(
                "Error in extracting tool call from response. Response: %s",
                model_output,
            )
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
    ) -> Union[DeltaMessage, None]:
        """
        Streaming tool calling is not supported in this simplified parser.
        """
        raise NotImplementedError(
            "Tool calling is not supported in streaming mode for NemotronJSONToolParser."
        )



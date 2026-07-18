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

from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


@ReasoningParserManager.register_module("nano_v3")
class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
    """Reasoning parser for NVIDIA Nemotron Nano v3 models."""

    def extract_reasoning(self, model_output, request):
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)
        thinking_enabled = not (
            chat_template_kwargs
            and chat_template_kwargs.get("enable_thinking") is False
        )

        # Nano 3 can close ``</think>`` *after* emitting a complete Qwen-style
        # XML tool call. vLLM parses tools exclusively from final content, so
        # leaving this block in reasoning silently turns the tool call into a
        # plain assistant message. Hoist only a complete tool call that is
        # demonstrably inside the thinking span; malformed output keeps the
        # base parser's fail-closed behavior.
        if thinking_enabled and isinstance(model_output, str):
            tool_start = model_output.find("<tool_call>")
            tool_end = model_output.find("</tool_call>", tool_start)
            think_end = model_output.find("</think>")
            tool_is_inside_thinking = tool_start >= 0 and (
                think_end < 0 or tool_start < think_end
            )
            tool_is_complete = tool_end >= 0 and (think_end < 0 or tool_end < think_end)
            if tool_is_inside_thinking and tool_is_complete:
                reasoning_prefix = model_output[:tool_start]
                reasoning_content, _ = super().extract_reasoning(
                    f"{reasoning_prefix}</think>", request
                )
                if think_end >= 0:
                    final_content = (
                        model_output[tool_start:think_end]
                        + model_output[think_end + len("</think>") :]
                    )
                else:
                    final_content = model_output[tool_start:]

        if not thinking_enabled and final_content is None:
            return None, reasoning_content

        return reasoning_content, final_content

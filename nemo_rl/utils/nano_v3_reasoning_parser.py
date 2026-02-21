import logging

from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

logger = logging.getLogger(__name__)

# This will print when the module is imported/loaded
print("=" * 60)
print("NanoV3ReasoningParser plugin loaded and registered!")
print("=" * 60)


@ReasoningParserManager.register_module("nano_v3")
class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        print(f"NanoV3ReasoningParser initialized with tokenizer: {type(tokenizer)}")

    def extract_reasoning(self, model_output, request):
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )
        if (
            hasattr(request, "chat_template_kwargs")
            and request.chat_template_kwargs
            and request.chat_template_kwargs.get("enable_thinking") is False
            and final_content is None
        ):
            print("NanoV3ReasoningParser: Swapping reasoning and final content (enable_thinking=False)")
            reasoning_content, final_content = final_content, reasoning_content

        return reasoning_content, final_content

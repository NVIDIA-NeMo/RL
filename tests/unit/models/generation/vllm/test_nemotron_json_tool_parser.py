import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


def _load_nemotron_json_tool_parser(monkeypatch, tool_parser_cls):
    registered_tool_parsers = {}

    for module_name in (
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion",
        "vllm.logger",
        "vllm.tool_parsers",
    ):
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    class FunctionCall:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class ToolCall:
        def __init__(self, type, function):
            self.type = type
            self.function = function

    class ExtractedToolCallInformation:
        def __init__(self, tools_called, tool_calls, content):
            self.tools_called = tools_called
            self.tool_calls = tool_calls
            self.content = content

    class ToolParserManager:
        @staticmethod
        def register_module(name):
            def decorator(parser_cls):
                registered_tool_parsers[name] = parser_cls
                return parser_cls

            return decorator

    protocol_module = types.ModuleType(
        "vllm.entrypoints.openai.chat_completion.protocol"
    )
    protocol_module.ChatCompletionRequest = type("ChatCompletionRequest", (), {})
    protocol_module.DeltaMessage = type("DeltaMessage", (), {})
    protocol_module.FunctionCall = FunctionCall
    protocol_module.ToolCall = ToolCall
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.chat_completion.protocol",
        protocol_module,
    )

    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: MagicMock()
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    abstract_tool_parser_module = types.ModuleType(
        "vllm.tool_parsers.abstract_tool_parser"
    )
    abstract_tool_parser_module.ExtractedToolCallInformation = (
        ExtractedToolCallInformation
    )
    abstract_tool_parser_module.ToolParser = tool_parser_cls
    abstract_tool_parser_module.ToolParserManager = ToolParserManager
    monkeypatch.setitem(
        sys.modules,
        "vllm.tool_parsers.abstract_tool_parser",
        abstract_tool_parser_module,
    )

    repo_root = Path(__file__).resolve().parents[5]
    parser_path = (
        repo_root / "nemo_rl/models/generation/vllm/tool_parsers/nemotron_json.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_nemotron_json_tool_parser",
        parser_path,
    )
    assert spec is not None
    assert spec.loader is not None
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)

    return parser_module, registered_tool_parsers


def test_nemotron_json_tool_parser_extracts_tool_calls(monkeypatch):
    class ToolParser:
        def __init__(self, tokenizer, tools):
            self.tokenizer = tokenizer
            self.tools = tools

    parser_module, registered_tool_parsers = _load_nemotron_json_tool_parser(
        monkeypatch,
        ToolParser,
    )

    assert (
        registered_tool_parsers["nemotron_json"]
        is parser_module.NemotronJSONToolParser
    )
    parser = parser_module.NemotronJSONToolParser(tokenizer=object(), tools=[])

    result = parser.extract_tool_calls(
        (
            "assistant text"
            '<TOOLCALL>{"name": "email_search_emails", '
            '"arguments": {"query": "status"}}</TOOLCALL>'
        ),
        request=object(),
    )

    assert result.tools_called is True
    assert result.content == "assistant text"
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "email_search_emails"
    assert json.loads(tool_call.function.arguments) == {"query": "status"}


def test_nemotron_json_tool_parser_ignores_malformed_tool_calls(monkeypatch):
    class ToolParser:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    parser_module, registered_tool_parsers = _load_nemotron_json_tool_parser(
        monkeypatch,
        ToolParser,
    )

    assert "nemotron_json" in registered_tool_parsers
    parser = parser_module.NemotronJSONToolParser(tokenizer=object(), tools=[])
    malformed_output = 'prefix<TOOLCALL>{"name": "broken"'

    result = parser.extract_tool_calls(malformed_output, request=object())

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == malformed_output

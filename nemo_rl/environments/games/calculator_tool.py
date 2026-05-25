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

import ast
import json
import operator
import re
from typing import Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

# Tool definition for tokenizer.apply_chat_template(tools=...)
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression. "
            "Supports +, -, *, /, parentheses, and ** for exponents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical expression to evaluate, "
                        "e.g. '(100 * 0.75) + 15.99'"
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def safe_eval_expr(expr_str: str) -> Optional[float]:
    """Safely evaluate a mathematical expression using AST parsing.

    Supports +, -, *, /, **, parentheses, and numeric literals.
    Returns None for invalid or dangerous expressions.
    """
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
    except SyntaxError:
        return None

    def _eval_node(node: ast.expr) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op_func = _OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError
            if isinstance(node.op, ast.Pow) and (abs(left) > 1e6 or abs(right) > 100):
                raise ValueError("Exponent too large")
            return op_func(left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        raise ValueError(f"Unsupported node type: {type(node).__name__}")

    try:
        return _eval_node(tree)
    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
        return None


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_tool_call(text: str) -> Optional[str]:
    """Extract calculator expression from a <tool_call> in the response."""
    text = _strip_think_blocks(text)
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not match:
        return None
    try:
        call = json.loads(match.group(1).strip())
        if call.get("name") == "calculate" and "arguments" in call:
            return call["arguments"].get("expression")
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return None


def extract_final_answer(text: str) -> Optional[float]:
    """Extract numeric answer from <answer>NUMBER</answer> tags."""
    text = _strip_think_blocks(text)
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not match:
        return None
    try:
        return float(match.group(1).strip().replace(",", "").replace("$", ""))
    except ValueError:
        return None


class CalculatorMetadata(TypedDict):
    expected_answer: float
    problem: str
    tool_calls_remaining: int
    max_tool_calls: int
    tolerance: float
    relative_tolerance: float


@ray.remote  # pragma: no cover
class CalculatorToolEnv(EnvironmentInterface[CalculatorMetadata]):
    """Calculator tool-calling environment.

    Multi-turn: the model calls a calculator tool to solve word problems,
    then provides a final answer. Rewards are based on answer accuracy.
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[CalculatorMetadata],
    ) -> EnvironmentReturn[CalculatorMetadata]:
        observations = []
        rewards_list = []
        terminateds_list = []
        next_stop_strings_list = []
        next_metadata_list = []
        answers_list = []

        for message_log, meta in zip(message_log_batch, metadata):
            if meta is None:
                observations.append({"role": "tool", "content": ""})
                rewards_list.append(0.0)
                terminateds_list.append(True)
                next_stop_strings_list.append(None)
                next_metadata_list.append(None)
                answers_list.append(None)
                continue

            response = message_log[-1]["content"] if message_log else ""

            # Check for final answer
            final_answer = extract_final_answer(response)
            if final_answer is not None:
                reward = _compute_reward(
                    final_answer,
                    meta["expected_answer"],
                    meta["tolerance"],
                    meta["relative_tolerance"],
                )
                observations.append({"role": "tool", "content": ""})
                rewards_list.append(reward)
                terminateds_list.append(True)
                next_stop_strings_list.append(None)
                next_metadata_list.append(None)
                answers_list.append(str(final_answer))
                continue

            # Check for tool call
            expression = parse_tool_call(response)
            if expression is not None and meta["tool_calls_remaining"] > 0:
                result = safe_eval_expr(expression)
                if result is not None:
                    result_str = f"{result:.6g}"
                else:
                    result_str = "Error: invalid expression"

                obs_content = f"\n<tool_response>\n{result_str}\n</tool_response>\n"
                new_meta: CalculatorMetadata = {
                    **meta,
                    "tool_calls_remaining": meta["tool_calls_remaining"] - 1,
                }

                observations.append({"role": "tool", "content": obs_content})
                rewards_list.append(0.0)
                terminateds_list.append(False)
                next_stop_strings_list.append(["</tool_call>", "</answer>"])
                next_metadata_list.append(new_meta)
                answers_list.append(None)
                continue

            # No valid tool call or final answer
            if meta["tool_calls_remaining"] <= 0:
                # Out of tool calls, terminate with 0 reward
                observations.append({"role": "tool", "content": ""})
                rewards_list.append(0.0)
                terminateds_list.append(True)
                next_stop_strings_list.append(None)
                next_metadata_list.append(None)
                answers_list.append(None)
            else:
                # Invalid format but still has tool calls, let model try again
                obs_content = (
                    "\n<tool_response>\n"
                    "Error: no valid tool call or answer detected. "
                    "Use <tool_call> or <answer> tags.\n"
                    "</tool_response>\n"
                )
                new_meta = {
                    **meta,
                    "tool_calls_remaining": meta["tool_calls_remaining"] - 1,
                }
                observations.append({"role": "tool", "content": obs_content})
                rewards_list.append(0.0)
                terminateds_list.append(False)
                next_stop_strings_list.append(["</tool_call>", "</answer>"])
                next_metadata_list.append(new_meta)
                answers_list.append(None)

        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata_list,
            next_stop_strings=next_stop_strings_list,
            rewards=torch.tensor(rewards_list, dtype=torch.float32),
            terminateds=torch.tensor(terminateds_list, dtype=torch.bool),
            answers=answers_list,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        if len(final_rewards) == 0:
            return batch, {"accuracy": 0.0, "partial_credit": 0.0}
        accuracy = (final_rewards == 1.0).float().mean().item()
        partial = (final_rewards == 0.5).float().mean().item()
        return batch, {"accuracy": accuracy, "partial_credit": partial}


def _compute_reward(
    predicted: float,
    expected: float,
    tolerance: float,
    relative_tolerance: float,
) -> float:
    """Compute reward based on answer accuracy."""
    diff = abs(predicted - expected)
    if diff <= tolerance:
        return 1.0
    if expected != 0 and diff / abs(expected) <= relative_tolerance:
        return 0.5
    return 0.0

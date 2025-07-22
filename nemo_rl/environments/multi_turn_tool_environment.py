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

import copy
import json
import logging
import re
import importlib
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Set
import torch
import ray

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class MultiTurnToolMetadata(TypedDict):
    """Metadata for tracking multi-turn tool state."""
    id: str
    current_turn: int
    max_turns: int
    ground_truth: List[List[str]]  # GT tool calls per turn
    user_question_bank: List[List[Dict[str, str]]]  # Next user questions
    model_tool_instances: Dict[str, Any]  # Model's tool instances
    gt_tool_instances: Dict[str, Any]  # Ground truth tool instances
    model_calls_per_turn: List[List[str]]  # Model's calls per turn

class MultiTurnEnvConfig(TypedDict):
    """Configuration for MultiTurnToolEnvironment."""
    num_workers: int
    max_turns: int


class ToolManager:
    """Manages tool initialization and execution."""
    
    TOOL_CLASS_MAPPING = {
        "GorillaFileSystem": "nemo_rl.environments.tools.gorilla_file_system",
        "TicketAPI": "nemo_rl.environments.tools.ticket_api",
        "TwitterAPI": "nemo_rl.environments.tools.twitter_api",
    }
    
    def initialize_tools(self, tool_names: List[str], initial_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize tool instances."""
        tools = {}
        for tool_name in tool_names:
            if tool_name in self.TOOL_CLASS_MAPPING:
                module_path = self.TOOL_CLASS_MAPPING[tool_name]
                try:
                    module = importlib.import_module(module_path)
                    tool_class = getattr(module, tool_name)
                    
                    # Create instance with empty constructor
                    tool_instance = tool_class()
                    
                    # Load scenario/configuration
                    class_initial_config = initial_config.get(tool_name, {})
                    tool_instance._load_scenario(copy.deepcopy(class_initial_config))
                    
                    tools[tool_name] = tool_instance
                except Exception as e:
                    print(f"Failed to initialize {tool_name}: {e}")
        return tools

    def parse_tool_calls(self, assistant_response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from assistant response within <tool> tags."""
        
        # Extract from <tool> tags
        tool_tag_pattern = r'<tool>(.*?)</tool>'
        tool_matches = re.findall(tool_tag_pattern, assistant_response, re.DOTALL)
        if tool_matches:
            # Parse JSON array inside tool tags
            try:
                tool_content = tool_matches[0].strip()
                tool_calls_raw = json.loads(tool_content)

                tool_calls: List[Dict[str, Any]] = []
                if isinstance(tool_calls_raw, list):
                    for item in tool_calls_raw:
                        if isinstance(item, dict):
                            tool_calls.append(item)
                        elif isinstance(item, str):
                            try:
                                parsed_item = json.loads(item)
                                if isinstance(parsed_item, dict):
                                    tool_calls.append(parsed_item)
                            except json.JSONDecodeError:
                                # Skip malformed entries; they'll be handled later.
                                continue
                return tool_calls
            except json.JSONDecodeError:
                return []
        return []

    def execute_tool_call(self, tool_call: Dict[str, Any], tools: Dict[str, Any]) -> Tuple[str, bool]:
        """Execute a single tool call."""
        func_name = tool_call.get('name', '')
        args = tool_call.get('args', {})
        
        # Build method-to-tool mapping for explicit tool selection
        method_to_tool = {}
        for tool_name, tool_instance in tools.items():
            # Get all public methods of this tool
            for method_name in dir(tool_instance):
                if not method_name.startswith('_') and callable(getattr(tool_instance, method_name)):
                    if method_name in method_to_tool:
                        # Method collision - prefer first tool or could add disambiguation
                        continue
                    method_to_tool[method_name] = (tool_name, tool_instance)
        
        # Execute the method call
        if func_name in method_to_tool:
            tool_name, tool_instance = method_to_tool[func_name]
            try:
                method = getattr(tool_instance, func_name)
                result = method(**args)
                result_str = str(result) if result is not None else "Success"
                return f"[{tool_name}.{func_name}] {result_str}", True
            except Exception as e:
                return f"[{tool_name}.{func_name}] Error: {str(e)}", False
        return f"Tool function '{func_name}' not found in any tool", False





class RewardCalculator:
    """Calculates rewards for turns."""
    
    def calculate_reward(self, metadata: MultiTurnToolMetadata, is_final_turn: bool) -> float:
        """Calculate reward for current turn."""
        
        # if not is_final_turn:
        #     return 0.0  # No reward for intermediate turns
        
        # Final turn - calculate reward
        # Compare states only for tools the model actually invoked this turn
        cur_turn = metadata.get("current_turn", 0)
        calls_this_turn: List[str] = []
        if cur_turn < len(metadata.get("model_calls_per_turn", [])):
            calls_this_turn = metadata["model_calls_per_turn"][cur_turn]

        used_tools: Set[str] = set()
        for call in calls_this_turn:
            func = call.split("(")[0].strip()
            for t_name, t_inst in metadata["model_tool_instances"].items():
                if hasattr(t_inst, func):
                    used_tools.add(t_name)

        state_score = 0.0
        # compute state score only on final turn
        if is_final_turn:
            state_score = self._compare_tool_states(
                metadata["model_tool_instances"],
                metadata["gt_tool_instances"],
                used_tools,
            )
        
        call_score = self._compare_tool_calls(
            metadata["model_calls_per_turn"],
            metadata["ground_truth"],
            cur_turn,
        )
        # breakpoint()
        return state_score, call_score
    
    def _compare_tool_states(
        self,
        model_tools: Dict[str, Any],
        gt_tools: Dict[str, Any],
        tool_subset: Set[str],
    ) -> float:
        """Compare states for the specified subset of tools."""

        if not tool_subset:
            return 1.0  # nothing to compare yet

        total_tools = len(tool_subset)
        matching_tools = 0
        
        for tool_name in tool_subset:
            model_instance = model_tools.get(tool_name)
            gt_instance = gt_tools.get(tool_name)
            if model_instance is None or gt_instance is None:
                continue

            if type(model_instance) != type(gt_instance):
                continue
                
            # Compare all non-private attributes
            states_match = True
            for attr_name in vars(gt_instance):
                if attr_name.startswith("_"):
                    continue
                    
                model_attr = getattr(model_instance, attr_name)
                gt_attr = getattr(gt_instance, attr_name)
                
                if model_attr != gt_attr:
                    states_match = False
                    break
            
            if states_match:
                matching_tools += 1
        
        return matching_tools / total_tools if total_tools > 0 else 1.0
    
    def _compare_tool_calls(
        self,
        model_calls: List[List[str]],
        gt_calls: List[List[str]],
        turn_index: int,
    ) -> float:
        """Return 1.0 if the model's calls for ``turn_index`` exactly match ground truth, else 0.0."""

        # Guard against out-of-range indices or missing data.
        if (
            turn_index >= len(gt_calls)
            or turn_index >= len(model_calls)
        ):
            return 0.0

        model_set = set(model_calls[turn_index])
        gt_set = set(gt_calls[turn_index])

        # Full reward if model exactly matches ground-truth.
        if model_set == gt_set:
            return 1.0

        if not gt_set:
            # No expected calls but model differs → zero.
            return 0.0

        # Partial reward: proportion of correctly predicted calls.
        correct_calls = len(model_set.intersection(gt_set))
        # Penalise extra incorrect calls by dividing by total unique calls made/expected.
        total_unique = len(model_set.union(gt_set))

        return correct_calls / total_unique

@ray.remote
class MultiTurnToolEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM
    """Multi-turn tool environment"""

    def __init__(self, cfg: Optional[MultiTurnEnvConfig] = None):
        self.cfg = cfg or {"num_workers": 1, "max_turns": 10}
        self.tool_manager = ToolManager()
        self.reward_calculator = RewardCalculator()

    def _initialize_episode_metadata(self, sample_metadata: Dict[str, Any]) -> MultiTurnToolMetadata:
        """Initialize metadata for new episode."""
        involved_classes = sample_metadata.get("involved_classes", [])
        initial_config = sample_metadata.get("initial_config", {})
        
        # Initialize tool instances for both model and ground truth
        model_tools = self.tool_manager.initialize_tools(involved_classes, initial_config)
        gt_tools = self.tool_manager.initialize_tools(involved_classes, initial_config)
        
        return {
            "id": sample_metadata.get("id", ""),
            "current_turn": 0,
            "max_turns": len(sample_metadata.get("ground_truth", [])),
            "ground_truth": sample_metadata.get("ground_truth", []),
            "user_question_bank": sample_metadata.get("user_question_bank", []),
            "model_tool_instances": model_tools,
            "gt_tool_instances": gt_tools,
            "model_calls_per_turn": [],
            "turn_metadata": {}
        }

    def _should_continue(self, metadata: MultiTurnToolMetadata) -> bool:
        """Check if conversation should continue to next turn."""
        return (
            metadata["current_turn"] <= metadata["max_turns"] - 1
        )
    
    def _get_next_observation(self, tool_results: str, metadata: MultiTurnToolMetadata) -> Dict[str, str]:
        """Generate observation for next turn or termination."""
        return {"role": "environment", "content": f"<tool_result> {tool_results} </tool_result>"}

    def _process_turn(self, message_log: LLMMessageLogType, metadata: MultiTurnToolMetadata) -> Tuple[str, List[str], bool]:
        """Process current turn and return tool results and calls made."""
        
        # Get latest assistant response
        assistant_response = ""
        for msg in reversed(message_log):
            if msg["role"] == "assistant":
                assistant_response = msg["content"]
                break
        model_calls_made = []
        tool_results = []
        # turn_success is True only if ALL expected tool calls execute successfully
        # Initialize as True and set to False on any failure or missing call.
        turn_success = True
        # Check if tool tags exist
        if '<tool>' not in assistant_response:
            tool_results.append("Function call not found in current assistant response.")
            model_calls_made.append("No function call made'")
            turn_success = False
        else:
            # Parse tool calls
            model_tool_calls = self.tool_manager.parse_tool_calls(assistant_response)
            if not model_tool_calls:
                tool_results.append("Error: Invalid tool command. Parsing tool calls failed. Ensure correct formatting. "
                                "Tool command must be one list of JSON objects.")
                model_calls_made.append('No function call made')
                turn_success = False
            else:

                for tool_call in model_tool_calls:
                    result, success = self.tool_manager.execute_tool_call(
                        tool_call, metadata["model_tool_instances"]
                    )
                    tool_results.append(result)
                    func_name = tool_call.get('name', '')
                    args = tool_call.get('args', {})
                    
                    if isinstance(args, dict):
                        arg_repr = ', '.join([f"{k}={repr(v)}" for k, v in args.items()])
                    else:
                        # Unexpected format – record the raw representation
                        arg_repr = repr(args)
                        # Mark turn as failed because format is not as expected
                        turn_success = False

                    call_str = f"{func_name}({arg_repr})"
                    model_calls_made.append(call_str)
                    # Fail turn if execution failed OR result string contains 'error'.
                    if not success or ("error" in result.lower()):
                        turn_success = False
                        # Stop on first error
                        break
        # TODO: ykarnati - both the tool calls might be successful
        #  but these might be wrong calls.
        # should we still make the turn success ?
        
        # Execute ground truth calls for this turn
        current_turn = metadata["current_turn"]
        if current_turn < len(metadata["ground_truth"]):
            gt_calls = metadata["ground_truth"][current_turn]
            for call_str in gt_calls:
                result, _ = self._execute_gt_call(call_str, metadata["gt_tool_instances"])

        # breakpoint()
        return "\n".join(tool_results), model_calls_made, turn_success
    
    def _execute_gt_call(self, call_str: str, tools: Dict[str, Any]) -> Tuple[str, bool]:
        """Execute ground truth call string."""
        # Parse call string like "cd(folder='document')"
        if '(' in call_str and call_str.endswith(')'):
            func_name = call_str.split('(')[0].strip()
            args_str = call_str[len(func_name)+1:-1]
            
            args = {}
            if args_str:
                for arg in args_str.split(','):
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        args[key.strip()] = value.strip().strip('"\'')
            
            tool_call = {'name': func_name, 'args': args}
            return self.tool_manager.execute_tool_call(tool_call, tools)
        
        return f"Invalid call format: {call_str}", False

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata: List[Dict[str, Any]],
    ) -> EnvironmentReturn:
        """Process single turn for each sample in batch."""
        
        # Initialize or update metadata
        processed_metadata = []
        for meta in metadata:
            if isinstance(meta, dict) and "current_turn" not in meta:
                # First turn - initialize
                processed_metadata.append(self._initialize_episode_metadata(meta))
            else:
                # Continuing turn
                processed_metadata.append(meta)
        
        # Process each sample
        observations = []
        rewards = []
        terminateds = []
        next_stop_strings = []
        next_metadata = []
        
        for i, (message_log, sample_metadata) in enumerate(zip(message_log_batch, processed_metadata)):
            # Process current turn
            tool_results, model_calls, turn_success = self._process_turn(
                message_log, sample_metadata
            )
            
            sample_metadata["model_calls_per_turn"].append(model_calls)
            sample_metadata["turn_success"] = turn_success
            
            # Check if should continue
            should_continue = self._should_continue(sample_metadata)
            
            # Generate observation
            observation = self._get_next_observation(tool_results, sample_metadata)
            
            # Calculate reward
            is_final_turn = not should_continue
            state_score, call_score = self.reward_calculator.calculate_reward(sample_metadata, is_final_turn)
            reward = 0.5 * state_score + 1 * call_score
            sample_metadata["turn_metadata"][sample_metadata["current_turn"]] = {}
            sample_metadata["turn_metadata"][sample_metadata["current_turn"]].update({
                "state_score": state_score,
                "call_score": call_score,
                "turn_success": turn_success,
                "tool_results": tool_results,
            })
            # Update for next turn
            if should_continue:
                sample_metadata["current_turn"] += 1
                
                terminateds.append(False)
                next_stop_strings.append(None)
            else:
                # next_metadata.append(None)
                terminateds.append(True)
                next_stop_strings.append(None)
            next_metadata.append(sample_metadata)
            observations.append(observation)
            rewards.append(reward)
        # breakpoint()
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=next_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Compute environment metrics."""
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        
        metrics = {
            "accuracy": batch["rewards"].mean().item(),
            "success_rate": (batch["rewards"] >= 1.0).float().mean().item(),
            "fraction_properly_ended": batch["is_end"].float().mean().item(),
        }
        
        return batch, metrics
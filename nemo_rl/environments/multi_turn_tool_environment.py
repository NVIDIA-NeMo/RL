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
from typing import Any, Dict, List, Optional, Tuple, TypedDict
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
                tool_calls = json.loads(tool_content)
                return tool_calls if isinstance(tool_calls, list) else []
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
        
        if not is_final_turn:
            return 0.0  # No reward for intermediate turns
        
        # Final turn - calculate reward
        state_score = self._compare_tool_states(
            metadata["model_tool_instances"], 
            metadata["gt_tool_instances"]
        )
        
        call_score = self._compare_tool_calls(
            metadata["model_calls_per_turn"], 
            metadata["ground_truth"]
        )

        return 0.5 * state_score + 0.5 * call_score
    
    def _compare_tool_states(self, model_tools: Dict[str, Any], gt_tools: Dict[str, Any]) -> float:
        """Compare final tool states by checking internal attributes."""
        if set(model_tools.keys()) != set(gt_tools.keys()):
            return 0.0
        
        total_tools = len(model_tools)
        matching_tools = 0
        
        for tool_name in model_tools:
            model_instance = model_tools[tool_name]
            gt_instance = gt_tools[tool_name]
            
            # Check if instances are of the same type
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
    
    def _compare_tool_calls(self, model_calls: List[List[str]], gt_calls: List[List[str]]) -> float:
        """Compare tool calls across all turns."""
        if not gt_calls:
            return 1.0
        
        scores = []
        for i in range(len(gt_calls)):
            if i < len(model_calls):
                model_set = set(model_calls[i])
                gt_set = set(gt_calls[i])
                score = 1.0 if model_set == gt_set else 0.0
            else:
                score = 0.0
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0

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
        }

    def _should_continue(self, metadata: MultiTurnToolMetadata) -> bool:
        """Check if conversation should continue to next turn."""
        return (
            metadata["current_turn"] < metadata["max_turns"] - 1 and
            metadata["current_turn"] < len(metadata["user_question_bank"])
        )
    
    def _get_next_observation(self, tool_results: str, metadata: MultiTurnToolMetadata) -> Dict[str, str]:
        """Generate observation for next turn or termination."""
        # TODO: ykarnati - is there better way to include  tool result and next user question?
        if self._should_continue(metadata):
            # Get next user question
            next_question = metadata["user_question_bank"][metadata["current_turn"]][0]
            observation_content = f"<tool_result>{tool_results}</tool_result>\n\n{next_question['content']}"
            return {"role": "environment", "content": observation_content}
        else:
            return {"role": "environment", "content": f"<tool_result>{tool_results}</tool_result>"}

    def _process_turn(self, message_log: LLMMessageLogType, metadata: MultiTurnToolMetadata) -> Tuple[str, List[str]]:
        """Process current turn and return tool results and calls made."""
        
        # Get latest assistant response
        assistant_response = ""
        for msg in reversed(message_log):
            if msg["role"] == "assistant":
                assistant_response = msg["content"]
                break

        model_calls_made = []
        tool_results = []
        # Check if tool tags exist
        if '<tool>' not in assistant_response:
            tool_results.append("Function call not found in current assistant response.")
            model_calls_made.append("No function call made'")
        else:
            # Parse tool calls
            model_tool_calls = self.tool_manager.parse_tool_calls(assistant_response)
            
            if not model_tool_calls:
                tool_results.append("Error: Invalid tool command. Parsing tool calls failed. Ensure correct formatting. "
                                "Tool command must be one list of JSON objects.")
                model_calls_made.append('No function call made')
            else:

                for tool_call in model_tool_calls:
                    result, success = self.tool_manager.execute_tool_call(
                        tool_call, metadata["model_tool_instances"]
                    )
                    tool_results.append(result)
                    if success:
                        func_name = tool_call.get('name', '')
                        args = tool_call.get('args', {})
                        call_str = f"{func_name}({', '.join([f'{k}={repr(v)}' for k, v in args.items()])})"
                        model_calls_made.append(call_str)
                    else:
                        # Stop on first error
                        break
        
        # Execute ground truth calls for this turn
        current_turn = metadata["current_turn"]
        if current_turn < len(metadata["ground_truth"]):
            gt_calls = metadata["ground_truth"][current_turn]
            for call_str in gt_calls:
                result, success = self._execute_gt_call(call_str, metadata["gt_tool_instances"])

        
        return "\n".join(tool_results), model_calls_made
    
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
            tool_results, model_calls = self._process_turn(
                message_log, sample_metadata
            )
            
            sample_metadata["model_calls_per_turn"].append(model_calls)
            
            # Check if should continue
            should_continue = self._should_continue(sample_metadata)
            
            # Generate observation
            observation = self._get_next_observation(tool_results, sample_metadata)
            
            # Calculate reward
            is_final_turn = not should_continue
            reward = self.reward_calculator.calculate_reward(sample_metadata, is_final_turn)
            
            # Update for next turn
            if should_continue:
                sample_metadata["current_turn"] += 1
                next_metadata.append(sample_metadata)
                terminateds.append(False)
                next_stop_strings.append(None)
            else:
                next_metadata.append(None)
                terminateds.append(True)
                next_stop_strings.append(None)
            
            observations.append(observation)
            rewards.append(reward)

        # logging for sample 0
        logging.debug("*"*100)
        logging.debug(f"[MultiTurnToolEnvironment] Current turn: {next_metadata[0]['current_turn']}")
        logging.debug(f"[MultiTurnToolEnvironment] Observation: {observations[0]}")
        logging.debug(f"[MultiTurnToolEnvironment] GT fn calls: {next_metadata[0]['ground_truth'][next_metadata[0]['current_turn'] - 1]}")
        logging.debug(f"[MultiTurnToolEnvironment] Model fn calls: {next_metadata[0]['model_calls_per_turn'][next_metadata[0]['current_turn'] - 1]}")
        logging.debug(f"[MultiTurnToolEnvironment] Reward: {rewards[0]}")
        logging.debug("*"*100)
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
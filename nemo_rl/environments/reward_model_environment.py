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
import os
import ray
import torch
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoTokenizer

from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.utils.venvs import create_local_venv
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES,RayVirtualCluster


@ray.remote
class RewardModelEnvironment(EnvironmentInterface):
    """Environment that uses a reward model to score conversations."""
    
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.BASE

    def __init__(self, config: Dict[str, Any]):
        """Initialize the reward model environment.
        
        Args:
            config: Configuration dictionary containing reward model settings
        """
        print("🚀 REWARD MODEL ENVIRONMENT INITIALIZATION STARTED")
        print("=" * 60)
        print(f"📋 Received config: {config}")
        
        self.config = config
        self.reward_model_worker = None
        self.virtual_cluster = None
        
        # Initialize reward model worker with proper resource management
        print("🔧 Setting up reward model worker...")
        self._setup_reward_model_worker()
        print("✅ REWARD MODEL ENVIRONMENT INITIALIZATION COMPLETE")


    def _setup_reward_model_worker(self):
        """Setup the reward model worker with proper resource management."""
        # Import the reward model worker class
        from nemo_rl.models.policy.dtensor_reward_model_worker import DTensorRewardModelWorker
        from nemo_rl.models.policy import PolicyConfig
        
        # Get tensor parallel size
        tensor_parallel_size = self.config.get("tensor_parallel_size", 2)  # Default to 2 instead of 4
        
        print(f"🔧 Reward model configuration:")
        print(f"   - Model: {self.config['model_name']}")
        print(f"   - Tensor Parallel Size: {tensor_parallel_size}")
        print(f"   - Dtype: {self.config.get('dtype', 'bfloat16')}")
        print(f"   - GPU Memory Utilization: {self.config.get('gpu_memory_utilization', 0.8)}")
        print(f"   - Max Model Length: {self.config.get('max_model_len', 4096)}")
        print(f"   - Full config: {self.config}")
        
        # Setup virtual cluster for proper resource allocation (following LLM judge pattern exactly)
        # NOTE: When running with GRPO, the policy already creates a virtual cluster
        # So we should NOT create another virtual cluster to avoid placement group conflicts
        # Instead, let Ray's default scheduler handle GPU allocation
        self.virtual_cluster = None
        placement_groups = []
        
        print(f"🔧 Using Ray's default scheduler (no virtual cluster) to avoid placement group conflicts with policy")
        
        # Pass down critical environment variables (similar to LLM judge)
        env_vars_to_pass = {}
        for key in [
            "HF_HOME",
            "TRANSFORMERS_CACHE", 
            "WANDB_API_KEY",
            "HUGGINGFACE_HUB_DISABLE_XET",
            "HF_TOKEN",
            "PYTORCH_CUDA_ALLOC_CONF",
        ]:
            if key in os.environ:
                env_vars_to_pass[key] = os.environ[key]
        
        # Ensure xet is disabled and CUDA memory management is set
        env_vars_to_pass.setdefault("HUGGINGFACE_HUB_DISABLE_XET", "1")
        env_vars_to_pass.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        
        # Create worker configuration compatible with PolicyConfig
        worker_config = PolicyConfig({
            "reward_model_name": self.config["model_name"],
            "precision": self.config.get("dtype", "bfloat16"),
            "reward_batch_size": self.config.get("batch_size", 2),
            "dtensor_cfg": {
                "cpu_offload": False,
                "tensor_parallel_size": tensor_parallel_size,
                "sequence_parallel": False,
                "activation_checkpointing": True,
            }
        })
        
        # Setup worker options with proper runtime environment
        worker_options = {
            "runtime_env": {
                "py_executable": self.DEFAULT_PY_EXECUTABLE,
                "env_vars": env_vars_to_pass,
            },
        }
        
        # Setup scheduling strategy - always use default Ray scheduler
        # For reward model, request all GPUs on the node (like LLM judge does)
        # This ensures proper node alignment for distributed setup
        gpus_per_node = 8  # Standard node configuration
        worker_options["num_gpus"] = gpus_per_node
        scheduling_kwargs = {}  # No placement group - use Ray's default scheduler
        
        print(f"🔧 Requesting {gpus_per_node} GPUs (full node) via Ray's default scheduler for tensor_parallel_size={tensor_parallel_size}")
        
        # Initialize the reward model worker as a Ray remote actor
        self.reward_model_worker = DTensorRewardModelWorker.options(
            **worker_options,
            **scheduling_kwargs,
        ).remote(worker_config)
        
        print(f"✓ Reward model worker initialized with {self.config['model_name']} using {tensor_parallel_size} GPUs")

    def get_rewards(
        self, reward_data: BatchedDataDict, micro_batch_size: int = None
    ) -> BatchedDataDict:
        """Get rewards for the given data.

        Args:
            reward_data: BatchedDataDict containing conversation data
            micro_batch_size: Optional micro batch size for processing

        Returns:
            BatchedDataDict with rewards
        """
        return self.reward_model_worker.get_rewards.remote(
            reward_data, micro_batch_size
        )

    def step(
        self,
        message_logs: List[LLMMessageLogType],
        env_infos: List[Dict[str, Any]],
    ) -> EnvironmentReturn:
        """Calculate rewards for the given message logs using the reward model.
        
        Args:
            message_logs: List of conversation message logs
            env_infos: List of environment info dictionaries
            
        Returns:
            EnvironmentReturn with rewards and termination info
        """
        
        # Create data in the format expected by DTensorRewardModelWorker
        reward_data = BatchedDataDict()
        reward_data["messages"] = message_logs
        
        # Deepcopy the reward_data to ensure complete isolation
        reward_data_copy = copy.deepcopy(reward_data)
        
        # Get rewards from the reward model worker using Ray remote call
        results_future = self.reward_model_worker.get_rewards.remote(reward_data_copy)
        results = ray.get(results_future)
        
        # Extract rewards from the results
        rewards_tensor = results["rewards"]
        if isinstance(rewards_tensor, torch.Tensor):
            rewards = rewards_tensor.cpu().numpy().tolist()
        else:
            rewards = rewards_tensor

        
        # Create observations with meaningful content based on rewards (like math environment)
        observations = []
        for i, reward in enumerate(rewards):
            content = "Environment: filler content"
            
            observations.append({
                "role": "environment",
                "content": content
            })
        
        # All episodes terminate after one step in reward model environment
        terminateds = [True] * len(message_logs)
        
        # No additional metadata
        metadata = [None] * len(message_logs)
        
        # No stop strings needed
        next_stop_strings = [None] * len(message_logs)

        answers = [message_log[-1]["content"] for message_log in message_logs]
        
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=torch.tensor(rewards).cpu(),
            terminateds=torch.tensor(terminateds, dtype=torch.bool).cpu(),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Post processing function after all rollouts are done for the batch and returns metrics.
        
        Args:
            batch: The batch data dictionary
            
        Returns:
            Tuple of (processed_batch, metrics_dict)
        """
        # For reward model environment, no post-processing is needed
        # Just return the batch as-is and empty metrics
        metrics = {
            "reward_model_env/num_samples": len(batch.get("message_log", [])),
        }
        
        # Add reward statistics if available
        if "rewards" in batch:
            rewards = batch["rewards"]
            if isinstance(rewards, torch.Tensor):
                metrics.update({
                    "reward_model_env/mean_reward": float(rewards.mean()),
                    "reward_model_env/std_reward": float(rewards.std()),
                    "reward_model_env/min_reward": float(rewards.min()),
                    "reward_model_env/max_reward": float(rewards.max()),
                })
        
        return batch, metrics

    def shutdown(self):
        """Shutdown the reward model worker and virtual cluster."""
        if self.reward_model_worker is not None:
            try:
                ray.kill(self.reward_model_worker)
            except Exception as e:
                print(f"Warning: Error shutting down reward model worker: {e}")
            self.reward_model_worker = None
        
        if self.virtual_cluster is not None:
            try:
                self.virtual_cluster.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down virtual cluster: {e}")
            self.virtual_cluster = None 
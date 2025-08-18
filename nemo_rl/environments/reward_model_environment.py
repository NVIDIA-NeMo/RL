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
import ray
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.utils.venvs import create_local_venv
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES,RayVirtualCluster
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.models.policy.lm_policy import Policy

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
        self.virtual_cluster = RayVirtualCluster(
            name="grpo_reward_model_cluster",
            bundle_ct_per_node_list=[self.config["resources"]["gpus_per_node"]] * self.config["resources"]["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=self.config["resources"]["gpus_per_node"],
            max_colocated_worker_groups=1,
        )
        
        # Initialize reward model worker with proper resource management
        print("🔧 Setting up reward model worker...")
        self._setup_reward_model_worker()
        print("✅ REWARD MODEL ENVIRONMENT INITIALIZATION COMPLETE")


    def _setup_reward_model_worker(self):
        """Setup the reward model worker with proper resource management."""
        # Import the reward model worker class
        

        print(f"🔧 Reward model configuration:")
        print(f"   - Model: {self.config['model_name']}")
        print(f"   - Tensor Parallel Size: {self.config['dtensor_cfg']['tensor_parallel_size']}")
        print(f"   - Dtype: {self.config.get('dtype', 'bfloat16')}")
        print(f"   - GPU Memory Utilization: {self.config.get('gpu_memory_utilization', 0.8)}")
        print(f"   - Max Model Length: {self.config.get('max_model_len', 4096)}")
        print(f"   - Full config: {self.config}")

        weights_path = self.config.get("checkpoint_path", None)
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
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
        self.config["env_vars"] = env_vars_to_pass
        self.reward_model_policy = Policy(
            cluster=self.virtual_cluster,
            config=self.config,
            tokenizer=tokenizer,
            name_prefix="reward_model_policy",
            init_optimizer=False,
            init_reference_model=False,
            weights_path=weights_path,
        )

        print(f"✓ Reward model worker initialized with {self.config['model_name']} using {self.config['resources']['num_nodes']} Nodes")

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
        dp_size = self.reward_model_policy.sharding_annotations.get_axis_size("data_parallel")
        sharded_data = reward_data.shard_by_batch_size(dp_size, batch_size=None)
        return self.reward_model_policy.worker_group.run_all_workers_sharded_data(
            "get_rewards",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["tensor_parallel", "pipeline_parallel"],
            common_kwargs={"micro_batch_size": micro_batch_size},
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
        results = BatchedDataDict.from_batches(
            self.reward_model_policy.worker_group.get_all_worker_results(results_future),

        )
        
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
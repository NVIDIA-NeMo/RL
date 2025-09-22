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

import os
from typing import Any, Dict, List, NotRequired, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.interfaces import LLMMessageLogType, TaskDataSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_formatted_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.models.generation.vllm import VllmConfig
from nemo_rl.models.policy import DynamicBatchingConfig, SequencePackingConfig
from nemo_rl.models.policy.lm_policy import Policy


class RewardModelEnvironmentConfig(TypedDict):
    """Configuration for RewardModelEnvironment.

    Attributes:
        enabled: Whether the reward model environment is enabled
        model_name: Name of the reward model to use (e.g., "Skywork/Skywork-Reward-V2-Qwen3-0.6B")
        tokenizer: Tokenizer configuration
        precision: Model precision (e.g., "bfloat16", "float16", "float32")
        batch_size: Batch size for processing conversations
        checkpoint_path: Path to model checkpoint (optional)
        max_model_len: Maximum sequence length for the model
        logprob_batch_size: Batch size for log probability computation
        resources: Resource allocation configuration
        reward_model_cfg: Reward model specific configuration
        dtensor_cfg: DTensor configuration for distributed training
        dynamic_batching: Dynamic batching configuration
        sequence_packing: Sequence packing configuration
        max_grad_norm: Maximum gradient norm for training
        generation: Generation configuration for VLLM
    """

    enabled: bool
    model_name: str
    precision: str
    batch_size: int
    checkpoint_path: str
    logprob_batch_size: int
    resources: Dict[str, Any]
    dtensor_cfg: Optional[Dict[str, Any]]
    dynamic_batching: DynamicBatchingConfig = {"enabled": False}
    sequence_packing: NotRequired[SequencePackingConfig] = {"enabled": False}
    max_grad_norm: Optional[float] = None
    generation: Optional[VllmConfig] = None


@ray.remote
class RewardModelEnvironment(EnvironmentInterface):
    """Environment that uses a reward model to score conversations.

    This environment implements a reward model-based scoring system for reinforcement
    learning tasks. It takes conversation logs as input and returns rewards based on
    the quality of the assistant's responses as judged by a pre-trained reward model.

    Attributes:
        config: Configuration dictionary containing all environment settings
        virtual_cluster: Ray virtual cluster for resource management
        tokenizer: Tokenizer for text processing
        reward_model_policy: Policy object containing the reward model
    """

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.BASE

    def __init__(self, config: Dict[str, Any]):
        """Initialize the reward model environment.

        Args:
            config: Configuration dictionary containing reward model settings.
                   Must include model_name, tokenizer, resources, and other
                   required parameters as defined in RewardModelEnvironmentConfig.
        """
        print("🚀 REWARD MODEL ENVIRONMENT INITIALIZATION STARTED")
        print("=" * 60)
        print(f"📋 Received config: {config}")

        self.config = config

        assert self.config["reward_model_cfg"]["enabled"], (
            "Please set reward_model_cfg.enabled = True in the reward model environment config to enable reward model."
        )
        assert (
            self.config["reward_model_cfg"]["reward_model_type"] == "bradley_terry"
        ), (
            "Reward model environment currently only support with Bradley-Terry reward model."
        )
        assert not self.config["dynamic_batching"]["enabled"], (
            "Dynamic batching is currently not supported with reward model environment."
        )
        assert not self.config["sequence_packing"]["enabled"], (
            "Sequence packing is currently not supported with reward model environment."
        )
        assert self.config["dtensor_cfg"]["enabled"], (
            "Reward model environment currently only support with DTensor. You can show your interest in mcore path by upvoting on https://github.com/NVIDIA-NeMo/RL/issues/1154"
        )
        assert self.config["max_grad_norm"] == None, (
            "Max grad norm must be None in reward model environment."
        )
        assert not self.config["dtensor_cfg"]["cpu_offload"], (
            "CPU offload is currently not supported with reward model environment."
        )
        assert not self.config["dtensor_cfg"]["activation_checkpointing"], (
            "Activation checkpointing is currently not supported with reward model environment."
        )
        # Add values for reward model cfg. reward_model_cfg must be enabled in reward model environment config.
        self.config.setdefault("reward_model_cfg", {})
        self.config["reward_model_cfg"]["enabled"] = True
        self.config["reward_model_cfg"]["reward_model_type"] = "bradley_terry"
        # Dynamic batching and sequence packing are disabled in reward model environment config.
        self.config.setdefault("dynamic_batching", {})
        self.config.setdefault("sequence_packing", {})
        self.config["dynamic_batching"]["enabled"] = False
        self.config["sequence_packing"]["enabled"] = False
        self.config["max_grad_norm"] = None
        # Reward model environment is always using DTensor
        self.config["dtensor_cfg"]["enabled"] = True
        self.config["dtensor_cfg"]["cpu_offload"] = False
        self.config["dtensor_cfg"]["activation_checkpointing"] = False

        self.task_data_spec = TaskDataSpec(
            task_name="reward_model_env",
        )

        # Remove CUDA_VISIBLE_DEVICES to let ray fully control the GPU allocation
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        self.virtual_cluster = RayVirtualCluster(
            name="grpo_reward_model_cluster",
            bundle_ct_per_node_list=[self.config["resources"]["gpus_per_node"]]
            * self.config["resources"]["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=self.config["resources"]["gpus_per_node"],
            max_colocated_worker_groups=1,
        )
        print(
            f"🔧 Virtual cluster created with {self.virtual_cluster.get_placement_groups()} "
        )
        # Initialize reward model worker with proper resource management
        print("🔧 Setting up reward model worker...")
        weights_path = self.config.get("checkpoint_path", None)
        # Initialize tokenizer
        self.tokenizer = get_tokenizer(self.config["tokenizer"])

        print(
            f"✅ Tokenizer initialized with pad_token_id: {self.tokenizer.pad_token_id}"
        )
        self.reward_model_policy = None
        self.reward_model_policy = Policy(
            cluster=self.virtual_cluster,
            config=self.config,
            tokenizer=self.tokenizer,
            name_prefix="reward_model_policy",
            init_optimizer=False,
            init_reference_model=False,
            weights_path=weights_path,
        )

        print("✅ REWARD MODEL ENVIRONMENT INITIALIZATION COMPLETE")

    def preprocess_data(
        self, message_logs: List[LLMMessageLogType]
    ) -> BatchedDataDict[GenerationDatumSpec]:
        """Preprocess the message logs for the reward model.

        This method tokenizes and formats conversation logs into the format expected
        by the reward model. It handles:
        - Tokenization of user and assistant messages
        - Formatting with proper special tokens
        - Batching and padding for efficient processing
        - Sequence length validation and truncation

        Args:
            message_logs: List of conversation message logs, where each log contains
                         a list of messages with 'role' and 'content' fields.

        Returns:
            BatchedDataDict containing tokenized and formatted data ready for
            reward model inference.
        """
        # Tokenize each message_log
        tokenized_message_logs = []
        for message_log in message_logs:
            tokenized_log = get_formatted_message_log(
                message_log,
                tokenizer=self.tokenizer,
                task_data_spec=self.task_data_spec,
                add_bos_token=True,
                add_eos_token=True,
                add_generation_prompt=False,
            )
            tokenized_message_logs.append(tokenized_log)

        # Convert message logs to flat representation and pad for batching
        cat_and_padded, input_lengths = batched_message_log_to_flat_message(
            tokenized_message_logs,
            pad_value_dict={"token_ids": self.tokenizer.pad_token_id},
        )

        # Create data in the format expected by DTensorRewardModelWorker
        reward_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": cat_and_padded["token_ids"],
                "input_lengths": input_lengths,
            }
        )
        return reward_data

    def step(
        self,
        message_logs: List[LLMMessageLogType],
        env_infos: List[Dict[str, Any]],
    ) -> EnvironmentReturn:
        """Calculate rewards for the given message logs using the reward model.

        This method processes conversation logs through the reward model to compute
        quality scores for each conversation. The rewards are based on the reward
        model's assessment of how well the assistant's responses align with human
        preferences.

        Args:
            message_logs: List of conversation message logs to be scored.
                         Each log should contain alternating user and assistant messages.
            env_infos: List of environment info dictionaries (currently unused
                      but required by the interface).

        Returns:
            EnvironmentReturn containing:
            - observations: List of observation dictionaries with reward information
            - metadata: List of metadata dictionaries (currently None)
            - next_stop_strings: List of stop strings (currently None)
            - rewards: Tensor of computed rewards for each conversation
            - terminateds: Tensor indicating episode termination (all True)
            - answers: List of assistant responses from the conversations

        """
        # Preprocess the message logs
        reward_data = self.preprocess_data(message_logs)

        # Score the message logs
        rewards = self.reward_model_policy.score(reward_data)["scores"]

        # Create observations with meaningful content based on rewards (like math environment)
        observations = []
        for i, reward in enumerate(rewards):
            content = "Environment: " + str(reward)
            observations.append({"role": "environment", "content": content})

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
            rewards=rewards.cpu(),
            terminateds=torch.tensor(terminateds, dtype=torch.bool).cpu(),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Post processing function after all rollouts are done for the batch and returns metrics.

        This method computes aggregate statistics and metrics from the processed batch.
        It provides insights into reward distribution and processing statistics.

        Args:
            batch: The batch data dictionary containing processed conversations and rewards.

        Returns:
            Tuple of (processed_batch, metrics_dict) where:
            - processed_batch: The input batch (no modifications)
            - metrics_dict: Dictionary containing computed metrics including:
              - reward_model_env/num_samples: Number of samples processed
              - reward_model_env/mean_reward: Average reward across the batch
              - reward_model_env/std_reward: Standard deviation of rewards
              - reward_model_env/min_reward: Minimum reward in the batch
              - reward_model_env/max_reward: Maximum reward in the batch
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
                metrics.update(
                    {
                        "reward_model_env/mean_reward": float(rewards.mean()),
                        "reward_model_env/std_reward": float(rewards.std()),
                        "reward_model_env/min_reward": float(rewards.min()),
                        "reward_model_env/max_reward": float(rewards.max()),
                    }
                )

        return batch, metrics

    def shutdown(self):
        """Shutdown the reward model worker and virtual cluster.

        This method properly cleans up resources by shutting down the reward model
        policy and virtual cluster. It should be called when the environment is
        no longer needed to prevent resource leaks.

        Note:
            The environment will also automatically call this method in its destructor,
            but it's recommended to call it explicitly for better resource management.
        """
        if (
            hasattr(self, "reward_model_policy")
            and self.reward_model_policy is not None
        ):
            try:
                self.reward_model_policy.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down reward model policy: {e}")
            self.reward_model_policy = None
            try:
                self.virtual_cluster.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down virtual cluster: {e}")
            self.virtual_cluster = None

    def __del__(self):
        """Destructor that ensures proper cleanup when the object is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown() and
        the pointer to the object is lost due to leaving a function scope. It's always
        recommended that the user calls shutdown() explicitly for better resource
        management.
        """
        self.shutdown()

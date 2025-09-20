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

from typing import Any, Dict, List, Optional, Tuple, TypedDict

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
from nemo_rl.models.policy.lm_policy import Policy


class RewardModelEnvironmentConfig(TypedDict):
    enabled: bool
    model_name: str
    precision: str
    batch_size: int
    checkpoint_path: str
    logprob_batch_size: int
    resources: Dict[str, Any]
    dtensor_cfg: Optional[Dict[str, Any]] = None
    dynamic_batching: Optional[Dict[str, Any]] = None
    sequence_packing: Optional[Dict[str, Any]] = None
    max_grad_norm: Optional[float] = None
    generation: Optional[VllmConfig] = None


# @ray.remote
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
            bundle_ct_per_node_list=[self.config["resources"]["gpus_per_node"]]
            * self.config["resources"]["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=self.config["resources"]["gpus_per_node"],
            max_colocated_worker_groups=1,
        )
        print(
            f"🔧 Virtual cluster created with {self.virtual_cluster.get_placement_groups()} "
        )
        self.reward_model_policy = None

        # Initialize reward model worker with proper resource management
        print("🔧 Setting up reward model worker...")
        weights_path = self.config.get("checkpoint_path", None)
        # Initialize tokenizer
        self.tokenizer = get_tokenizer(self.config["tokenizer"])

        # Ensure tokenizer has pad_token_id
        # self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        #     print(
        #         f"⚠️  Tokenizer pad_token_id was None, set to {self.tokenizer.pad_token_id}"
        #     )

        print(
            f"✅ Tokenizer initialized with pad_token_id: {self.tokenizer.pad_token_id}"
        )

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

        Args:
            message_logs: List of conversation message logs

        """
        task_data_spec = TaskDataSpec(
            task_name="reward_model_env",
        )

        # Tokenize each message_log
        tokenized_message_logs = []
        for message_log in message_logs:
            tokenized_log = get_formatted_message_log(
                message_log,
                tokenizer=self.tokenizer,
                task_data_spec=task_data_spec,
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

        max_model_len = self.config["max_model_len"]
        if input_lengths.max().item() > max_model_len:
            print(
                f"⚠️  Truncating sequences from {input_lengths.max().item()} to {max_model_len}"
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

        Args:
            message_logs: List of conversation message logs
            env_infos: List of environment info dictionaries

        Returns:
            EnvironmentReturn with rewards and termination info
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
        """Shutdown the reward model worker and virtual cluster."""
        if self.reward_model_policy is not None:
            try:
                self.reward_model_policy.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down reward model policy: {e}")
            self.reward_model_policy = None

        if self.virtual_cluster is not None:
            try:
                self.virtual_cluster.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down virtual cluster: {e}")
            self.virtual_cluster = None

    def __del__(self):
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls shutdown().
        """
        self.shutdown()

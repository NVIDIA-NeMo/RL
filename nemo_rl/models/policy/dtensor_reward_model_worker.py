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

import gc
import os
import copy
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import ray
import torch
from torch import nn
from torch.distributed.fsdp import (
    FSDPModule,
)
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.accelerate import find_tied_parameters

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    PY_EXECUTABLES,
)
from nemo_rl.models.dtensor.parallelize import (
    _parallelize_model,
    to_local_if_dtensor,
)
from nemo_rl.models.huggingface.common import ModelFlag
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import (
    get_gpu_info,
    sliding_window_overwrite,
)



@ray.remote
class DTensorRewardModelWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.BASE

    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        weights_path: Optional[str] = None,
    ):
        self.cfg = config
        
        # Print initial configuration
        print("=" * 60)
        print("🚀 INITIALIZING REWARD MODEL WORKER")
        print("=" * 60)
        
        # Show CUDA environment
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPUs detected")
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                print(f"✓ CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            else:
                print("⚠️  CUDA_VISIBLE_DEVICES not set - using all available GPUs")
            
            # Show current device assignment
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"✓ Current CUDA device: {current_device} ({device_name})")
        else:
            print("❌ CUDA not available - will use CPU")
        
        # Check if we're in a distributed environment
        # If not, skip distributed setup for single-actor testing
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # torch distributed init. Envars for rank, world_size, and master_addr and master_port are set from the ray remote call
            torch.distributed.init_process_group(backend="nccl")
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            use_distributed = True
            print(f"✓ Distributed setup: Rank {rank}/{world_size}")
        else:
            # Single process mode for testing
            print("✓ Running in single-process mode (no distributed setup)")
            rank = 0
            world_size = 1
            use_distributed = False
        
        # Use the reward model name from config or default to Nemotron-70B-Reward
        model_name = self.cfg.get("reward_model_name", "nvidia/Llama-3.1-Nemotron-70B-Reward-HF")
        tensor_parallel_size = self.cfg["dtensor_cfg"]["tensor_parallel_size"]
        
        print(f"📋 Model configuration:")
        print(f"   - Model: {model_name}")
        print(f"   - Tensor Parallel Size: {tensor_parallel_size}")
        print(f"   - Precision: {self.cfg['precision']}")

        self.cpu_offload = self.cfg["dtensor_cfg"]["cpu_offload"]

        if self.cfg["precision"] == "float32":
            self.dtype = torch.float32
        elif self.cfg["precision"] == "bfloat16":
            self.dtype = torch.bfloat16
        elif self.cfg["precision"] == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError(f"Unknown precision: {self.cfg['precision']}")

        print(f"🔄 [Rank {rank}] Loading reward model {model_name}...")
        print(f"   This may take several minutes for large models...")
        
        # Add a small staggered delay to reduce network congestion when multiple ranks download simultaneously
        if use_distributed and rank > 0:
            delay_seconds = rank * 2  # 2 seconds per rank
            print(f"   [Rank {rank}] Waiting {delay_seconds}s to stagger downloads...")
            import time
            time.sleep(delay_seconds)
        
        # Show memory before loading
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / (1024**3)
            reserved_before = torch.cuda.memory_reserved() / (1024**3)
            print(f"   GPU memory before loading: {allocated_before:.2f}GB allocated, {reserved_before:.2f}GB reserved")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            **sliding_window_overwrite(
                model_name
            ),  # due to https://github.com/huggingface/transformers/issues/38002
        )
        
        # Load tokenizer for the same model
        print(f"🔤 [Rank {rank}] Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Only do DTensor setup if we're in distributed mode
        if use_distributed and world_size > 1:
            print(f"⚙️  Setting up DTensor for distributed training...")
            # caching since this property is not always preserved after FSDP
            self.num_tied_weights = len(find_tied_parameters(self.model))
            self.skip_tie_check = os.environ.get(
                "NRL_SKIP_TIED_WEIGHT_CHECK"
            ) or ModelFlag.SKIP_DTENSOR_TIED_WEIGHTS_CHECK.matches(model_name)

            # ------------------------------------------------
            # Initialize device mesh for tensor parallelism
            # ------------------------------------------------
            tp_size = self.cfg["dtensor_cfg"]["tensor_parallel_size"]
            dp_size = world_size // tp_size
            assert world_size % tp_size == 0, (
                f"World size({world_size}) must be divisible by TP size({tp_size}) to use DTensor"
            )

            mesh_2d = torch.distributed.device_mesh.init_device_mesh(
                "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
            )
            self.dp_mesh, self.tp_mesh = mesh_2d["dp"], mesh_2d["tp"]
            self.dp_size = dp_size
            self.tp_size = tp_size

            print(f"   - Data Parallel Size: {dp_size}")
            print(f"   - Tensor Parallel Size: {tp_size}")

            # ------------------------------------------------
            # Parallelize the model using DTensor
            # ------------------------------------------------
            print(f"🔧 Parallelizing model with DTensor...")
            self.model = _parallelize_model(
                self.model,
                self.tp_mesh,
                self.cfg["dtensor_cfg"]["sequence_parallel"],
                self.cfg["dtensor_cfg"]["activation_checkpointing"],
                self.skip_tie_check,
            )
            print(f"✓ Model parallelized successfully")
        else:
            print(f"✓ Single-process mode - skipping DTensor setup")
            self.dp_mesh = None
            self.tp_mesh = None
            self.dp_size = 1
            self.tp_size = 1

        print("✅ REWARD MODEL WORKER INITIALIZATION COMPLETE")
        print("=" * 60)

    def is_alive(self):
        return True

    def get_gpu_info(self):
        return get_gpu_info()


    def get_rewards(
        self, data: BatchedDataDict, micro_batch_size: int = None
    ) -> BatchedDataDict:
        """Get rewards for conversation sequences.

        Args:
            data: BatchedDataDict containing:
                - "messages": List of conversation histories (list of messages with role and content)

        Returns:
            BatchedDataDict with "rewards" key containing scalar rewards [batch_size]
        """
        print("🎯 REWARD MODEL INFERENCE STARTING")
        print("=" * 50)
        
        reward_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg.get("reward_batch_size", 8)
        )
        
        all_rewards = []
        self.model.eval()

        # Process in batches
        with torch.no_grad():
            
            # Process conversation histories
            conversations = data.get("messages")
            
            print(f"🔄 Processing {len(conversations)} conversations...")
            
            for i in range(0, len(conversations), reward_batch_size):
                batch_conversations = conversations[i:i + reward_batch_size]
                
                print(f"   Batch {i//reward_batch_size + 1}: conversations {i+1}-{min(i+reward_batch_size, len(conversations))}")
                
                batch_rewards = []
                for j, conversation in enumerate(batch_conversations):
                    # Apply chat template and tokenize
                    tokenized_message = self.tokenizer.apply_chat_template(
                        conversation, 
                        tokenize=True, 
                        add_generation_prompt=False, 
                        return_tensors="pt", 
                        return_dict=True
                    )
                    
                    # Move to GPU if available and not already there
                    device = next(self.model.parameters()).device
                    if device.type == 'cuda':
                        input_ids = tokenized_message['input_ids'].to(device)
                        attention_mask = tokenized_message['attention_mask'].to(device)
                    else:
                        # Fallback to CPU if no GPU available
                        input_ids = tokenized_message['input_ids']
                        attention_mask = tokenized_message['attention_mask']
                    
                    # Generate one token to get reward score
                    device = next(self.model.parameters()).device
                    autocast_device = device.type if device.type in ['cuda', 'cpu'] else 'cpu'
                    
                    # Clone tensors to avoid modifying the original tensors in-place during autocast
                    input_ids_clone = copy.deepcopy(input_ids)
                    attention_mask_clone = copy.deepcopy(attention_mask)
                    
                    with torch.autocast(device_type=autocast_device, dtype=self.dtype):
                        response_token_ids = self.model.generate(
                            input_ids_clone,
                            attention_mask=attention_mask_clone,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        
                        # Extract reward from the first token's first logit
                        reward = response_token_ids['scores'][0][0][0].item()
                        batch_rewards.append(reward)
                        
                        # Show sample details for first few
                        if i == 0 and j < 2:  # Only show first 2 samples of first batch
                            print(f"     Sample {j+1}: reward = {reward:.4f}")
                            print(f"       Full conversation history:")
                            for msg in conversation:
                                print(f"         {msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                
                all_rewards.extend(batch_rewards)
                print(f"     Batch rewards: {[f'{r:.4f}' for r in batch_rewards]}")

        # Concatenate all batches and return
        if all_rewards and isinstance(all_rewards[0], (int, float)):
            # From prompt-response processing - all_rewards is a flat list of floats
            rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
        elif all_rewards and torch.is_tensor(all_rewards[0]):
            # From pre-tokenized processing - all_rewards is a list of tensors
            rewards_tensor = torch.cat(all_rewards, dim=0)
        else:
            # Empty or unknown format
            rewards_tensor = torch.tensor([], dtype=torch.float32)
        
        # Final summary
        if len(rewards_tensor) > 0:
            mean_reward = rewards_tensor.mean().item()
            std_reward = rewards_tensor.std().item()
            min_reward = rewards_tensor.min().item()
            max_reward = rewards_tensor.max().item()
            
            print(f"📈 INFERENCE COMPLETE:")
            print(f"   Total samples: {len(rewards_tensor)}")
            print(f"   Mean reward: {mean_reward:.4f}")
            print(f"   Std reward: {std_reward:.4f}")
            print(f"   Min reward: {min_reward:.4f}")
            print(f"   Max reward: {max_reward:.4f}")
        else:
            print("⚠️  No rewards computed!")
        
        print("=" * 50)
            
        return_data = BatchedDataDict()
        return_data["rewards"] = rewards_tensor
        
        return return_data

    def shutdown(self):
        """Shutdown the worker and clean up resources."""
        try:
            if hasattr(self, 'model'):
                del self.model
            
            torch.cuda.empty_cache()
            gc.collect()
            
            # Only destroy process group if it was initialized
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                
            return True
        except Exception as e:
            print(f"Error during reward model worker shutdown: {e}")
            return False 
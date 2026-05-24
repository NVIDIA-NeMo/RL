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

"""Generate a JSONL dataset of NeMo-RL technical questions for RLHF training."""

import json
from pathlib import Path

QUESTIONS = [
    # Environment & architecture
    "How do I add a custom environment to NeMo-RL?",
    "What is the difference between native environments and NeMo-Gym environments?",
    "How does the EnvironmentInterface work in NeMo-RL?",
    "What is the difference between single-turn and multi-turn environments?",
    "How do I register a new environment in ENV_REGISTRY?",
    "What does the step() method return in an environment?",
    "How do I implement global_post_process_and_metrics()?",
    "What is EnvironmentReturn and what fields does it contain?",
    "How do I create a multi-turn environment like the sliding puzzle?",
    "What is the role of metadata in multi-turn environments?",
    # GRPO training
    "Why is my GRPO training reward flat at 0?",
    "How do I configure GRPO hyperparameters for a new task?",
    "What does num_generations_per_prompt control in GRPO?",
    "How does the KL penalty work in GRPO training?",
    "What is the leave-one-out baseline in GRPO?",
    "How do I enable reward normalization in GRPO?",
    "What is the difference between GRPO and Reinforce++ advantage estimators?",
    "How do I use dynamic sampling in GRPO?",
    "What does max_rollout_turns control?",
    "How do I configure the ratio clipping parameters?",
    # Multi-GPU & distributed training
    "How do I configure multi-GPU training with FSDP?",
    "What is the difference between DTensor and Megatron paths?",
    "How do I enable tensor parallelism in NeMo-RL?",
    "What does cpu_offload do in the DTensor config?",
    "How do I configure sequence parallelism?",
    "What is activation checkpointing and when should I use it?",
    "How do I scale from 1 GPU to 8 GPUs?",
    "What is the difference between colocated and non-colocated generation?",
    "How do I configure distributed data parallel settings?",
    "What does train_global_batch_size need to be for GRPO?",
    # Reward models & RLHF
    "How do I use a reward model for RLHF training?",
    "What reward models are compatible with NeMo-RL?",
    "How do I configure the Skywork reward model?",
    "What is a Bradley-Terry reward model?",
    "How do I allocate GPUs between policy and reward model?",
    "Why does the reward model require DTensor to be enabled?",
    "How do I interpret negative reward scores from the reward model?",
    "What is reward scaling and when should I use it?",
    "How do I switch from a rule-based environment to a reward model?",
    "What does the reward_model_type parameter control?",
    # vLLM generation
    "My vLLM generation is running out of memory, how do I fix it?",
    "How do I configure gpu_memory_utilization for vLLM?",
    "What does max_model_len control in vLLM?",
    "How do I enable tensor parallel generation with vLLM?",
    "What is the difference between eager and compiled vLLM execution?",
    "How do I configure stop_strings for generation?",
    "What does the async_engine option do in vLLM?",
    "How do I set temperature and top_p for generation?",
    "What is the relationship between max_new_tokens and max_total_sequence_length?",
    "How do I debug vLLM convergence issues?",
    # Data & datasets
    "How do I add a new dataset to NeMo-RL?",
    "What is the openai_format dataset and how do I use it?",
    "How do I create a custom data processor?",
    "What is a DatumSpec and what fields does it require?",
    "How do I configure multiple datasets for training?",
    "What does the sft_processor do?",
    "How do I handle overlength sequences in a data processor?",
    "What is the PROCESSOR_REGISTRY and how do I register a new processor?",
    "How do I use a JSONL file as training data?",
    "What is the difference between ResponseDataset and openai_format?",
    # Configuration
    "How do I create a new YAML config for a training run?",
    "What does the defaults field do in a YAML config?",
    "How do I override config values from the command line?",
    "What are the required sections in a GRPO config?",
    "How do I configure wandb logging?",
    "What does the checkpointing section control?",
    "How do I set up validation during training?",
    "What is the cluster config section for?",
    "How do I configure the learning rate scheduler?",
    "What does precision: bfloat16 mean?",
    # Model selection
    "Which model should I start with for a new RL task?",
    "What is the tradeoff between base and instruct models for RL?",
    "How do I use Qwen3-0.6B for GRPO training?",
    "What models support the DTensor path?",
    "How do I configure a model's tokenizer separately?",
    "What does enable_thinking do in chat_template_kwargs?",
    "How do I use LoRA with GRPO training?",
    "What is the max_total_sequence_length and how do I choose it?",
    "How do I fine-tune a model that keeps running out of GPU memory?",
    "What is the difference between train_micro_batch_size and train_global_batch_size?",
    # Debugging & troubleshooting
    "My training loss is NaN, what should I check?",
    "How do I debug a data processor that produces incorrect tokenization?",
    "What does 'total batch size is not a multiple of batch_size' mean?",
    "My reward is oscillating and not converging, what should I do?",
    "How do I inspect logged samples during training?",
    "What does the seq_logprob_error_threshold parameter do?",
    "How do I fix 'CPUOffload doesn't work on single GPU for AutoModel'?",
    "My environment returns all zeros for rewards, how do I debug it?",
    "How do I check if my data processor formats prompts correctly?",
    "What causes KL divergence to explode during training?",
    # Advanced topics
    "How do I implement tool-calling in a NeMo-RL environment?",
    "What is sequence packing and when should I enable it?",
    "How do I configure async GRPO training?",
    "What is importance sampling correction in the loss function?",
    "How do I use FP8 training in NeMo-RL?",
    "What is the draft model configuration for speculative decoding?",
    "How do I configure MoE models with NeMo-RL?",
    "What is the difference between dynamic batching and sequence packing?",
    "How do I run NeMo-RL on Kubernetes?",
    "How do I save and load checkpoints during GRPO training?",
]


def main():
    output_path = Path(__file__).parent / "nemorl_questions.jsonl"
    with open(output_path, "w") as f:
        for question in QUESTIONS:
            # Format: user question + placeholder answer.
            # The math_hf_data_processor expects messages[0]=user, messages[1]=assistant.
            # For RLHF with a reward model, the ground truth is unused — the reward model
            # scores model outputs directly.
            entry = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ""},
                ],
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(QUESTIONS)} questions to {output_path}")


if __name__ == "__main__":
    main()

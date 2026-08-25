
# Performance

As part of the NVIDIA NeMo Framework, NeMo RL provides optimal performance for reinforcement learning on generative AI models by incorporating the latest optimizations - such as refit optimizations, mixed-precision training, and off-policy training.

This page provides performance benchmarks for LLMs and VLMs using NeMo RL across different GPU systems and configurations. The recipes to reproduce these runs, in yaml file form, can be found under [this folder](https://github.com/NVIDIA-NeMo/RL/tree/v0.7.0/examples/configs/recipes/llm/performance).

## Nomenclature

- **GBS**: Global Batch Size
- **MBS**: Micro Batch Size
- **TP**: Tensor Parallel Size
- **PP**: Pipeline Parallel Size
- **CP**: Context Parallel Size
- **VPP**: Virtual Pipeline Parallel Size
- **EP**: Expert Parallel Size
- **T-**: Training related
- **G-**: Generation related
- **T-GBS**: Training global batch size (`policy.train_global_batch_size`)
- **G-GBS**: Number of rollout samples generated per step (`grpo.num_prompts_per_step * grpo.num_generations_per_prompt`)
- **Training backend**: NeMo RL has two training backends: Megatron and PyTorch DTensor. This performance summary currently only shows numbers from the Megatron backend.

## Performance Metrics

Since reinforcement learning consists of training, generation and transition between the two, performance measurement also reflects this. Specifically, we track the following metrics:
- **Step time**: Time for each step, which includes training, generation, policy logprobs, and refit time.
- **Tokens/sec/GPU**: The rate at which the tokens are processed by a stage (such as training, generation, or refitting) on a single GPU:

    $$
    \text{Tokens/sec/GPU} = \frac{\text{Total Tokens Processed}}{\text{Time for Stage} \times \text{Number of GPUs}}
    $$

- **Training MFU**: Model floating-point operations per second per GPU


## Performance Summary for Large Language Models

Below are performance benchmarks for various large language models organized by release version. These results were obtained using performance recipes available [here](https://github.com/NVIDIA-NeMo/RL/tree/v0.7.0/examples/configs/recipes/llm/performance).

The performance data includes:

- **RL Performance**: Performance metrics for various model sizes and architectures on different RL algorithms (GRPO and in the future DAPO, PPO, for both on-policy and asynchronous).
- **System Configurations**: Results across different GPU systems (DGX-H100 and in the future DGX-GB200, DGX-B200)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8)

---

## NeMo RL v0.7

### H100 BF16 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2); DAPO dataset: [DAPOMath17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
* System: DGX-H100
* Precision: Training BF16, Generation BF16
* Training Backend: Megatron Core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO | DeepSeek V3 | On policy | 1,536 | 722 | 256 | 512 | 512 | [32,1] | [1,1,16,16,n/a] | 17.8 | 93 |
| GRPO | DeepSeek V3 | On policy | 1,536 | 724 | 512 | 512 | 512 | [32,1] | [1,1,32,16,n/a] | 10.2 | 81.2 |
| GRPO | DeepSeek V3 | 1-step Off | 1,536 | 729 | 512 | 512 | 512 | [32,1] | [1,1,16,16,n/a] | 18 | 46.6 |
| GRPO | Qwen3-235B-A22B | On policy | 8,192 | 5,719 | 128 | 512 | 512 | [16,1] | [2,2,16,8,n/a] | 63.7 | 367 |
| GRPO | Qwen3-235B-A22B | On policy | 8,192 | 5,698 | 256 | 512 | 512 | [16,1] | [2,2,16,8,n/a] | 40.8 | 285 |
| GRPO | Qwen3-235B-A22B | 1-step Off | 8,192 | 5,695 | 256 | 512 | 512 | [8,1] | [4,1,16,8,n/a] | 58.3 | 204 |
| GRPO | Qwen3-30B-A3B | On policy | 4,096 | 3,198 | 32 | 2,048 | 2,048 | [2,1] | [1,1,8,1,n/a] | 1,232 | 171 |
| GRPO | Qwen3-30B-A3B | 1-step Off | 4,096 | 3,203 | 32 | 2,048 | 2,048 | [2,1] | [1,1,8,2,n/a] | 1,522 | 141 |
| GRPO | Qwen3-30B-A3B | 8-step Off | 4,096 | 3,203 | 192 | 2,048 | 512 | [2,1] | [1,1,8,1,n/a] | 1,067 | 33.1 |
| GRPO | Qwen3-32B | 1-step Off | 4,096 | 3,258 | 64 | 2,048 | 2,048 | [4,1] | [4,1,1,4,n/a] | 675 | 161 |
| GRPO | Qwen3-32B | On policy | 4,096 | 3,256 | 32 | 2,048 | 2,048 | [4,1] | [4,1,1,4,n/a] | 665 | 323 |
| GRPO | Qwen3-30B-A3B | On policy | 40,960 | 8,170 | 32 | 2,048 | 512 | [2,1] | [4,8,8,1,n/a] | 262 | 2,022 |
| GRPO | Nemotron-3-Super-120B-A12B | On policy | 8,192 | 3,197 | 256 | 256 | 256 | [8,1] | [4,1,32,1,n/a] | 30.7 | 108 |
| GRPO | Nemotron-3-Super-120B-A12B | 1-step Off | 8,192 | 3,207 | 256 | 256 | 256 | [8,1] | [4,1,32,1,n/a] | 48.6 | 71.9 |

### H100 FP8 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
* System: DGX-H100
* Precision: Generation FP8, Training FP8
* Training Backend: Megatron Core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO | DeepSeek V3 | 1-step Off | 1,536 | 738 | 512 | 512 | 512 | [16,1] | [1,1,16,16,n/a] | 14 | 61.9 |

### GB200 BF16 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
* System: GB200-NVL72
* Precision: Training BF16, Generation BF16
* Training Backend: Megatron Core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO | DeepSeek V3 | On policy | 1,536 | 719 | 128 | 512 | 512 | [32,1] | [1,1,16,8,n/a] | 42 | 78.5 |
| GRPO | DeepSeek V3 | On policy | 1,536 | 728 | 256 | 512 | 512 | [32,1] | [1,1,32,8,n/a] | 23.6 | 70.7 |
| GRPO | DeepSeek V3 | 1-step Off | 1,536 | 719 | 256 | 512 | 512 | [16,1] | [1,1,16,8,n/a] | 47.9 | 35 |
| GRPO | Qwen3-235B-A22B | On policy | 8,192 | 5,707 | 64 | 512 | 512 | [8,1] | [2,2,16,4,n/a] | 174 | 268 |
| GRPO | Qwen3-235B-A22B | On policy | 8,192 | 5,711 | 128 | 512 | 512 | [8,1] | [2,2,16,4,n/a] | 111 | 210 |
| GRPO | Qwen3-235B-A22B | 1-step Off | 8,192 | 5,699 | 128 | 512 | 512 | [8,1] | [4,1,16,4,n/a] | 146 | 160 |
| GRPO | Qwen3-30B-A3B | On policy | 4,096 | 3,198 | 16 | 2,048 | 2,048 | [1,1] | [1,1,16,1,n/a] | 2,265 | 186 |
| GRPO | Qwen3-30B-A3B | 1-step Off | 4,096 | 3,200 | 16 | 2,048 | 2,048 | [1,1] | [1,1,8,1,n/a] | 1,748 | 242 |
| GRPO | Qwen3-32B | 1-step Off | 4,096 | 3,255 | 32 | 2,048 | 2,048 | [1,1] | [2,1,1,1,n/a] | 1,539 | 140 |
| GRPO | Qwen3-32B | On policy | 4,096 | 3,254 | 16 | 2,048 | 2,048 | [2,1] | [2,1,1,4,n/a] | 1,458 | 295 |
| GRPO | Nemotron-3-Super-120B-A12B | On policy | 8,192 | 3,219 | 128 | 256 | 256 | [4,1] | [2,1,16,1,n/a] | 56.5 | 119 |
| GRPO | Nemotron-3-Super-120B-A12B | 1-step Off | 8,192 | 3,177 | 128 | 256 | 256 | [4,1] | [2,1,16,1,n/a] | 92.2 | 76 |

Note:

* All Mixture-of-expert (MoE) model training uses token drop-less. 
* The following metrics are extracted from the average of 5 steps: G-Average Seq len, Tokens/sec/gpu, Total Step time(s). Because of the averaging, the numbers in the table do not completely match the equation stated in Performance Metrics above but the difference is small.

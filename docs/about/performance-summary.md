
# Performance

As part of the NVIDIA NeMo Framework, NeMo RL provides optimal performance for reinforcement learning on generative AI models by incorporating the latest optimizations - such as refit optimizations, mixed-precision training, and off-policy training.

This page provides performance benchmarks for LLMs and VLMs using NeMo RL across different GPU systems and configurations. The recipes to reproduce these runs, in yaml file form, can be found under [this folder](https://github.com/NVIDIA-NeMo/RL/tree/r0.8.0/examples/configs/recipes/llm/performance).

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
- **Training backend**: NeMo RL has two training backends: Megatron and Automodel. This performance summary currently only shows numbers from the Megatron backend.

## Performance Metrics

Since reinforcement learning consists of training, generation and transition between the two, performance measurement also reflects this. Specifically, we track the following metrics:
- **Step time**: Time for each step, which includes training, generation, policy logprobs, and refit time.
- **Tokens/sec/GPU**: The rate at which the tokens are processed by a stage (such as training, generation, or refitting) on a single GPU:

    $$
    \text{Tokens/sec/GPU} = \frac{\text{Total Tokens Processed}}{\text{Time for Stage} \times \text{Number of GPUs}}
    $$

- **Training MFU**: Model floating-point operations per second per GPU


## Performance Summary for Large Language Models

Below are performance benchmarks for various large language models organized by release version. These results were obtained using performance recipes available [here](https://github.com/NVIDIA-NeMo/RL/tree/r0.8.0/examples/configs/recipes/llm/performance).

The performance data includes:

- **RL Performance**: Performance metrics for various model sizes and architectures on different RL algorithms (GRPO and in the future DAPO, PPO, for both on-policy and asynchronous).
- **System Configurations**: Results across different GPU systems (DGX-H100 and GB200-NVL72)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8, MXFP8)

---

## NeMo RL v0.8

### H100 BF16 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2); DAPO dataset: [DAPOMath17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
* System: DGX-H100
* Precision: Training BF16, Generation BF16
* Training Backend: Megatron Core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO | DeepSeek V3 | On policy | 1,536 | 712 | 256 | 512 | 512 | [32,1] | [1,1,16,16,n/a] | 21.6 | 75.6 |
| GRPO | DeepSeek V3 | On policy | 1,536 | 719 | 512 | 512 | 512 | [32,1] | [1,1,32,16,n/a] | 11.6 | 71.3 |
| GRPO | DeepSeek V3 | 1-step Off | 1,536 | 728 | 512 | 512 | 512 | [32,1] | [1,1,16,16,n/a] | 39.9 | 24 |
| GRPO | Qwen3-235B-A22B | On policy | 8,192 | 5,699 | 128 | 512 | 512 | [16,1] | [2,2,16,8,n/a] | 76.9 | 303 |
| GRPO | Qwen3-235B-A22B | On policy | 8,192 | 5,699 | 256 | 512 | 512 | [16,1] | [2,2,16,8,n/a] | 48.6 | 240 |
| GRPO | Qwen3-235B-A22B | 1-step Off | 8,192 | 5,699 | 256 | 512 | 512 | [8,1] | [4,1,16,8,n/a] | 171 | 116 |
| GRPO | Qwen3-30B-A3B | On policy | 4,096 | 3,199 | 32 | 2,048 | 2,048 | [2,1] | [1,1,8,1,n/a] | 1,532 | 138 |
| GRPO | Qwen3-30B-A3B | 1-step Off | 4,096 | 3,199 | 32 | 2,048 | 2,048 | [2,1] | [1,1,8,2,n/a] | 1,917 | 111 |
| GRPO | Qwen3-30B-A3B | 8-step Off | 4,096 | 3,198 | 192 | 2,048 | 512 | [2,1] | [1,1,8,1,n/a] | 1,064 | 33.1 |
| GRPO | Qwen3-32B | 1-step Off | 4,096 | 3,253 | 64 | 2,048 | 2,048 | [4,1] | [4,1,1,4,n/a] | 715 | 150 |
| GRPO | Qwen3-32B | On policy | 4,096 | 3,250 | 32 | 2,048 | 2,048 | [4,1] | [4,1,1,4,n/a] | 746 | 288 |
| GRPO | Qwen3-30B-A3B | On policy | 40,960 | 7,930 | 32 | 2,048 | 512 | [2,1] | [4,8,8,1,n/a] | 328 | 1,572 |
| GRPO | Nemotron-3-Super-120B-A12B | On policy | 8,192 | 3,161 | 256 | 256 | 256 | [8,1] | [4,1,32,1,n/a] | 44.4 | 74 |
| GRPO | Nemotron-3-Super-120B-A12B | 1-step Off | 8,192 | 3,253 | 256 | 256 | 256 | [8,1] | [4,1,32,1,n/a] | 116 | 42.6 |

### H100 FP8 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
* System: DGX-H100
* Precision: Generation FP8, Training FP8
* Training Backend: Megatron Core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO | DeepSeek V3 | 1-step Off | 1,536 | 739 | 512 | 512 | 512 | [16,1] | [1,1,16,16,n/a] | 11.8 | 76.1 |

### GB200 BF16 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
* System: GB200-NVL72
* Precision: Training BF16, Generation BF16
* Training Backend: Megatron Core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO | DeepSeek V3 | On policy | 1,536 | 712 | 128 | 512 | 512 | [32,1] | [1,1,16,8,n/a] | 50.5 | 64.7 |
| GRPO | DeepSeek V3 | On policy | 1,536 | 711 | 256 | 512 | 512 | [32,1] | [1,1,32,8,n/a] | 28 | 58.3 |
| GRPO | DeepSeek V3 | 1-step Off | 1,536 | 721 | 256 | 512 | 512 | [16,1] | [1,1,16,8,n/a] | 107 | 15.5 |
| GRPO | Qwen3-235B-A22B | On policy | 8,192 | 5,707 | 64 | 512 | 512 | [8,1] | [2,2,16,4,n/a] | 174 | 268 |
| GRPO | Qwen3-235B-A22B | On policy | 8,192 | 5,729 | 128 | 512 | 512 | [4,2] | [2,2,16,4,n/a] | 130 | 180 |
| GRPO | Qwen3-235B-A22B | 1-step Off | 8,192 | 5,706 | 128 | 512 | 512 | [8,1] | [4,1,16,4,n/a] | 277 | 98.2 |
| GRPO | Qwen3-30B-A3B | On policy | 4,096 | 3,201 | 16 | 2,048 | 2,048 | [1,1] | [1,1,16,1,n/a] | 2,656 | 159 |
| GRPO | Qwen3-30B-A3B | 1-step Off | 4,096 | 3,200 | 16 | 2,048 | 2,048 | [1,1] | [1,1,8,1,n/a] | 2,031 | 208 |
| GRPO | Qwen3-32B | 1-step Off | 4,096 | 3,261 | 32 | 2,048 | 2,048 | [1,1] | [2,1,1,1,n/a] | 1,528 | 141 |
| GRPO | Qwen3-32B | On policy | 4,096 | 3,257 | 16 | 2,048 | 2,048 | [2,1] | [2,1,1,4,n/a] | 1,589 | 271 |
| GRPO | Nemotron-3-Super-120B-A12B | On policy | 8,192 | 3,186 | 128 | 256 | 256 | [4,1] | [2,1,16,1,n/a] | 74.8 | 96.4 |
| GRPO | Nemotron-3-Super-120B-A12B | 1-step Off | 8,192 | 3,232 | 128 | 256 | 256 | [4,1] | [2,1,16,1,n/a] | 202 | 42.8 |

### GB200 MXFP8 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
* System: GB200-NVL72
* Precision: Training BF16, Generation MXFP8
* Training Backend: Megatron Core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO | DeepSeek V3 | 1-step Off | 1,536 | 730 | 256 | 512 | 512 | [16,1] | [1,1,16,8,n/a] | 111 | 15.1 |
| GRPO | Qwen3-235B-A22B | On policy | 8,192 | 5,695 | 64 | 512 | 512 | [4,2] | [2,2,16,4,n/a] | 122 | 393 |
| GRPO | Qwen3-235B-A22B | 1-step Off | 8,192 | 5,688 | 128 | 512 | 512 | [4,1] | [4,1,16,4,n/a] | 282 | 89.3 |
| GRPO | Qwen3-30B-A3B | 1-step Off | 4,096 | 3,197 | 16 | 2,048 | 2,048 | [1,1] | [1,1,8,1,n/a] | 1,819 | 232 |
| GRPO | Qwen3-32B | 1-step Off | 4,096 | 3,244 | 32 | 2,048 | 2,048 | [1,1] | [2,1,1,1,n/a] | 1,506 | 142 |
| GRPO | Qwen3-32B | On policy | 4,096 | 3,239 | 16 | 2,048 | 2,048 | [2,1] | [2,1,1,4,n/a] | 1,677 | 255 |
| GRPO | Nemotron-3-Super-120B-A12B | On policy | 8,192 | 3,155 | 128 | 256 | 256 | [4,1] | [2,1,16,1,n/a] | 70.1 | 101 |
| GRPO | Nemotron-3-Super-120B-A12B | 1-step Off | 8,192 | 3,201 | 128 | 256 | 256 | [4,1] | [2,1,16,1,n/a] | 255 | 32.9 |

Note:

* All Mixture-of-expert (MoE) model training uses token drop-less. 
* The following metrics are extracted from the average of steps 3-8: G-Average Seq len, Tokens/sec/gpu, Total Step time(s). Because of the averaging, the numbers in the table do not completely match the equation stated in Performance Metrics above but the difference is small.
* The GB200 BF16 Qwen3-235B-A22B on-policy row with 64 GPUs reports v0.7 numbers and its v0.7 generation config (`[8,1]`). The v0.8 recipe uses `[4,2]`; its v0.8 run did not complete.

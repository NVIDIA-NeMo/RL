
# Performance

As part of the NVIDIA NeMo Framework, NeMo RL, provides optimal performance for reinforcement learning on generative AI models by incorporating the latest optimizations - such as refit optimizations, mixed-precision training, and off-policy training.

This page provides performance benchmarks for LLMs and VLMs using NeMo RL across different GPU systems and configurations.

## Nomenclature

- **GBS**: Global Batch Size
- **MBS**: Micro Batch Size
- **TP**: Tensor Parallel Size
- **PP**: Pipeline Parallel Size
- **CP**: Context Parallel Size
- **VP**: Virtual Pipeline Parallel Size
- **EP**: Expert Parallel Size
- **T-**: Training related
- **G-**: Generation related
- **Training backend**: NeMo RL have two training backends: Megatron and PyTorch DTensor. This performance summary currently only shows number from Megatron backend.

## Performance Metrics

Since reinforcement learning consists of training, generation and transition between the two, performance measurement also reflects this. Specifically, we track the following metrics:
- **Step time**: Time for each step, which includes training, generation, policy logprobs, and refit time.
- **Tokens/sec/GPU**: The rate at the tokens are processed by a stage (such as training, generation, or refitting) on a single GPU:

    $$
    \text{Tokens/sec/GPU} = \frac{\text{Total Tokens Processed}}{\text{Time for Stage} \times \text{Number of GPUs}}
    $$

- **Training MFU**: Model floating-point operations per second per GPU


## Performance Summary for Large Language Models

Below are performance benchmarks for various large language models organized by release version. These results were obtained using performance recipes available [here](https://github.com/NVIDIA-NeMo/RL/tree/r0.4.0/examples/configs/recipes/llm/performance).

The performance data includes:

- **RL Performance**: Performance metrics for various model sizes and architectures on different RL algorithms (GRPO and in the future DAPO, PPO, for both on-policy and asynchronous).
- **System Configurations**: Results across different GPU systems (DGX-H100 and in the future DGX-GB200, DGX-B200)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8)

---

## Nemo RL v0.4

* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
* System: DGX-H100
* Precision: Training BF16, Generation BF16
* Training Backend: Megatron-core.

| Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
|LLAMA3.1_8B|On policy    |4,096                |1,060            |16    |2,048|512  |[1,1]             |[1,1,1,1,1,2,n/a]         |1,562             | 97.7|
|LLAMA3.1_8B|1-step Off   |4,096                |1,129            |16    |2,048|512  |[1,1]             |[1,1,1,1,1,2,n/a]         |2,161             | 74.6|
|DeepSeek V3|On policy    |1,536                |745              |256   |512  |512  |[32,1]            |[1,1,16,16,n/a]           |11                | 154|
|DeepSeek V3|1-step Off   |1,536                |744              |512   |512  |512  |[32,1]            |[1,1,16,16,n/a]           |11.0              | 77.9|
|Qwen3-235B |On policy    |8,192                |5,671            |128   |512  |512  |[16,1]            |[2,2,16,8,n/a]            |45.7              | 506|
|Qwen3-235B |1-step Off   |8,192                |5,691            |256   |512  |512  |[8,1]             |[4,1,16,8,n/a]            |52.2              | 241|
|Qwen3-30B3A|On policy    |4,096                |3,154            |32    |2,048|512  |[4,1]             |[2,1,8,1,n/a]             |925               | 225|
|Qwen3-30B3A|1-step Off   |4,096                |3,158            |32    |2,048|512  |[4,1]             |[2,1,8,1,n/a]             |864               | 244|
|Qwen3-32B  |On policy    |4,096                |3,206            |32    |2,048|512  |[4,1]             |[4,1,1,4,n/a]             |540               | 393|
|Qwen3-32B  |1-step Off   |4,096                |3,207            |64    |2,048|512  |[4,1]             |[4,1,1,4,n/a]             |494               | 215|


Note:

* All Mixture-of-expert (MoE) model training uses token drop-less. 
* The following metrics are extracted from the average of 5 steps: G-Average Seq len, Tokens/sec/gpu, Total Step time(s). Because of the averaging, the numbers in table does not completely match the equation stated in Performance Metrics above but the difference is small.
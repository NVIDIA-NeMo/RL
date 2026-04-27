
# Performance

As part of the NVIDIA NeMo Framework, NeMo RL provides optimal performance for reinforcement learning on generative AI models by incorporating the latest optimizations - such as refit optimizations, mixed-precision training, and off-policy training.

This page provides performance benchmarks for LLMs and VLMs using NeMo RL across different GPU systems and configurations. The recipes to reproduce these runs, in yaml file form, can be found under [this folder](https://github.com/NVIDIA-NeMo/RL/tree/r0.6.0/examples/configs/recipes/llm/performance).

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

Below are performance benchmarks for various large language models organized by release version. These results were obtained using performance recipes available [here](https://github.com/NVIDIA-NeMo/RL/tree/r0.4.0/examples/configs/recipes/llm/performance).

The performance data includes:

- **RL Performance**: Performance metrics for various model sizes and architectures on different RL algorithms (GRPO and in the future DAPO, PPO, for both on-policy and asynchronous).
- **System Configurations**: Results across different GPU systems (DGX-H100 and in the future DGX-GB200, DGX-B200)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8)

---

## Nemo RL v0.6

### H100 BF16 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2); DAPO dataset: [DAPOMath17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k); SWE dataset: refer to [Nemotron super-v3 documentation - stage 2.2](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/docs/guides/nemotron-3-super.md#stage-22---swe-2-64-nodes)
* System: DGX-H100
* Precision: Training BF16, Generation BF16
* Training Backend: Megatron-core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO      |DeepSeek V3|On policy    |1,536                |706              |256   |512  |512  |[32,1]            |[1,1,16,16,n/a]           |12.1              | 135|
| GRPO      |DeepSeek V3|On policy    |1,536                |706              |512   |512  |512  |[32,1]            |[1,1,16,16,n/a]           |7.0              | 116|
| GRPO      |DeepSeek V3|1-step Off   |1,536                |705              |512   |512  |512  |[32,1]            |[1,1,16,16,n/a]           |12.0              | 68.4|
| GRPO      |Qwen3-235B |On policy    |8,192                |5,729            |128   |512  |512  |[16,1]            |[2,2,16,8,n/a]            |54.6              | 429|
| GRPO      |Qwen3-235B |On policy    |8,192                |5,718            |256   |512  |512  |[16,1]            |[2,2,16,8,n/a]            |35.1              | 333|
| GRPO      |Qwen3-235B |1-step Off   |8,192                |5,692            |256   |512  |512  |[8,1]             |[4,1,16,8,n/a]            |59.4              | 218|
| GRPO      |Qwen3-30B3A|On policy    |4,096                |3,197            |32    |2,048|512  |[2,1]             |[1,1,8,1,n/a]             |1088               | 194|
| GRPO      |Qwen3-30B3A|1-step Off   |4,096                |3,201            |32    |2,048|512  |[2,1]             |[1,1,8,2,n/a]             |1443               | 150|
| GRPO      |Qwen3-30B3A|8-step Off   |4,096                |3,203            |192   |2,048|512  |[2,1]             |[1,1,8,1,n/a]             |1036               | 34.0|

### H100 FP8 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
* System: DGX-H100
* Precision: Generation FP8, Training FP8
* Training Backend: Megatron-core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO      |DeepSeek V3|1-step Off   |1,536                |730              |512   |512  |512  |[16,1]            |[1,1,16,16,n/a]           |14.6              | 57.48|

### GB200 BF16 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
* System: GB200-NVL72
* Precision: Training BF16, Generation BF16
* Training Backend: Megatron-core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO      |DeepSeek V3|On policy    |1,536                |711              |128   |512  |512  |[32,1]            |[1,1,16,8,n/a]            |30.2              | 108|
| GRPO      |DeepSeek V3|On policy    |1,536                |674              |256   |512  |512  |[32,1]            |[1,1,16,8,n/a]            |17.4              | 92.2|
| GRPO      |DeepSeek V3|1-step Off   |1,536                |708              |256   |512  |512  |[16,1]            |[1,1,16,8,n/a]            |26.7              | 61.7|
| GRPO      |Qwen3-235B |On policy    |8,192                |5,688            |64    |512  |512  |[8,1]            |[2,2,16,4,n/a]            |164              | 284|
| GRPO      |Qwen3-235B |On policy    |8,192                |5,700            |128   |512  |512  |[8,1]            |[2,2,16,4,n/a]            |69.2              | 337|
| GRPO      |Qwen3-235B |1-step Off   |8,192                |5,719            |128   |512  |512  |[8,1]             |[4,1,16,4,n/a]            |85.8              | 277|
| GRPO      |Qwen3-30B3A|On policy    |4,096                |3,199            |16    |2,048|512  |[1,1]             |[1,1,16,1,n/a]             |1,934               | 219|
| GRPO      |Qwen3-30B3A|1-step Off   |4,096                |3,202            |16    |2,048|512  |[1,1]             |[1,1,16,1,n/a]             |1,415               | 299|
| SWE       |Nemotron-3-Nano-30B-A3B|1-step Off   |131,072  |31,599           |128   |512  |512  |[8,1]             |[8,8,8,1,n/a]             |37.5               | 430|

Note:

* All Mixture-of-expert (MoE) model training uses token drop-less. 
* The following metrics are extracted from the average of 5 steps: G-Average Seq len, Tokens/sec/gpu, Total Step time(s). Because of the averaging, the numbers in the table do not completely match the equation stated in Performance Metrics above but the difference is small.


# Performance

As part of the NVIDIA NeMo Framework, NeMo RL, provides optimal performance for reinforcement learning on generative AI models by incorporating the latest optimizations - such as refit optimizations, mixed-precision training, and off-policy training.

This page provides performance benchmarks for LLMs and VLMs using NeMo RL across different GPU systems and configurations. The recipes to reproduce these runs, in yaml file form, can be found under [this folder](https://github.com/NVIDIA-NeMo/RL/tree/r0.5.0/examples/configs/recipes/llm/performance).

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

## Nemo RL v0.5

### H100 BF16 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2); DAPO dataset: [DAPOMath17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
* System: DGX-H100
* Precision: Training BF16, Generation BF16
* Training Backend: Megatron-core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO      |LLAMA3.1_8B|On policy    |4,096                |1,019            |16    |2,048|512  |[1,1]             |[1,1,1,1,1,2,n/a]         |1,581             | 92.8|
| GRPO      |LLAMA3.1_8B|1-step Off   |4,096                |1,123            |16    |2,048|512  |[1,1]             |[1,1,1,1,1,1,n/a]         |2,478             | 64.8|
| GRPO      |DeepSeek V3|On policy    |1,536                |744              |256   |512  |512  |[32,1]            |[1,1,16,16,n/a]           |12.7              | 134|
| GRPO      |DeepSeek V3|1-step Off   |1,536                |738              |512   |512  |512  |[32,1]            |[1,1,16,16,n/a]           |13.1              | 64.9|
| DAPO      |DeepSeek V3|On policy    |1,536                |974              |512   |512  |512  |[64,1]            |[8,4,32,8,n/a]            |2.45              | 974|
| GRPO      |Qwen3-235B |On policy    |8,192                |5,700            |128   |512  |512  |[16,1]            |[2,2,16,8,n/a]            |54.1              | 431|
| GRPO      |Qwen3-235B |1-step Off   |8,192                |5,707            |256   |512  |512  |[8,1]             |[4,1,16,8,n/a]            |58.7              | 203|
| GRPO      |Qwen3-30B3A|On policy    |4,096                |3,196            |32    |2,048|512  |[2,1]             |[1,1,8,1,n/a]             |1066               | 198|
| GRPO      |Qwen3-30B3A|1-step Off   |4,096                |3,201            |32    |2,048|512  |[2,1]             |[1,1,8,2,n/a]             |1391               | 154|
| GRPO      |Qwen3-32B  |On policy    |4,096                |3,251            |32    |2,048|512  |[4,1]             |[4,1,1,4,n/a]             |571               | 376|
| GRPO      |Qwen3-32B  |1-step Off   |4,096                |3,252            |64    |2,048|512  |[4,1]             |[4,1,1,4,n/a]             |538               | 200|

### H100 FP8 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
* System: DGX-H100
* Precision: Generation FP8, Training FP8
* Training Backend: Megatron-core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO      |LLAMA3.1_8B|1-step Off   |4,096                |1,128            |16    |2,048|512  |[1,1]             |[1,1,1,1,1,1,n/a]         |3,052             | 53.0|
| GRPO      |DeepSeek V3|1-step Off   |1,536                |761              |512   |512  |512  |[16,1]            |[1,1,16,16,n/a]           |14.1              | 67.6|

### GB200 BF16 Benchmarks
* GRPO Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
* System: GB200-NVL72
* Precision: Training BF16, Generation BF16
* Training Backend: Megatron-core.

| Algorithm | Model     |On/Off policy|T-Max Sequence Length|G-Average Seq len|#-GPUs|G-GBS|T-GBS|Generation [TP,PP]|Training [TP,CP,EP,PP,VPP]|Tokens / sec / GPU|Total Step time(s)|
|---------  |-------    |--------     |-----                |-----            |------|---- |---- |----              |----                      |---               |---|
| GRPO      |LLAMA3.1_8B|On policy    |4,096                |1,066            |8     |2,048|512  |[1,1]             |[1,1,1,1,1,1,n/a]         |3,359             | 91.0|
| GRPO      |LLAMA3.1_8B|1-step Off   |4,096                |1,107            |8     |2,048|512  |[1,1]             |[1,1,1,1,1,1,n/a]         |4,463             | 71.1|
| GRPO      |DeepSeek V3|On policy    |1,536                |996              |128   |512  |512  |[32,1]            |[1,1,16,8,n/a]            |34.3              | 128|
| GRPO      |DeepSeek V3|1-step Off   |1,536                |994              |256   |512  |512  |[16,1]            |[1,1,16,8,n/a]            |31.7              | 64.5|
| GRPO      |Qwen3-235B |On policy    |8,192                |5,711            |64    |512  |512  |[8,1]            |[2,2,16,4,n/a]            |140              | 332|
| GRPO      |Qwen3-235B |1-step Off   |8,192                |5,711            |128   |512  |512  |[8,1]             |[4,1,16,4,n/a]            |87.9              | 268|
| GRPO      |Qwen3-30B3A|On policy    |4,096                |3,198            |16    |2,048|512  |[1,1]             |[1,1,16,1,n/a]             |1,822               | 232|
| GRPO      |Qwen3-30B3A|1-step Off   |4,096                |3,204            |32    |2,048|512  |[1,1]             |[1,1,16,1,n/a]             |1,558               | 136|
| GRPO      |Qwen3-32B  |On policy    |4,096                |3,253            |16    |2,048|512  |[1,1]             |[2,1,1,1,n/a]             |1,127              | 381|
| GRPO      |Qwen3-32B  |1-step Off   |4,096                |3,258            |32    |2,048|512  |[1,1]             |[2,1,1,1,n/a]             |1,025               | 210|

Note:

* All Mixture-of-expert (MoE) model training uses token drop-less. 
* The following metrics are extracted from the average of 5 steps: G-Average Seq len, Tokens/sec/gpu, Total Step time(s). Because of the averaging, the numbers in table does not completely match the equation stated in Performance Metrics above but the difference is small.

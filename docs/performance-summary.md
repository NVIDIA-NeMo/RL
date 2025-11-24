# Performance

As part of the NVIDIA NeMo Framework, NeMo RL, provides optimal performance for reinforcement learning on generative AI models by incorporating the latest optimizations - such as refit optimizations, mixed-precision training, and off-policy training.

This page provides performance benchmarks for LLMs and VLMs using NeMo RL across different GPU systems and configurations.

## Nomenclature

- **GBS**: Global Batch Size
- **MBS**: Micro Batch Size
- **FSDP**: Fully Sharded Data Parallel
  - FSDP = 1: use FSDP
  - FSDP = 0: use DDP (Distributed Data Parallel)
- **TP**: Tensor Parallel Size
- **PP**: Pipeline Parallel Size
- **CP**: Context Parallel Size
- **VP**: Virtual Pipeline Parallel Size
- **EP**: Expert Parallel Size
- **GA**: Number of Gradient Accumulations
- **T-**: Training related
- **G-**: Generation related
- **Training backend**: NeMo RL have two training backends: Megatron and PyTorch DTensor

## Performance Metrics

Since reinforcement learning consists of training, generation and transition between the two, performance measurement also reflects this. Specifically, we track the following metrics:
- **Step time**: Time for each step, which includes training, generation, policy logprobs, and refit time.
- **Tokens/sec/GPU**: [insert formula here]
- **Training MFU**: Model floating-point operations per second per GPU

```{contents}
:local:
:depth: 2
```

## Performance Summary for Large Language Models

Below are performance benchmarks for various large language models organized by release version. These results were obtained using performance recipes available [insert script link here].

The performance data includes:

- **RL Performance**: Performance metrics for various model sizes and architectures on different RL algorithms (GRPO (on-policy and ene-step off), and in the future DAPO, PPO)
- **System Configurations**: Results across different GPU systems (DGX-H100 and in the future DGX-GB200, DGX-B200)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8)

---

## Nemo RL v0.4

### GRPO Performance on-policy. Dataset: [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)

#### System: DGX-H100. Precision: Training BF16, Generation BF16.

| Model | Training Backend|T-Max Sequence Length | G-Average Seq len| #-GPUs | G-GBS |T-GBS|G-TP|G-PP|G-DP|T-TP|T-CP| T-EP| T-PP|T-VPP|T-DP | Tokens / sec / GPU | Total Step time(s) |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|------|----|-----|-|
| LLAMA3.1_8B_instruct |Megatron|4096 | 1056 | 16|2048|512| 1 | 1 | 1 | 1 | 1|1|2|n/a | 8| 1496| 99 |
| LLAMA3.1_8B_instruct |PyTorch|4096 | 1056 | 16|2048|512| 1 | 1 | 16| 1 | 1|1|1|n/a | 16| 1482| 114.9 |

will add more rows
- Is MOE token drop or dropless? [Guyue to confirm]

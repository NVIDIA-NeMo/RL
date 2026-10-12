# Evaluation

Use [NeMo Gym](https://docs.nvidia.com/nemo/gym/) to evaluate models and agents with repeatable benchmarks or [your own tasks](https://docs.nvidia.com/nemo/gym/data). The same NeMo Gym environments can produce rollouts and rewards for RL training.

NeMo Gym packages datasets, [agent harnesses](https://docs.nvidia.com/nemo/gym/agent-server), tools, [verifiers](https://docs.nvidia.com/nemo/gym/build-verifiers), rollout collection, and metrics into reusable environments. It supports single-turn and multi-turn evaluations against hosted endpoints, self-hosted model servers, and local checkpoints.

## Choose a Benchmark

List the currently available benchmarks, then inspect one to see its dataset, agent harness, and repeat count:

```bash
gym list benchmarks
gym list benchmarks aime24
```

## Run a Benchmark

The following example evaluates a model on AIME24.

This example requires:

- A Linux system with an NVIDIA GPU supported by vLLM.
- [NeMo Gym installed](https://docs.nvidia.com/nemo/gym/get-started/installation).
- [vLLM](https://docs.nvidia.com/nemo/gym/model-server/vllm) installed in the model-serving environment.
- Network access to download the public model and AIME dataset from Hugging Face.

To evaluate a NeMo RL checkpoint saved in DCP or Megatron format, first [convert it to Hugging Face format](../design-docs/checkpointing.md). The example below uses a public Hugging Face model directly.

Start a vLLM server with the model or converted checkpoint you want to evaluate. This example uses Qwen2.5-Math-1.5B-Instruct:

```bash
vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 10240
```

From another terminal, prepare the benchmark data. This one-time step downloads and converts AIME24 into the format expected by NeMo Gym:

```bash
gym eval prepare --benchmark aime24
```

Run the benchmark against the vLLM endpoint:

```bash
gym eval run \
    --benchmark aime24 \
    --model-type vllm_model \
    --model-url http://127.0.0.1:10240/v1 \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --model-api-key dummy_key \
    --output results/aime24.jsonl
```

The command prints key metrics and writes:

- `results/aime24.jsonl`: rollout records and rewards.
- `results/aime24_materialized_inputs.jsonl`: the rendered benchmark inputs.
- `results/aime24_aggregate_metrics.json`: [aggregate evaluation metrics](https://docs.nvidia.com/nemo/gym/evaluation/aggregate-metrics).

## Analyze and Compare

- [Compare baseline and candidate evaluation runs](https://docs.nvidia.com/nemo/gym/reference/cli-commands#gym-eval-compare).
- [Investigate task-level failures and score changes](https://docs.nvidia.com/nemo/gym/evaluation/diagnose-results).


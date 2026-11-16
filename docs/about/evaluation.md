# Evaluation

Use [NeMo Gym](https://docs.nvidia.com/nemo/gym/) to evaluate models and agents with repeatable benchmarks or [your own tasks](https://docs.nvidia.com/nemo/gym/data). The same NeMo Gym environments can produce rollouts and rewards for RL training.

NeMo Gym packages datasets, [agent harnesses](https://docs.nvidia.com/nemo/gym/agent-server), tools, [verifiers](https://docs.nvidia.com/nemo/gym/build-verifiers), rollout collection, and metrics into reusable environments. It supports single-turn and multi-turn evaluations against hosted endpoints, self-hosted model servers, and local checkpoints.

## Choose a Benchmark

List the currently available benchmarks, then inspect one to see its dataset, agent harness, and repeat count:

```bash
uv run --extra nemo_gym gym list benchmarks
uv run --extra nemo_gym gym list benchmarks aime24
```

## Run a Benchmark

The following example evaluates a model on AIME24.

This example requires:

- A Linux system with an NVIDIA GPU supported by vLLM.
- A NeMo RL checkout with submodules, set up as in the [installation guide](installation.md). The commands below run through `uv run`: `--extra nemo_gym` provides the `gym` command and `--extra vllm` provides `vllm serve`.
- Network access to download the public model and AIME dataset from Hugging Face.

To evaluate a NeMo RL checkpoint that is not in Hugging Face format, convert it first:

- [Automodel DCP checkpoints](../design-docs/checkpointing.md#converting-torch-distributed-checkpoints-to-hugging-face-format)
- [Megatron checkpoints](../design-docs/checkpointing.md#converting-megatron-checkpoints-to-hugging-face-format)
- [Megatron LoRA adapter checkpoints](../design-docs/checkpointing.md#converting-megatron-lora-adapter-checkpoints-to-hugging-face-format)

The example below uses a public Hugging Face model directly.

Start a vLLM server with the model or converted checkpoint you want to evaluate. This example uses Qwen2.5-Math-1.5B-Instruct:

```bash
uv run --extra vllm vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 10240
```

From another terminal, prepare the benchmark data. This one-time step downloads and converts AIME24 into the format expected by NeMo Gym:

```bash
uv run --extra nemo_gym gym eval prepare --benchmark aime24
```

Run the benchmark against the vLLM endpoint:

```bash
uv run --extra nemo_gym gym eval run \
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

## Run a Quick Evaluation with NeMo RL

For quick checks without NeMo Gym, use NeMo RL's [built-in evaluator](../../examples/run_eval.py) to evaluate a Hugging Face-format checkpoint against its supported benchmarks:

```sh
uv run python examples/run_eval.py generation.model_name=/path/to/hugging-face-checkpoint
```

- When you evaluate a checkpoint you trained, use the same prompt and chat template as in training: set `data.prompt_file`, `data.system_prompt_file`, and `tokenizer.chat_template` to your training values. The defaults in `examples/configs/evals/eval.yaml` use the model's native chat template with no prompt file, which suits public models.
- Supported `data.dataset_name` values: `AIME2024`, `AIME2025`, `AIME2026`, `gpqa`, `gpqa_diamond`, `math`, `math500`, `mmlu`, `mmlu_{language}` (one of `AR-XY`, `BN-BD`, `DE-DE`, `ES-LA`, `FR-FR`, `HI-IN`, `ID-ID`, `IT-IT`, `JA-JP`, `KO-KR`, `PT-BR`, `ZH-CN`, `SW-KE`, `YO-NG`), `mmlu_pro`, `mmau`, and `daily-omni`. Ready-made configs are in `examples/configs/evals/`.


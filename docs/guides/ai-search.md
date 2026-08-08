# Train an AI Search Agent with GRPO

The AI-search example trains Qwen2.5-7B-Instruct to call a private search tool
while answering questions. It combines three pieces:

- **NeMo RL** runs vLLM rollout generation and the GRPO update.
- **NeMo Gym** controls the multi-turn question/search/answer interaction and
  calculates the reward.
- **NVIDIA cuVS** searches a local dense-vector index on the GPU.

The model, not the training script, writes each search query. A rollout may
search repeatedly, and only model-generated tokens are trained.

## Run the one-step example

Start from a recursive NeMo RL checkout on a CUDA 13-capable GPU with `uv`
available:

```bash
bash examples/nemo_gym/ai_search/prepare_ai_search.sh
bash examples/nemo_gym/ai_search/run_ai_search.sh
```

The full-parameter recipe targets a B300-class GPU. The launcher defaults to
one real optimizer step with two questions and four rollouts per question.
Artifacts are written under `/tmp/nemo-rl-ai-search/runs` so model checkpoints
do not accidentally fill a shared source filesystem. CPU offload alone is not
a supported small-GPU substitute because it can require hundreds of GiB of
host memory.

Run the 50-step recipe with:

```bash
AI_SEARCH_MAX_STEPS=50 \
AI_SEARCH_RUN_DIR=/tmp/nemo-rl-ai-search/runs/train-50 \
  bash examples/nemo_gym/ai_search/run_ai_search.sh
```

## Replace the sample data

The example includes only 32 documents and a few synthetic questions. Its
purpose is to validate the complete system, not to establish model quality.

Use the formats shown in:

- `examples/nemo_gym/ai_search/resources_servers/ai_search/data/corpus.jsonl`
- `examples/nemo_gym/ai_search/resources_servers/ai_search/data/train.jsonl`
- `examples/nemo_gym/ai_search/resources_servers/ai_search/data/validation.jsonl`

After replacing the corpus, regenerate the E5 embeddings and cuVS artifact.
The startup-time manifest check rejects an index whose corpus hash or encoder
settings are stale.

## Choose the cuVS index

The default is exact cuVS brute-force retrieval, which is appropriate for a
small corpus. For a larger corpus, change `index.algorithm` to `cagra` and
`index.metric` to `sqeuclidean` in
`resources_servers/ai_search/configs/ai_search.yaml`. Measure recall while
tuning `itopk_size` and `search_width`; approximate-index latency without recall
is not enough to select a configuration.

The Gym-facing contract is `retrieval.types.SearchProvider`. A private
inverted-index adapter can implement the same batched query/results interface,
leaving the agent and GRPO recipe unchanged.

## Measure the target system

```bash
bash examples/nemo_gym/ai_search/profile_ai_search.sh
```

This compares CPU and GPU vector-search implementations, then measures E5
encoding, cuVS search, document fetch, cache hits, and concurrent search
microbatching. See the example's
[performance report](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/nemo_gym/ai_search/PERFORMANCE.md)
for one RTX 6000D run and its limitations, and the
[README](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/nemo_gym/ai_search/README.md)
for the data schema, reward, configuration, and extension points.

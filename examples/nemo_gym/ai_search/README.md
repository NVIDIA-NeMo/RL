# AI-search GRPO with NeMo Gym and NVIDIA cuVS

This example trains a model to answer questions by calling a private search
tool. NeMo Gym owns the question, search session, tool execution, and reward;
NeMo RL owns rollout generation and the GRPO update. The default vector search
backend is NVIDIA cuVS.

The checked-in corpus is intentionally tiny. It is a functional example for
testing the complete path, not a benchmark dataset or a pretrained search
agent.

## What happens in one training step

With the default smoke-test launcher:

1. NeMo RL reads two questions.
2. vLLM asks Qwen2.5-7B-Instruct to answer each question four times, producing
   eight independent trajectories.
3. During a trajectory, the model may call the structured `search` tool. NeMo
   Gym sends the query to the local resources server, which batches nearby
   requests, encodes them with E5, searches cuVS, and returns the top passages.
4. The model sees those passages and may search again. The Gym agent permits up
   to five model turns, while the resources server permits up to four searches.
5. The verifier scores answer token-F1, retrieval recall, output format, and
   search efficiency. Their weighted sum is the scalar GRPO reward.
6. NeMo RL computes group-relative advantages, the policy/KL loss, gradients,
   and one optimizer update.

The model generates the search query. The search engine never sees the original
question unless the model chooses to use it as its query.

## Quick start

Requirements are a CUDA 13-capable NVIDIA GPU, `uv`, and a recursive NeMo RL
checkout. The helper keeps environments, model downloads, and compiler caches
under `/tmp/nemo-rl-ai-search` by default.

```bash
# Build the two local environments and generate passage embeddings.
bash examples/nemo_gym/ai_search/prepare_ai_search.sh

# Run a real one-step GRPO smoke test: 2 questions x 4 trajectories.
bash examples/nemo_gym/ai_search/run_ai_search.sh
```

The recipe performs full-parameter training and targets a B300-class GPU. It is
configured for 50 steps, while the launcher defaults to one step. Override the
launcher when a longer run is wanted:

```bash
AI_SEARCH_MAX_STEPS=50 \
AI_SEARCH_RUN_DIR=/tmp/nemo-rl-ai-search/runs/train-50 \
  bash examples/nemo_gym/ai_search/run_ai_search.sh
```

Useful overrides are:

```bash
UV_BIN=/absolute/path/to/uv
AI_SEARCH_RUNTIME_DIR=/fast/local/disk/nemo-rl-ai-search
AI_SEARCH_NUM_PROMPTS=8
AI_SEARCH_NUM_GENERATIONS=4
AI_SEARCH_FORCE_INSTALL=1
```

Do not assume that CPU offload makes this full-parameter recipe suitable for a
smaller GPU: it moves substantial policy and optimizer pressure into host RAM.
Use the target B300-class GPU for the default recipe. A separate 7B LoRA
diagnostic on an RTX 6000D is recorded in [PERFORMANCE.md](PERFORMANCE.md); it
validates the pipeline but is not equivalent to the full-parameter run.

`prepare_ai_search.sh` uses `uv` throughout. It installs policy, vLLM, and Gym
dependencies into one node-local environment because the example runs every
actor on one GPU and does not require optional MoE kernels.

## Use a private corpus

Replace `resources_servers/ai_search/data/corpus.jsonl`. Each line is one
document:

```json
{"id":"doc-1","title":"Document title","text":"Searchable passage text."}
```

Then delete the old generated files under `data/index/` or rebuild explicitly:

```bash
source examples/nemo_gym/ai_search/prepare_ai_search.sh

PYTHONPATH=examples/nemo_gym/ai_search \
  "$NEMO_GYM_VENV_DIR/resources_servers/ai_search/.venv/bin/python" \
  -m resources_servers.ai_search.prepare_index \
  --config examples/nemo_gym/ai_search/resources_servers/ai_search/configs/ai_search.yaml \
  --force --build-serialized-index
```

The manifest stores the corpus hash and encoder settings. Startup fails instead
of silently using stale embeddings when either changes.

Training and validation rows contain the question, accepted answers, supporting
document IDs, the Gym agent reference, and the Responses API prompt/tool
definition. See `data/train.jsonl` for the complete schema. Replace the toy rows
with a disjoint, realistically sized dataset before drawing quality conclusions.

## Retrieval backends

`retrieval.types.SearchProvider` is the boundary between the Gym server and a
search implementation. It receives a batch of text queries and returns ranked
documents plus timing data. `DenseSearchEngine` is the implementation supplied
here:

```text
query text -> E5 GPU encoder -> cuVS vector index -> document store -> passages
```

Both cuVS exhaustive search and CAGRA are implemented. Select CAGRA for larger
indexes by changing the resource-server config:

```yaml
index:
  kind: cuvs
  algorithm: cagra
  metric: sqeuclidean
```

Embeddings remain normalized, so squared Euclidean distance preserves cosine
ordering. The NumPy index is a CPU correctness baseline, not the production
default.

To connect an inverted index, implement the same `SearchProvider.search_batch`
contract and construct that provider in `AISearchResourcesServer`. The Gym tool,
rollout loop, verifier, and training recipe do not need to change. An inverted
adapter is intentionally not included yet because its query syntax, filtering,
and service client depend on the user's search stack.

## Reward

The scalar reward is:

```text
1.00 * answer token-F1
+ 0.25 * supporting-document recall
+ 0.10 * valid "Final Answer:" format
+ 0.05 * efficient successful search
```

All components and retrieval timings are logged independently. This avoids the
strict-exact-match failure mode where a model can improve its score mostly by
shortening an already-correct answer. For a serious run, monitor semantic answer
quality, retrieval recall, format validity, search count, and duplicate queries
together.

## Correctness checks and profiling

```bash
# Gym discovery/config validation
source examples/nemo_gym/ai_search/prepare_ai_search.sh
gym env validate \
  --config-dir examples/nemo_gym/ai_search \
  --agent ai_search_simple_agent

# Retrieval and server tests
gym env test --resources-server ai_search

# Same-hardware backend and pipeline profiling
bash examples/nemo_gym/ai_search/profile_ai_search.sh
```

The profiler compares NumPy CPU, FAISS CPU, a direct PyTorch CUDA baseline,
cuVS brute force, and cuVS CAGRA. It also measures E5 encoding, index search,
document fetch, query caching, and asynchronous microbatching. See
[PERFORMANCE.md](PERFORMANCE.md) for measured results and
[RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the design comparison.

The performance report separates the historical 1.5B full-parameter baseline
from the current 7B LoRA plumbing check. Neither is a B300 throughput claim for
the default 7B full-parameter recipe.

## Current limits

- The included data has only 32 documents, eight training questions, and four
  validation questions.
- The launcher is single-node and single-GPU; NeMo RL can scale farther, but
  resource placement for a dedicated multi-GPU retriever is not configured here.
- CAGRA tuning is workload-dependent. Measure recall and latency on the real
  corpus instead of copying the sample parameters blindly.
- This is private-corpus search. It does not crawl pages, call a public web-search
  API, or execute untrusted code, so it does not need a code sandbox.

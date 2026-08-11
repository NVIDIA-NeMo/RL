# Aligned AI-search training comparison

This directory defines the controlled workload used to compare NeMo RL,
Search-R1, and Alibaba ZeroSearch. It is a performance experiment, not a model
quality benchmark.

All three runs use Qwen2.5-3B-Instruct, BF16 full-parameter GRPO, four prompts
per optimizer step, four trajectories per prompt, at most three model calls
(two searchable rounds plus one final answer), top-3 retrieval, normalized
exact-match answer reward, temperature 1.0, and the same
32-train/8-validation fixture.
The first step warms caches; steps 2-4 are the measured window. A final
validation and checkpoint are reported separately from core-step throughput.

The fixture contains synthetic two-hop facts. Each question first identifies a
project custodian and then asks for that person's birthplace. Answers and names
are generated identifiers, so the policy must use the supplied evidence rather
than pretrained factual memory. `prepare_aligned_data.py` writes both the NeMo
Gym JSONL view and the nested Parquet schema consumed by the two veRL forks.

All runs call `retriever_server.py`, a deterministic in-memory BM25 service.
It accepts both the Search-R1 batch protocol and ZeroSearch's single-query
protocol. Keeping retrieval and corpus identical isolates differences in the
trainer, model serving, and agent loop. Framework-native action syntax remains
different: NeMo Gym uses structured tool calls; Search-R1 and ZeroSearch use
`<search>`/`<answer>` tags. Reports must therefore include actual token counts
and token-normalized throughput beside wall time.

The measured 4x A100 result and bottleneck analysis are in
[`PERFORMANCE.md`](PERFORMANCE.md). Machine-readable aggregates and per-step
records are in `a100_sxm4_4x_20260809.json`.

Generate the fixture with:

```bash
uv run --with pandas --with pyarrow \
  python examples/nemo_gym/ai_search/comparison/prepare_aligned_data.py
```

Generated Parquet and runtime artifacts remain untracked. Each experiment log
records the exact source commits, commands, hardware, package versions, stage
timings, search request log, and one-second GPU/host memory samples.

The separate full-quality experiment follows Search-R1's official 7B training
setup instead of this small performance fixture. Its exact parity contract,
including the few places where the paper and repository disagree, is recorded
in [`SEARCH_R1_PARITY.md`](SEARCH_R1_PARITY.md).

Download and convert the official question data with:

```bash
bash examples/nemo_gym/ai_search/comparison/download_search_r1_data.sh \
  /path/to/nq_hotpotqa_train

uv run --with pandas --with pyarrow \
  python examples/nemo_gym/ai_search/comparison/prepare_search_r1_data.py \
  --source-dir /path/to/nq_hotpotqa_train
```

The 2018 Wikipedia corpus and E5 index are separate artifacts of roughly 70 GB
combined. Stage them on storage visible to the retrieval service rather than in
this repository.

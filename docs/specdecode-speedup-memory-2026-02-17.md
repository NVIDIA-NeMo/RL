# Spec-Decoding Speedup Handoff Memory (2026-02-17)

This note captures where we paused while setting up a verifier/draft speedup experiment.

## Goal

Measure whether speculative decoding provides net generation speedup on `umbriel-b200-036` for:

- verifier: Qwen3-14B
- draft: Qwen3-1.7B

## Confirmed environment state

- Node GPUs: 8x NVIDIA B200 (183359 MiB each), no active processes at check time.
- Runtime stack check passed:
  - `torch 2.10.0+cu129`
  - `cuda True`
  - `gpu_count 8`
  - `vllm 0.16.0`
- Cached Qwen snapshots present locally:
  - `Qwen3-14B`: `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18`
  - `Qwen3-1.7B`: `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
  - `Qwen3-0.6B`: `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`

## Spec-decode correctness status before this handoff

- Non-zero acceptance already validated for same-model verifier/draft smoke check:
  - `Early Token Acceptance Rate: 1.0000 (55/55)`
  - source: `/home/scratch.shaunakj_other/logs/specdecode-1step-greedy-autoformat-fixed-skipvenv-14b/run.log`

## Where this speedup task paused

- Started setting up baseline vs speculative throughput A/B.
- Attempted to use vLLM bench CLI:
  - non-elevated run hit tempfile resolution issue
  - moved to elevated run with explicit `TMPDIR`, but command was user-interrupted before completion.

## Resume plan for tomorrow

1. Export stable scratch paths:

```bash
export HOME=/home/scratch.shaunakj_other
export TMPDIR=/home/scratch.shaunakj_other/tmp
export UV_CACHE_DIR=/home/scratch.shaunakj_other/.cache/uv
export XDG_CACHE_HOME=/home/scratch.shaunakj_other/.cache
export VLLM_CACHE_ROOT=/home/scratch.shaunakj_other/.cache/vllm
export HF_HOME=/home/scratch.shaunakj_other/.cache/huggingface
export HF_DATASETS_CACHE=/home/scratch.shaunakj_other/.cache/hf_json_cache
export RAY_TMPDIR=/home/scratch.shaunakj_other/raytmp
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$HF_DATASETS_CACHE" "$RAY_TMPDIR"
```

2. Re-check runtime:

```bash
cd /home/scratch.shaunakj_other/Development/RL
.venv/bin/python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), vllm.__version__)"
```

3. Continue benchmark setup:

```bash
cd /home/scratch.shaunakj_other/Development/RL
TMPDIR=/home/scratch.shaunakj_other/tmp .venv/bin/vllm bench throughput --help
```

4. Run A/B benchmark once flags are confirmed:
- Baseline: verifier only (`Qwen3-14B`)
- Speculative: verifier `Qwen3-14B` + draft `Qwen3-1.7B`
- Compare total output tokens/sec and latency; include acceptance ratio from speculative run.

## Related docs

- `docs/specdecode-zero-acceptance-memory-2026-02-17.md`
- `docs/repro-spec-decode-vllm-0.16.0-acceptance-rerun.md`
- `docs/repro-specdecode-32b14b-greedy-openmath64.md`

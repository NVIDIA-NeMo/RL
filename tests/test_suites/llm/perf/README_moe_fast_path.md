# MXFP8 MoE fast-path regression guards

These guards keep the Nemotron MXFP8 RL recipes on the optimized
**flashinfer trtllm-gen MXFP8 MoE** generation path. The recurring failure mode
in this area is that the fast path silently turns *off* — a dropped env var, a
deleted config knob, expert-parallel getting re-enabled, or a kernel-cache
regression — while generation still returns correct output, so only a
throughput drop ever reveals it. Each guard below fails loudly on one of those
regressions.

## What engages the fast path

For `grpo-nemotron-super-{8n8g,16n4g}-mxfp8`:

- **Suite-script env** (`tests/test_suites/llm/<recipe>.sh`):
  `VLLM_USE_FLASHINFER_MOE_FP8=1`, `VLLM_FLASHINFER_MOE_BACKEND=latency`,
  `NRL_VLLM_USE_V1=1`, and the CLI override
  `policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN`.
- **Recipe YAML** (resolved): `vllm_cfg.precision: fp8`, `fp8_cfg.is_mx: true`,
  generation `expert_parallel_size: 1`, `vllm_kwargs.mamba_ssm_cache_dtype:
  float32`, `compilation_config.cudagraph_mode: 0`.

At runtime a healthy MXFP8 serve logs `quantization=modelopt_mxfp8`,
`moe_backend='flashinfer_trtllm'`, and the kernel
`flashinfer::trtllm_fp8_block_scale_moe`. A BF16 run instead shows
`trtllm_bf16_moe` and no `modelopt_mxfp8`.

## The guards

| Guard | Layer | Runs in | Catches |
|---|---|---|---|
| `tests/unit/models/generation/test_moe_fast_path_guard.py` | static | default CI (no GPU) | dropped env var, `is_mx`→off, `precision`≠fp8, EP re-enabled, mamba cache dtype change |
| `tests/check_moe_fast_path.py <run_log>` | runtime log | nightly GPU suite | MoE silently ran bf16 / fell back; autotune-fallback storm |
| `tests/check_moe_speed.py --mxfp8 <json> --bf16 <json>` | runtime perf | nightly GPU suite | MXFP8 throughput regressed below BF16 at production concurrency |

The static guard `pytest.skip`s where the MXFP8 recipes are absent (e.g. public
upstream), so it is safe to merge anywhere and activates wherever they exist.

## Perf floor: the concurrency crossover

The MXFP8 trtllm-gen MoE wins decisively at production concurrency and only
loses at very low concurrency (small-M, memory-bound) — a structural kernel
property, not a regression. Measured head-to-head on Nemotron Ultra (standalone
`vllm serve` A/B, `ignore_eos`):

| shape | c64 | c128 | c256 | c320 |
|---|---|---|---|---|
| ab_mid    | 1.01× | 1.38× | 1.82× | 2.69× |
| ab_decode | 0.89× | 1.57× | 2.52× | 2.99× |

(MXFP8 / BF16 throughput.) So `check_moe_speed.py` enforces MXFP8 ≥ BF16 only at
concurrency ≥ 128 (the RL rollout regime); below that the loss is expected and
documented.

## Running the runtime A/B

`tests/test_suites/llm/perf/moe_mxfp8_vs_bf16_serve_bench.sh` is the standalone
pure-`vllm serve` benchmark used to mimic NeMo-RL generation for the speed
comparison (apply the nemo-speed vLLM overlay so the modelopt MXFP8 path loads).
Run it once per mode (`MODE=bf16` and `MODE=mxfp8`) pointing at the BF16 and
MXFP8 checkpoints; each writes a `bench_<mode>.json` sweep + a serve log. Then:

```bash
python tests/check_moe_fast_path.py serve_mxfp8.log          # path actually used
python tests/check_moe_speed.py --mxfp8 bench_mxfp8.json --bf16 bench_bf16.json
```

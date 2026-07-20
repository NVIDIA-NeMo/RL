# SWE Rollout-Benchmark Analysis Scripts

Reusable **post-hoc analysis tooling** for the NeMo-Gym SWE rollout benchmark
(NVIDIA-NeMo/RL, `examples/nemo_gym/run_grpo_rollout_benchmark.py`). These turn a run's
results + retained agent trajectories into CSV tables and distribution summaries.

**Scripts only — no run data or results are included here.**

## Inputs (produced by a benchmark run)
- `<run>/nemo/nemo_gym_eval_results.jsonl` — one row per rollout: `reward`, the full
  OpenHands `response.output` (reasoning / `function_call` / message items; each
  `function_call` carries `prompt_str` + `generation_str`), and per-trajectory timers.
- Retained OpenHands trajectories: `…/swe_agents/swebench_results_<ts>/<instance>/trajectories/<instance>/output.jsonl`
  — for **real** per-request / per-action latency (`metrics.response_latencies`,
  `metrics.action_execution_latencies`).
- The model `tokenizer.json` — for exact generated-token counts.

## Scripts
| script | purpose |
| :-- | :-- |
| `gen_analysis_data.py` | `results.jsonl` → `trajectories.csv` + `requests.csv` (+ `DATA_DICTIONARY.md`) |
| `analyze_data.py` | outcome taxonomy, timing decomposition, per-instance difficulty |
| `analyze_run.py` | quick per-run reward / timeout / run-time summary |
| `extract_requests.py` + `analyze_tokens.py` | per-request token table → per-step/trajectory/request + prefill-vs-generated + long-tail |
| `extract_gen_latency.py` + `analyze_gen_dist.py` | per-generation turns + e2e → prompt-level distributions |
| `step_timing.py` | per-(run, step) phase-timing breakdown |
| `extract_oh_latency.py` + `analyze_latency.py` | **measured** per-request LLM latency + per-action latency (from `output.jsonl`) |

## Usage
```bash
# per-trajectory + per-request tables (exact gen tokens via tokenizer)
python3 scripts/gen_analysis_data.py <run>/nemo/nemo_gym_eval_results.jsonl \
    --outdir out/<run> --tokenizer <model>/tokenizer.json

# real per-request / per-action latency (no instrumentation needed — OpenHands logs it)
python3 scripts/extract_oh_latency.py out/latency run1=<swebench_results_dir> [run2=... ]
python3 scripts/analyze_latency.py out/latency
```

## Notes
- **Generated tokens** are exact (model tokenizer). **Prefill** is char-ratio estimated
  (~3.66 chars/tok, ±2 %) because `prompt_str` re-grows every turn (~3 B chars/run).
- **Latency** (per-request, per-action) is **measured** from OpenHands' own logs, not derived.
- Scripts stream the (multi-GB) results file — no full load into memory.

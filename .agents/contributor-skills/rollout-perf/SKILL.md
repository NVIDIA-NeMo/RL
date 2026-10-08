---
name: rollout-perf
description: Rollout (generation) performance for NeMo-RL on every generation backend (vLLM, SGLang, TensorRT-LLM, Megatron inference, Dynamo), for batch GRPO-style training and multi-turn agentic rollouts. Covers finding where rollout time goes, proving the effective engine configuration, the levers ranked by measured effect, an A/B protocol with correctness gates, and landing the change in recipes and CI time budgets.
when_to_use: Adding or changing a recipe or nightly/release/performance test that generates; a test is slow or close to its NUM_MINUTES; a recipe sets `enforce_eager`, `vllm_cfg`, `vllm_kwargs`, `sglang_cfg`, `trtllm_cfg`, `trtllm_kwargs`, `mcore_generation_config` or `dynamo_cfg`; comparing or switching generation backends; tuning NeMo Gym / agentic rollout throughput; reviewing such a PR; 'rollout is slow', 'generation time', 'cuda graph', 'enforce_eager', 'max_num_batched_tokens', 'max_num_tokens', 'prefix caching', 'block reuse', 'which engine is faster', 'speed up nightly test', 'GPU hours budget'.
---

# Rollout Performance

Generation is often most of an RL step, and the default configuration of an
engine can waste much of it. The method below works for every backend. The
engine-specific knob names live in the reference files:

- [`references/levers.md`](references/levers.md): each lever, its knob in
  every backend, the measured evidence and the traps.
- [`references/engines.md`](references/engines.md): per backend, how the
  NeMo-RL config reaches the engine, the log lines that prove the effective
  setting, the metrics, and the known landmines.
- [`references/measurement.md`](references/measurement.md): metrics, gates,
  evidence grades, the A/B protocol, comparison scopes and gap decomposition.

Seven rules:

1. **Find where the time goes before tuning.** The right lever depends on the
   bottleneck. In one investigation, a "25% slower engine" turned out to be
   71% caused by one trajectory that hit its timeout. Request latency matched.
2. **A YAML value is not evidence.** Prove each setting from the engine's own
   startup log or metrics. Several "configured" settings never took effect,
   for example:
   - SGLang `disable_radix_cache` is declared in NeMo-RL's config but never
     forwarded to the engine;
   - a TRT-LLM state-rounding option was requested but disabled at runtime;
   - a "TEP4" MoE layout actually ran as TP4/EP1.
3. **Change one factor, or one declared bundle.** Credit a bundle's gain to
   the bundle until a split confirms which part did it.
4. **Gate on correctness before speed.**
   - Training: logprob consistency and accuracy.
   - Evaluation and agentic runs: reward parity, timeouts and coverage.
   - A broken evaluator looked like the fastest run.
5. **Compare like with like.** Use the same container, code, data, placement
   class and window. Repeat runs: at least 3 per arm for engine claims and
   small effects. Never pool runs across clusters, engine versions or
   harness patches.
6. **Report distributions and tails**, not only means. In a synchronous step,
   the slowest sample sets the step time.
7. **Re-measure version-specific rules.** One engine version's optimum, or its
   bug workaround, does not carry over. "PIECEWISE only" was a vLLM 0.20
   accuracy workaround. A token budget that won on one engine regressed on
   another topology.

## 1. Characterize the workload

Collect these before choosing a lever:

| Fact | Where (NeMo-RL GRPO) | Why it matters |
| :-- | :-- | :-- |
| Generation share of the step | `timing/train/generation` / `timing/train/total_step_time` | Below ~50%, rollout tuning moves wall time little |
| Refit share | `timing/train/prepare_for_generation/total` (sync); `timing/train/weight_sync`, `idle/refit_bubble` (async) | Weight sync can rival generation for large models |
| Prompt vs. generated tokens | `train/mean_prompt_length`, `train/mean_gen_tokens_per_sample` | Prefill-heavy and decode-heavy workloads need different levers |
| Longest sample vs. cap | `train/max_gen_tokens_per_sample` vs. `max_new_tokens`; `train/truncation_rate` | When the longest sample hits the cap every step, the step is decode-tail-bound |
| Turns and context growth | `train/avg_turns_per_sample`, `train/max_turns_reached_rate`, Gym `turns_per_sample/*` | Multi-turn prompts repeat a growing prefix, so prefix reuse dominates |
| Load per engine replica | prompts × generations per prompt (× DAPO `batch_multiplier`) ÷ replicas; replicas = generation GPUs ÷ (TP × PP) | Compare with the engine's KV capacity and admission cap |
| Engine queue and KV use | vLLM log `Running`/`Waiting`/`GPU KV cache usage`; `generation_metrics/*` (vLLM async, Dynamo) | Distinguishes engine-bound from feeder-bound |

Typical shapes:
- **Math or code GRPO/DAPO:** short prompts, long decodes; often
  decode-tail-bound with tiny KV use.
- **Agentic SWE rollouts:** about 115:1 raw prefill-to-generated tokens; 2.8:1
  with a working prefix cache; model calls 67–80% of a trajectory.

## 2. Find where the time goes

Run the helper on a test's artifacts. CI and `tools/launch` runs write
`tests/test_suites/<domain>/<exp>/metrics.json` and `<jobid>-logs/ray-driver.log`
inside the code snapshot.

```bash
uv run --no-project python .agents/contributor-skills/rollout-perf/rollout_perf_report.py \
  <snapshot>/tests/test_suites/llm/<exp>/metrics.json \
  --log <snapshot>/<jobid>-logs/ray-driver.log
```

It prints the step breakdown, the token shape, the effective engine settings
found in the log for each backend, and a regime guess. Then classify:

| Regime | Signals | Levers that help | Levers that won't |
| :-- | :-- | :-- | :-- |
| Not generation-bound | Generation < ~50% of the step | Refit, logprob or training work; rollout knobs come later | Rollout knobs |
| Refit-bound | `prepare_for_generation` or `weight_sync` large | Refit transport and buffer settings ([engines.md](references/engines.md)) | Engine decode knobs |
| Decode-tail-bound | Longest sample hits the cap on most steps; generation time steady; low KV use | CUDA graphs; more, smaller replicas; MoE/attention kernel backend; speculative decoding with care | Token budget, prefix cache, memory |
| Engine-bound, prefill-heavy | Long or multi-turn prompts; requests waiting; high KV use | Scheduler token budget; prefix/KV reuse with session affinity; prefill CUDA graphs; parallel layout | Frontend workers |
| Feeder-bound | In-flight batch well below the admission cap, nothing waiting, KV use in single digits, GPUs often idle | Tokenizer path, HTTP/response workers, harness CPU and agents per node, concurrency | More KV memory, larger budgets |
| Tail-bound | p50 healthy; a few samples, trajectories or timeouts set the step | Explain the tail first: classify timeouts, check evaluators, consider async GRPO | Engine knobs until the tail is explained |
| Startup-bound | Setup or wall time minus step time is large | Parallel weight loading, prebuilt environments, cached compilation | Rollout knobs; report startup separately |

For a gap between two runs, decompose it per step, per phase, and by GPU idle
time before touching knobs ([measurement.md](references/measurement.md#finding-where-a-gap-comes-from)).

## 3. Prove the effective configuration

For every setting that differs between arms, find the engine's own evidence.
Ignore the driver's `Overrides:` and `MasterConfig(` echo lines; they show
the request, not what ran. Per backend:

- **vLLM:**
  - the `non-default args:` line;
  - the `Initializing a V1 LLM engine ... with config:` line (`enforce_eager`,
    `enable_prefix_caching`, `compilation_config` with `cudagraph_mode`);
  - `Chunked prefill is enabled with max_num_batched_tokens=`;
  - `GPU KV cache size: ... Maximum concurrency for ...`;
  - `Capturing CUDA graphs (...)`.
- **SGLang:** the engine's `server_args=ServerArgs(...)` line. NeMo-RL forwards
  only a whitelist of `sglang_cfg` keys, so check that each key you set is
  in it.
- **TRT-LLM:** the engine's `LLM Args:` dump. `trtllm_kwargs` is applied last
  and overrides every typed field.
- **Megatron inference:** the resolved `mcore_generation_config`, plus
  `mcore async scheduling steps (cumul):` when async scheduling is on.
- **Dynamo:** `[Dynamo:<group>] launching argv=` and
  `[Dynamo] launching frontend argv=`.

Details and gaps are in [engines.md](references/engines.md). Also confirm the
topology that actually served requests: the number of URLs, which router (if
any) is on the request path, and whether multi-turn sessions stay on one
replica.

## 4. Pick the lever

Ranked by measured payoff. Each row links to its section in
[levers.md](references/levers.md), which has the knob in every backend.

| # | Lever | Best measured effect (grade) | Regime |
| --: | :-- | :-- | :-- |
| 1 | [CUDA graphs](references/levers.md#1-cuda-graphs) (decode first, then prefill) | vLLM eager → graphs: generation −60% to −85% on 9 H100 nightly tests (C) | Decode-tail-bound |
| 2 | [Scheduler token budget](references/levers.md#2-scheduler-token-budget) | TRT-LLM 8,480 → 32,768: timeouts 311/960 → 1/320, tokens/s +44% (B). vLLM 2,048 → 4,096 at TP4×2: −15% (C) | Prefill-heavy, long context |
| 3 | [Prefix/KV reuse and session affinity](references/levers.md#3-prefixkv-reuse-and-session-affinity) | TRT-LLM block reuse: tokens/s +22.5%, rollout −18.7% (B) | Multi-turn |
| 4 | [Parallel layout](references/levers.md#4-parallel-layout) | vLLM TP4×8 replicas beat TP8×4 by ~25% over 4 steps (C); MoE EP bundle −5% (B) | Tail-bound decode, MoE |
| 5 | [Admission cap and memory](references/levers.md#5-admission-cap-and-memory) | A too-high cap killed engine startup on a hybrid model; caps sized to load (C) | KV-bound or hybrid models |
| 6 | [Kernels and collectives](references/levers.md#6-kernels-and-collectives) | MoE backend fix: per-turn p50 −12.8% (C); `backend: eager` restored gates (C) | Any |
| 7 | [Frontend, tokenizer and harness](references/levers.md#7-frontend-tokenizer-and-harness) | Tokenizer fix −10% wall; response workers +6% tokens/s; placement +5% (C/B) | Feeder-bound, agentic |
| 8 | [Speculative decoding](references/levers.md#8-speculative-decoding) | Only with acceptance tracked across the whole run | Decode-tail-bound |
| 9 | [Startup](references/levers.md#9-startup) | Wall time only | Startup-bound |

Do not use these as performance knobs:

- **KV, weight or state precision changes** (FP8 KV, low-precision Mamba
  state). They change numerics and need their own convergence study.
- **Prefix caching with router replay (R3).** R3 needs routes for every
  prompt token.
- **Settings copied from another engine, version or topology** without an
  A/B. Equal numbers have different semantics across engines.
- **Shared-filesystem JIT caches.** Pointing `TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT` or `XDG_CACHE_HOME` at shared
  lustre made a 32-rank Triton warmup take 20 min instead of 48 s.

## 5. A/B it

Follow the protocol in
[measurement.md](references/measurement.md#ab-protocol):

1. Freeze the container (not `rl.nightly.sqsh`, which is rebuilt daily), code,
   data, seed and placement.
2. Give each arm its own `CODE_SNAPSHOT_DIRNAME`. Otherwise the second arm
   silently reuses the first arm's snapshot and checkpoints.
3. Canary, then formal runs.
4. Pair steps, since the same seed gives the same data order.
5. Grade the evidence.

```bash
uv run --no-project python .agents/contributor-skills/rollout-perf/rollout_perf_report.py \
  --base BASE1/metrics.json [BASE2/metrics.json ...] \
  --treat TREAT1/metrics.json [TREAT2/metrics.json ...]
```

The report:
- pairs steps and gives the per-step generation-time ratio, with its range
  and tail;
- compares step, validation and setup time;
- flags workload drift in generated tokens;
- prints the logprob gates per run;
- warns when an arm has fewer than 3 runs.

## 6. Gate it

The treatment must pass against its same-window control. The full list is in
[measurement.md](references/measurement.md#correctness-gates).

- **Training:**
  - `train/token_mult_prob_error` median and the count of steps ≥ 1.04 don't
    get worse;
  - `train/gen_kl_error` mean unchanged;
  - `train/mean_gen_tokens_per_sample` unchanged;
  - validation accuracy within run-to-run noise;
  - the test's `check_metrics` passes. If the control also fails a gate, the
    failure is pre-existing; say so.
- **Evaluation and agentic runs:**
  - reward Wilson intervals overlap;
  - timeouts classified and kept in the denominator;
  - the same row, step and request coverage;
  - no material p99 regression.

## 7. Land it

- Put the setting in the recipe YAML with a one-line comment giving the
  measured effect and its grade.
  - Delete keys that only restate an inherited default; for example, recipes
    inherit `enforce_eager: False`.
  - `tools/config_cli.py minimize-check` rejects such keys for `llm/` and
    `vlm/` recipes.
  - Never leave an empty mapping such as `vllm_cfg:`; it loads as null and
    wipes the inherited block.
- Check child recipes through the `defaults:` chain. A parent change reaches
  children that may run on other hardware or engines; pin the old value in a
  child you did not test.
- If a known-slow setting must stay (for example `enforce_eager: true` for a
  correctness bug), add a comment that explains why.
- Lower `NUM_MINUTES` in the test script to about 2× the new measured wall
  time.
  - Nightly budget = NUM_RUNS × NUM_NODES × GPUS_PER_NODE × NUM_MINUTES / 60.
  - The total is asserted in `tests/unit/test_recipes_and_test_suites.py`.
- If the recipe sets `checkpointing.checkpoint_must_save_by` (a
  `DD:HH:MM:SS` duration from the start of training), keep it below the new
  Slurm limit. Leave room for the checkpoint save, and for Ray startup and
  setup, which run before the timer starts.
- In the PR, include:
  - a before/after table (generation per step, step time, validation,
    setup, wall time, gates);
  - the evidence grade;
  - container and job IDs;
  - the effective-config evidence.

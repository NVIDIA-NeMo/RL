# Speculative Decoding Enablement and Correctness Report

Date: 2026-02-24

## 1) Scope

This report explains:

- how speculative decoding (specdec) was enabled,
- how Token Acceptance Rate (TAR) is produced,
- whether the implementation is correct.

Primary run examined:

- `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-think-t0p6-p0p95-k20-s2-l4096-actckpt-loadfmtauto-100steps-2026-02-23-202232/run-100steps-spec0p6b-think-t0p6-p0p95-k20-s2-l4096-actckpt-flashattn-loadfmtauto.log`

No-spec comparison run examined:

- `/home/scratch.shaunakj_other/logs/grpo-32b-nospec-think-t0p6-p1p0-knull-l4096-actckpt-loadfmtauto-100steps-roll64-safe-2026-02-23-232933/run-100steps-nospec-think-t0p6-p1p0-knull-l4096-actckpt-flashattn-loadfmtauto-roll64-safe.log`

## 2) Short Answer

- Specdec was enabled through `policy.generation.vllm_kwargs.speculative_config.*` Hydra overrides.
- NeMo-RL correctly forwards that config to vLLM.
- vLLM then runs draft-model speculative decoding and uses rejection sampling for acceptance.
- TAR logging is wired correctly and appears to be functioning as intended in spec runs.
- Important caveat: the specific run used `top_p=0.95` and `top_k=20`, which current NeMo-RL guards mark as unsupported for RL logprob correctness.

## 3) How Specdec Was Enabled

### 3.1 Config-level enablement (run command / overrides)

In the spec run log, the startup overrides include:

- `++policy.generation.vllm_kwargs.speculative_config.model=...Qwen3-0.6B...`
- `++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=2`
- `++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1`

Evidence:

- spec run log, line 7.

The same log also shows `speculative_config` populated in printed config:

- spec run log, line 149.

### 3.2 NeMo-RL forwarding path

NeMo-RL takes `vllm_kwargs` from config and passes them through to `vllm.LLM(...)`:

- `/home/scratch.shaunakj_other/Development/RL/nemo_rl/models/generation/vllm/vllm_worker.py`
  - reads kwargs: line 310
  - builds `llm_kwargs` and applies `**vllm_kwargs`: lines 409-430

So `speculative_config` is not reinterpreted or dropped by NeMo-RL; it is passed directly to vLLM.

### 3.3 Runtime confirmation in vLLM engine init

The vLLM engine init line in the spec run includes:

- `speculative_config=SpeculativeConfig(method='draft_model', model=..., num_spec_tokens=2)`

Evidence:

- spec run log, lines 279+.

In the no-spec run, engine init shows `speculative_config=None`, which is expected:

- no-spec run log, lines 276/294+.

## 4) Why `method='draft_model'` Appears Even Without Explicit `method`

Your override did not explicitly set `speculative_config.method`.

vLLM supports automatic method detection and defaults unresolved model-based cases to `draft_model`:

- `/home/scratch.shaunakj_other/Development/RL/.venv/lib/python3.12/site-packages/vllm/config/speculative.py`
  - method docs and fields: lines 46-77
  - post-init comment about defaulting behavior: lines 283-290

That is why engine logs show `method='draft_model'`.

## 5) Acceptance Algorithm Used

For draft-model speculative decoding, vLLM uses rejection sampling:

- `/home/scratch.shaunakj_other/Development/RL/.venv/lib/python3.12/site-packages/vllm/v1/sample/rejection_sampler.py`
  - `RejectionSampler` class: line 29
  - docstring states it follows the 2211.17192 algorithm.

It is wired from the GPU model runner:

- `/home/scratch.shaunakj_other/Development/RL/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py`
  - imports `RejectionSampler`: line 154
  - instantiates sampler for spec decode: line 481.

## 6) How TAR Is Computed in This Stack

### 6.1 Metric capture on each async vLLM worker

`vllm_worker_async.py` starts a metrics thread and tracks:

- accepted spec tokens timeline,
- proposed/draft tokens timeline,
- selected metric names,
- candidate counters seen.

Key logic:

- `/home/scratch.shaunakj_other/Development/RL/nemo_rl/models/generation/vllm/vllm_worker_async.py`
  - counter name matching and selection: lines 282-391
  - selection/warning prints: lines 393-417
  - metric export: lines 436-457

Spec run evidence that selection happened:

- `Selected spec-decode counters: accepted=vllm:spec_decode_num_accepted_tokens, proposed=vllm:spec_decode_num_draft_tokens`
- spec run log, lines 389/401+.

### 6.2 Aggregation across DP workers

NeMo-RL collects these per-worker metrics into `generation_logger_metrics`:

- `/home/scratch.shaunakj_other/Development/RL/nemo_rl/models/generation/vllm/vllm_generation.py`
  - collection/aggregation: lines 822-902.

### 6.3 Step-local TAR computation

`compute_spec_decode_token_acceptance_metrics(...)`:

- converts cumulative counter timelines to per-step deltas,
- sums accepted/proposed deltas across workers,
- computes TAR = accepted / proposed when available.

Code:

- `/home/scratch.shaunakj_other/Development/RL/nemo_rl/algorithms/utils.py`
  - delta helper: lines 386-402
  - TAR metric computation: lines 405-456.

### 6.4 Reporting

Printed in training results:

- `/home/scratch.shaunakj_other/Development/RL/nemo_rl/algorithms/grpo.py`
  - `Token Acceptance Rate`: lines 1846-1873.

Printed in perf block as well:

- `/home/scratch.shaunakj_other/Development/RL/nemo_rl/algorithms/utils.py`
  - `Spec Decode Token Acceptance Rate`: lines 667-686.

Spec run log confirms both are printed repeatedly (same values), e.g. lines 661 and 707.

## 7) Correctness Evaluation

### 7.1 What is correct

- Enablement mechanism is correct: overrides -> NeMo-RL pass-through -> vLLM engine receives spec config.
- Runtime proof is present in logs (`speculative_config=SpeculativeConfig(...)`).
- TAR instrumentation is present, active, and consistent with selected counters.
- No-spec run properly has no spec config and no TAR lines.

### 7.2 Important caveats / risks

1. Sampling/logprob correctness caveat in the examined run.

- The spec run used `top_p=0.95`, `top_k=20`.
- Current NeMo-RL explicitly rejects `top_p < 0.99` or `top_k < 8000` in vLLM V1 mode because filtered logprobs are not supported for this RL path:
  - `/home/scratch.shaunakj_other/Development/RL/nemo_rl/models/generation/vllm/vllm_generation.py`, lines 89-114.
- This means: specdec wiring itself is correct, but that run configuration is not considered policy-logprob-safe by current code standards.

2. TAR is a monitoring estimate, not exact event-level accounting.

- It is based on periodic snapshots (`vllm_metrics_logger_interval=0.5` in run config) and first/last counter deltas per step.
- Good for step-level trend and aggregate acceptance; not exact per-token trace reconstruction.

3. Counter-name heuristic dependence.

- Metric selection includes robust preferred names plus fallbacks.
- If upstream vLLM metric names change significantly, TAR could show `N/A` or require matcher updates.

4. Duplicate display lines.

- `Token Acceptance Rate` and `Spec Decode Token Acceptance Rate` are two reporting surfaces of the same underlying computation.
- Not wrong, but redundant in logs.

## 8) Practical Verdict

Implementation status:

- **Specdec enablement: correct**
- **TAR instrumentation: correct for operational monitoring**
- **Run-level RL correctness for the specific spec run (`top_p=0.95, top_k=20`): not aligned with current NeMo-RL safety checks**

If you want training-logprob-correct settings with current guards, use:

- `top_p >= 0.99` (commonly `1.0`)
- `top_k` unset / `-1` / effectively no filtering (or `>= 8000` per current threshold logic)

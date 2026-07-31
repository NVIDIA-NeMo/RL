# Token Capture (Gate-Authoritative): Setup and Try-It Guide

How to set up and run the gate-authoritative token-in/token-out capture
pipeline from this branch. For the design itself see
[tq-gym-gate-authoritative.md](tq-gym-gate-authoritative.md); for the
stage-by-stage evidence see
[tq-gym-gate-authoritative-implementation-log.md](tq-gym-gate-authoritative-implementation-log.md).

## What you get

With `token_capture.enabled=true`, NeMo-Gym rollouts in the async
SingleController GRPO path run token-in/token-out: the Gym gate holds each
rollout's token lineage, vLLM workers stage per-call token deltas + logprobs
directly to the TransferQueue, and agent-facing messages plus the Ray return
become token-free. With `enabled=false` (the default) every legacy codepath
behaves exactly as before — the feature is dormant.

## Prerequisites

- The requirements of the async SingleController + NeMo-Gym path: the
  feature only engages with `env.should_use_nemo_gym=true` and the async
  vLLM generation backend.
- 2 GPUs for the smoke test below.
- `HF_TOKEN` exported (the functional test downloads the workplace-assistant
  dataset from Hugging Face).

## 1. Clone with submodules

The feature spans this repo **and** a pinned NeMo-Gym fork branch. The
submodule gitlink points at commit `e3b3eac6` on
[`pthombre/tq-gate-capture`](https://github.com/NVIDIA-NeMo/Gym/tree/pthombre/tq-gate-capture)
of the public NVIDIA-NeMo/Gym repo (upstream main + a pinned rev of
[PR #2124](https://github.com/NVIDIA-NeMo/Gym/pull/2124) + the gate work),
so the standard recursive clone resolves it with no extra remotes:

```bash
git clone --recurse-submodules git@github.com:NVIDIA-NeMo/RL.git
cd RL
git checkout pthombre/tq-gym-gate-capture
git submodule update --init --recursive
```

Verify the pin: `git submodule status 3rdparty/Gym-workspace/Gym` should
show `e3b3eac6...`. If it shows a `+` or a fetch error, re-run
`git submodule update --init --recursive` from the repo root.

Environment setup is otherwise unchanged from
[installation](../about/installation.md) — the Gym fork is an editable uv
workspace member, so `uv run` picks it up automatically.

## 2. Configuration

All knobs live under `token_capture:` in the master config; defaults are on
`TokenCaptureConfig`
(`nemo_rl/algorithms/single_controller_utils/config.py`) and the exemplar
block is in `examples/configs/grpo_math_1B_single_controller.yaml`:

```yaml
token_capture:
  enabled: false                      # the only switch you must flip
  staging_partition: "rollout_staging"
  on_capture_failure: "continue"      # continue: placeholder row | abort: fail rollout
  mixed_weight_version_policy: "allow"
  min_valid_fraction_per_group: null
  registration_ttl_s: 3600.0
  staging_ttl_s: 3600.0
```

Enable it on any SC + NeMo-Gym recipe with `++token_capture.enabled=true`.
Setup validation will reject configurations that enable capture without the
NeMo-Gym path or with `rollout_max_attempts_to_avoid_lp_nan != 1`.

## 3. Smoke test (2 GPUs)

The same SC + Gym functional test CI runs (see
`tests/functional/L1_Functional_Tests_SingleController.sh`):

```bash
export HF_TOKEN=...
uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh \
    ++token_capture.enabled=true
```

This prepares the workplace-assistant dataset, runs a short GRPO training
job through the gate, and asserts on the resulting metrics. Run it once
without the override first if you want a legacy-path baseline from the same
tree.

## 4. Larger runs

`swe/` contains the SWE-bench token-capture vs. legacy perf A/B: launch
tooling (`launch_swe_ab.sh`, `make_capture_config.py`), the runbook
(`SWE_RUN.md`), the pinned launch record (`EXPERIMENT_LAUNCH.md`), and
`aggregate_perf.py` for the comparison. `examples/swe_bench/` holds the
underlying async GRPO SWE recipe and launcher.

## 5. What to watch

Per-train-step `gate/*` metrics land in the SC logger (wandb/tensorboard):

- `token_in_rate` — fraction of model calls served token-in (the happy
  path). Drops indicate marker stripping or history edits by the agent;
  the run stays correct (text-mode fallback) but wasteful.
- `fallback_rate` by cause, `capture_failure_rate`,
  `digest_verify_failures`, `invalid_row_rate`, finalize latency,
  `wv_spread`.

Debug switches (env-gated, off by default):

- `NRL_SC_DUMP_TRAIN_ROWS=<dir>` — dump canonical training rows at publish
  time for legacy-vs-capture row diffs.
- `NRL_HTTP_BYTES_DIR` / `NG_HTTP_BYTES_DIR` — per-call HTTP byte counters
  on the RL vLLM worker / Gym middleware respectively (the headline
  bytes-per-token comparison vs. the token-echo path).

## Troubleshooting

- **Submodule fetch fails**: the gitlink must resolve to `e3b3eac6` on
  NVIDIA-NeMo/Gym; check network access to github.com and re-run
  `git submodule update --init --recursive`.
- **Everything falls back to text mode** (`token_in_rate` ≈ 0): the agent
  or a proxy is stripping the `ng_call_id` marker from assistant messages,
  or rewriting history above it. Correct but slow — see design doc § 3.3.
- **Placeholder-heavy groups**: check `capture_failure_rate` (worker-side
  staging failures poison rollouts under `on_capture_failure: continue`)
  and gate TTL expiries in the gate logs.

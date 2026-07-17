# Span Groups

Span granularity in NeMo-RL is controlled by `NEMO_RL_OTEL_SPAN_GROUPS` (or `span_groups` in the `telemetry:` block). The spec accepts a preset keyword, a comma-separated list of individual group names, or a mix (e.g. `default,generation,reward`).

For the general span-group mechanism — how gating works, why a disabled group costs ~nothing — see [lens: span groups](https://github.com/NVIDIA-NeMo/Lens). This page covers NeMo-RL's groups and the per-algorithm span hierarchy.

## Preset keywords

| Preset | Groups included | Relative cost |
|---|---|---|
| `default` | `job`, `checkpoint`, `evaluate` | Lowest — safe for production |
| `per_step` | `step`, `checkpoint`, `evaluate`, `rollout`, `generation`, `logprob`, `reward`, `advantage`, `policy_update`, `reference_policy`, `data_processing` | Moderate |
| `all` | every group (`job` included) | Highest — dev/debug |

### `per_step` deliberately omits `job`

`per_step` **excludes** the `job` group on purpose. `job` is the whole-run root span; if it were enabled alongside `step`, every training step would nest under one giant, ever-growing trace. Omitting `job` makes **each training step its own root trace** — bounded in size and easy to search one step at a time.

`job` lives in `default` (coarse: job + checkpoint + evaluate) and in `all` (one whole-run trace, useful for a short run). Choose `per_step` when you want to inspect individual steps; choose `default`/`all` when you want one trace spanning the run.

## `RLSpanGroup`

Defined in `nemo_rl/telemetry/span_groups.py`. Extends lens's base `SpanGroup` with RL-specific groups.

| Group | Origin | Controls |
|---|---|---|
| `job` | base | the whole-run root span (`rl.<algo>.job`) |
| `checkpoint` | base | `rl.<algo>.save_checkpoint` |
| `evaluate` | base | `rl.<algo>.evaluate` |
| `model_init` | base | model initialisation spans |
| `load_checkpoint` | base | checkpoint restore spans |
| `step` | base | `rl.<algo>.step` (one per training step) |
| `forward_backward` | base | forward/backward spans |
| `optimizer` | base | optimizer-step spans |
| `rollout` | RL | `rl.<algo>.collect_rollouts` |
| `generation` | RL | `rl.<algo>` generation + the driver-side `rl.vllm.generate` / `rl.vllm.generate_text` spans |
| `logprob` | RL | `rl.<algo>.compute_logprobs` |
| `reward` | RL | `rl.<algo>.compute_rewards` |
| `advantage` | RL | `rl.<algo>.compute_advantages` |
| `policy_update` | RL | `rl.<algo>.policy_update` (and `value_update` for PPO) |
| `reference_policy` | RL | reference-policy log-prob spans |
| `data_processing` | RL | `rl.<algo>.data_processing` |

## Examples

```bash
# Coarse spans only — default
NEMO_RL_OTEL_SPAN_GROUPS=default

# Per-step traces (rollout / generation / reward / advantage / policy update)
NEMO_RL_OTEL_SPAN_GROUPS=per_step

# Coarse job trace + generation spans only
NEMO_RL_OTEL_SPAN_GROUPS=default,generation

# Everything
NEMO_RL_OTEL_SPAN_GROUPS=all
```

## Per-algorithm span names

Span names follow `rl.<algorithm>.<operation>`. The controlling group is shown for each; a span is only emitted when its group is enabled *and* the rank is exporting.

| Algorithm | Spans |
|---|---|
| **GRPO** (sync + async) | `rl.grpo.job`, `rl.grpo.step`, `rl.grpo.data_processing`, `rl.grpo.collect_rollouts`, `rl.grpo.compute_rewards`, `rl.grpo.compute_logprobs`, `rl.grpo.compute_advantages`, `rl.grpo.policy_update`, `rl.grpo.save_checkpoint`, `rl.grpo.evaluate` |
| **PPO** | `rl.ppo.job`, `rl.ppo.step`, `rl.ppo.data_processing`, `rl.ppo.collect_rollouts`, `rl.ppo.compute_rewards`, `rl.ppo.compute_logprobs`, `rl.ppo.compute_advantages`, `rl.ppo.policy_update`, `rl.ppo.value_update`, `rl.ppo.save_checkpoint`, `rl.ppo.evaluate` |
| **SFT** | `rl.sft.job`, `rl.sft.step`, `rl.sft.data_processing`, `rl.sft.policy_update`, `rl.sft.save_checkpoint`, `rl.sft.evaluate` |
| **DPO** | `rl.dpo.job`, `rl.dpo.step`, `rl.dpo.policy_update`, `rl.dpo.save_checkpoint`, `rl.dpo.evaluate` |
| **RM** | `rl.rm.job`, `rl.rm.step`, `rl.rm.save_checkpoint`, `rl.rm.evaluate` |
| **Distillation** | `rl.distillation.job`, `rl.distillation.step`, `rl.distillation.data_processing`, `rl.distillation.collect_rollouts`, `rl.distillation.teacher_logprobs`, `rl.distillation.policy_update`, `rl.distillation.save_checkpoint`, `rl.distillation.evaluate` |
| **vLLM** (driver-side) | `rl.vllm.generate`, `rl.vllm.generate_text` — `generation` group; nested under the active rollout span |

`rl.<algo>.job` is a function-level span (via `trace_fn`) wrapping the whole run. Under `per_step` it is suppressed, so each `rl.<algo>.step` becomes a root trace.

## Span tags (categorical attributes)

These are set on spans for filtering — they answer "which one?" / "what kind?", not "how much?". Numerical values that change over time are **metrics**, not span tags (see [Metrics](metrics.md)).

| Tag | Meaning |
|---|---|
| `rl.iteration` | training iteration index |
| `rl.epoch` | epoch index |
| `rl.step` | step index |
| `rl.num_generations_per_prompt` | GRPO group size |
| `rl.backend` | generation backend (e.g. `"vllm"`) |

## Resource attributes (process tags)

Stable-for-the-run values, set once at init and attached to every span/metric: `rl.algorithm`, `rl.model`, `nemo.precision`, `dl.tensor_parallel.size`, `dl.pipeline_parallel.size`, plus `dl.rank` / `dl.world_size` (set automatically by lens). See [Configuration — Resource attributes](configuration.md#resource-attributes).

## Granularity guidance

| Span groups | Relative cost | Recommendation |
|---|---|---|
| Disabled (`NEMO_RL_OTEL_ENABLED=0`) | None | Default for smoke tests |
| `default` | Lowest | Safe for all production runs |
| `per_step` | Moderate | Per-step profiling; each step is its own trace |
| `all` | Highest | Development / deep debugging |

Non-exporting ranks have an empty span-group set — `is_span_group_enabled()` returns `False` everywhere, so no span objects are created at all. The disabled path is a `frozenset` lookup and an immediate return. See [lens: architecture](https://github.com/NVIDIA-NeMo/Lens).

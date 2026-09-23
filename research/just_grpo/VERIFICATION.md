# Megatron verification

The research driver supports Megatron training with reference vLLM or in-process Megatron generation. The earlier Transformers training adapter and its verification-only worker have been removed. Historical artifacts remain in their original run directories and Git history.

## Normalization and recipe cleanup (2026-09-24)

Training now uses native streaming token-mask counts, without a
`global_normalization_counts` override. Standard and Fast schedules partition
the selected targets across levels; sample counts are corrected only for
reporting. CPU tests check that accumulated native-style token counts equal
the selected-token count, including excluded samples.

The research CPU suite passed **124 tests**, with **3 dependency-related
skips** (the Megatron adapter module and two full-controller cases). The core
split-worker test module was also skipped because this environment lacks
Megatron Bridge. These checks do not constitute a new GPU smoke run.

Only standard and Fast 32-GPU Megatron-inference long recipes remain in the
recipe directory. Their resolved experiment settings are preserved; validation
temperature now follows training temperature when overridden. The functional
smoke script applies two-GPU, three-update overrides. Recipe names and counts
below describe historical runs.

## Async GRPO smoke (2026-09-24 UTC)

The native async collector completed three Fast JustGRPO updates with shared
Megatron inference in job **19209956** (TP=1/DP=2). The functional metric checker
passed. Final reward was **0.128947**, generation/training KL **0.000757**, and
reference KL **0.000882**. Two-sample validation accuracy was 0.2 before and
0.0 after; this smoke does not establish convergence. The collector drained
generation before training and refit. Generation KL after the first update
includes the permitted one-update policy lag.

[W&B run](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/jvu3kl4y).

The research wrapper now derives from native `MegatronGeneration`, preserving
upstream's colocated lifecycle/type checks. Reference vLLM has a native async
worker extension sharing its sampling and context checks with the sync worker.
Four 32-GPU async recipes cover standard/Fast and both generation backends;
vLLM uses 16 training plus 16 dedicated generation GPUs. CPU validation passed
**134 tests**, with **3 optional-dependency skips**.

The initial dedicated vLLM smoke, **19209955**, hung inside NCCL communicator
initialization: training loaded NCCL 2.27.5 while generation loaded 2.28.9.
The retry submission preloads the existing generation runtime's NCCL 2.28.9
library for both sides, without modifying either installed environment.
Retry **19210830** completed all three updates and initial/final validation;
the functional checker passed. Final reward was **0.069408**, generation/training
KL **0.001792**, and reference KL **0.001076**. Two-sample validation accuracy
was 0.0 before and 0.340476 after. This used two policy GPUs and two dedicated
generation GPUs, and exercised NCCL refit after each update.

[vLLM W&B run](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/5niduumx).

These small Fast smokes verify the native async collector, recomputed previous
scores, confidence selection, reference scoring, training, weight versioning,
and validation. They do not establish convergence or four-node async execution.

Artifacts:
`/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/just_grpo_async_smoke_20260924/`.

## Multi-node launcher runtime (2026-09-24 UTC)

The initial four-node vLLM jobs 19206158 and 19206159 failed before the
training driver started: upstream `ray.sub` ignores `RAY_START_CMD` and
`RAY_STATUS_CMD`, selecting the container's incomplete default Ray runtime.
The research launcher now bind-mounts the provisioned interpreter's Ray CLI
at `/opt/nemo_rl_venv/bin/ray` in each container. Its shebang selects the same
Python environment as the driver. It also uses upstream `BASE_LOG_DIR` and
explicitly changes to the repository before running the driver.

Container probe **19207128** passed Ray head startup (including the Ray client
server) and `ray status`, using the provisioned Python 3.13 environment.
The two unstarted inference submissions were canceled for replacement.
This probe verifies container runtime selection; the replacement four-node
training runs still need startup and training verification.

## Native vLLM and live Megatron smoke (2026-09-24 UTC)

Jobs **19205331** (reference vLLM) and **19205332** (Megatron inference)
completed with exit code 0 in 8m14s and 9m05s. Both ran revision `0c3b1404c`
on one node/two GPUs, TP=1/DP=2, ordinary JustGRPO (all positions), three
updates, two prompts/four completions per prompt, and two validation samples
before and after training. They inherited the matched 6x6 long recipes,
including reference KL coefficient 0.01. W&B and native GRPO timers were enabled.

| Generation | Final training reward | Generation/training KL | Reference KL (`train/kl_penalty`) | Validation before / after |
| --- | --- | --- | --- | --- |
| [reference vLLM](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/m6szo927) | 0.160526 | 0.000391 | 0.000274 | 0.000000 / 0.552381 |
| [Megatron inference](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/x711bq7y) | 0.148684 | approximately 0 | 0.000239 | 0.200000 / 0.166667 |

Both passed the functional TensorBoard checker. The checker now allows k3
floating-point roundoff down to -1e-7 (Megatron observed about -2e-9) and
zero-gradient equal-reward batches, while requiring at least one nonzero
training gradient. The first vLLM batch had all-zero rewards and no update;
subsequent batches had nonzero gradients. No training implementation changes
were needed for these runs.

These runs verify native vLLM engine startup, sleep/wake and refit, live
Megatron decoding, previous-policy recomputation, reference scoring, training,
and final validation. Megatron used async rollout dispatch within synchronous
GRPO. Async GRPO training, TP=2, and reward convergence remain unverified by
these short runs.

Artifacts (submission scripts, inherited/resolved configs, source revision,
TensorBoard events and logs):
`/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/just_grpo_dual_generation_smoke_20260924_0223/`.

## Custom 6x6 environment cleanup

Merged the NeMo-RL dataset/environment integration into `environments/sudoku.py`.
Removed `upstream.py`, the 4x4 adapter, its config branches and tests, and the
Reasoning Gym optional dependency. The 6x6 generator, prompts, reward, and
rollout message contract are preserved. CPU tests passed **119 tests** with
**3 optional-dependency skips**; Ruff checks and offline `uv lock --check`
passed. The lockfile drops nine unused Sudoku dependency records and preserves
all remaining package versions. No experiment was submitted.

## Research-only generation wiring and async dispatch

The research driver now calls unmodified upstream `setup()` and replaces its
colocated Megatron generation wrapper afterward. The added `generation_factory`
argument and branches have been removed from core. The existing core change
carrying generation scores into synchronous scoring inputs remains.

Both sync and async GRPO use the research decoder's `generate_async()` through
upstream rollout dispatch. Async GRPO uses the upstream collector and pause/drain
lifecycle, with in-flight weight updates disabled. It does not overlap training
and generation on the shared model. An inherited two-GPU async recipe enables
importance-sampling correction and the native colocated refit configuration.

The CPU research suite passed **119 tests**, with **4 optional-dependency skips**.
Ruff lint, import ordering, formatting, and whitespace checks passed.
CPU tests cover DP routing, partial batches, indexed out-of-order results,
serialization, draining on worker errors/cancellation, configuration guards,
and research-driver selection of both upstream trainers. Driver tests execute
the native generation-wrapper and weight-synchronizer constructors with mocked
infrastructure; they do not execute the full Ray collector or GPU worker.
The complete upstream-controller test remains optional because this CPU
environment lacks `soundfile`. No GPU experiment was submitted for this change.

## In-process Megatron generation

The standalone Hugging Face/PyTorch decoder, rotary wrapper, checkpoint export,
and subprocess handoff have been removed. The research decoder uses the live
Megatron training model through upstream `model_forward()`, `Policy.generate()`,
and colocated `MegatronWeightSynchronizer`. It uses fixed-width doubled inputs,
TP=1, and no KV cache. The replacement two-GPU recipe uses TP=1/DP=2.

The CPU research suite passed **103 tests**, with **4 optional-dependency skips**.
Tests check at-commitment scores against independently recomputed diffusion
schedule scores for reveal widths 1, 2, and 4; sampled MASK and per-sample EOS;
fixed context alignment; mixed-length batching; seed reproducibility; live
weight changes; and the native synchronization contract. These are CPU model
and mocked lifecycle checks, not a GPU Megatron execution test. No experiment
was submitted, and there are no new reward or KL measurements for this decoder.
Previous PyTorch GPU results below describe the removed implementation.

## Native generation integration (previous revision)

The shared generation adapter has been removed. vLLM now uses native
`VllmGeneration` and a research worker extension; the then-current PyTorch path implemented
`GenerationInterface` in `pytorch_backend.py`. Tests cover the fork's temperature
sentinel, commitment logprobs, context reserve, and delegation to the native
worker. The CPU research suite passed **101 tests** with **4 optional-dependency
skips**. Previous GPU runs below used subprocess vLLM and do not verify native
engine initialization, sleep/wake, or weight synchronization with the reference
fork. No new experiment was submitted for this refactor.

## Mandatory previous-policy recomputation

The research driver uses upstream `Policy` and its `get_logprobs()` directly.
The former `DiffusionPolicy` wrapper has been removed; worker selection uses
the existing `policy.worker_extension_cls_fqn` configuration.
Previous-policy scores are recomputed before every update; generation scores
remain separate for confidence selection, sampling correction, and diagnostics.
The `old_logprobs` option has been removed, and `force_on_policy_ratio: true`
is rejected to prevent skipping this forward pass.

After removing the redundant policy wrapper, CPU research tests passed
**100 tests**, with **4 optional-dependency skips**. Before that cleanup,
the research suite in the GPU runtime passed **121 tests** in job **19197533**.
That job also started a three-update TP=1/DP=2 Sudoku smoke; its completion is
not part of the verification recorded for this revision.

Artifacts: `/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/just_grpo_prev_logprobs_smoke_20260924/`.

## Upstream GRPO controller

The runs below predate mandatory previous-policy recomputation and used rollout
logprobs as the PPO denominator. They do not verify the revised denominator path.

`train.py` now calls upstream `setup()` and `grpo_train()`. Job **19195357**
passed 114 research tests and the reference-vLLM three-update Fast JustGRPO smoke
on Sudoku 6x6 with TP=1/DP=2, including initial/final validation, reference KL,
weight export/reload, and native GRPO metrics/timers. Job **19195844** then
completed 120 research tests and the matching PyTorch-generation smoke.

| Generation | Final training reward | Generation/training KL | Reference KL | Validation before / after |
| --- | --- | --- | --- | --- |
| [reference_vllm](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/xx9lpzh0) | 0.087336 | 0.001650 | 0.001480 | 0.311905 / 0.200000 |
| [pytorch](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/3bdfkf1b) | 0.153618 | 0.002053 | 0.001428 | 0.200000 / 0.066667 |

The first allocation subsequently failed during PyTorch model-code import before its
first generation. Ray had passed the training worker's model-code cache through
`PYTHONPATH`; isolated decoding now puts its own `HF_MODULES_CACHE` first.
The successful PyTorch retry verifies that fix and generation from updated weights.

The refactor also fixes the diffusion schedule to the model's configured context
width: dynamic rollout widths do not match Nemotron's precomputed attention mask.
CPU coverage after these fixes passed 100 tests with 4 optional-dependency skips.
These short checks establish execution and score alignment, not convergence.
Checkpoint resume and TP=2 are not verified.

Artifacts: `/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/just_grpo_upstream_controller_smoke_20260924_retry2/`.
PyTorch retry: `/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/just_grpo_upstream_controller_pytorch_20260924_retry3`.

## Previous standalone driver with both generation backends

After removing the separate training adapter, job **19191574** completed the
123-test research suite and two sequential three-update Fast JustGRPO runs on
Sudoku 6x6, both TP=1/DP=2. The vLLM and PyTorch runs both passed functional
checks on both ranks, with identical DP gradient norms and parameter updates.

| Generation | Final training reward | Generation/training KL | Reference KL | Validation before / after |
| --- | --- | --- | --- | --- |
| [reference_vllm](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/kobchs9g) | 0.073529 | 0.002333 | 0.001747 | 0.311905 / 0.347619 |
| [pytorch](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/oy5wxel0) | 0.104779 | 0.002230 | 0.001222 | 0.200000 / 0.200000 |

At that revision, the Megatron smoke, standard long, and Fast long recipes resolved to
unchanged GRPO, loss, data, optimizer, generation, Megatron, and allocation
settings. PyTorch generation retains its own checkpoint loader and rotary
precision helper; policy training and reference scoring use Megatron only.
These smoke runs establish execution and score alignment, not convergence.

Artifacts: `/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/just_grpo_megatron_only_smoke_20260923_2135/`.

## Native iterator and split training API

Job **19189701** completed three Fast JustGRPO updates on Sudoku 6x6 with TP=1/DP=2 and vLLM generation. It used four denoising views, one optimizer update per rollout batch, original-batch normalization, reference KL, weight export/reload, and initial/final validation. Both ranks passed the functional checker with matching gradient norms and parameter updates.

[W&B run](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/bezzcu0m).

| Update | Training reward | Generation/training KL | Reference KL |
| --- | --- | --- | --- |
| 1 | 0.198529 | 0.001983 | 0.0 |
| 2 | 0.210084 | 0.002353 | 0.001147 |
| 3 | 0.083946 | 0.001345 | 0.001481 |

Validation reward was 0.311905 before and after. This three-update smoke verifies execution and score alignment, not reward convergence. Its artifacts are in `/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/fast_just_grpo_native_iterator_smoke_20260923_2058/`.

## Same-position core loss

Job **19190640** completed 20 research GPU tests and 140 core loss tests after both diffusion loss callers switched to `shift_labels=False`. Tests compare losses, metrics, and gradients against the former padded-metadata formulation, including first-position targets, token/sequence normalization, and reference KL. Existing autoregressive callers retain the default shift.

Artifacts: `/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/just_grpo_loss_alignment_tests_20260923_2118/`.

## Scope

CPU tests cover denoising schedules against independent serial canvases, Fast selection, score aggregation, PyTorch sampling, rotary precision, configuration inheritance, environment rewards, and upstream rollout contracts. Megatron tests cover processors, same-position gradients, native worker lifecycle, and original-batch metric normalization.

TP=2 verification is deferred (P1). The native-iterator GPU smoke above used TP=1. Upstream unmodified vLLM/SGLang leftmost diffusion generation and long-run PyTorch throughput/convergence are not established by these checks.

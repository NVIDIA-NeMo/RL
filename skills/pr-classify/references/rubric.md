# PR classification rubric

These rules come from a maintainer audit of 113 merged NeMo-RL PRs (November 2026). Each rule names the PR it was learned from, so you can look at that PR's diff when a case is close. A few came from the audit notes rather than one PR, and say so.

## How to apply the rules

- **Judge from the code only.** Use the diff, the changed paths, and the PR title and description for intent. Do not use review comments, comment counts, bugs found in review, or explainer pages: a new PR has none of these.
- **A specific rule beats a general one.** The general guidance under each call is a starting point. When a specific rule below matches, it decides the call.
- **Every call needs a basis.** That is either a rule from this file or a concrete fact from the diff (file + what changed). If you cannot write one, answer `unknown`. Medium vs high and low vs medium are the usual hard boundaries. If no rule or diff fact separates them, answer `unknown`.
- **Off by default does not lower a call.** Judge a feature as if it were turned on. Authors often ship a feature off by default so they can ablate it. New code that nothing calls yet is judged the same way, because someone can turn it on. This applies to complexity and convergence risk.
- **Judge the PR by its main change.** A side change that rides along (a lockfile re-resolve, a small restart-filter tweak inside a logging PR) does not decide the call unless a rule says it does.

## 1. Complexity: `low` | `medium` | `high` | `unknown`

How hard is this PR for a maintainer to review correctly?

General guidance:
- Start from source lines (not tests or docs), but size is only a start.
- Typos, CI labels, version strings and small test-only fixes are `low`.
- Design questions (new abstraction, new config surface, new worker or actor type, cross-module refactor) are `medium` to `high`.
- Timing and concurrency source changes (async code, Ray actor ordering, barriers, overlap, refit and weight sync, retries, timeouts) are at least `medium`, even when small. A recipe-only switch of such a setting (e.g. `refit_transport` in a YAML) is a recipe change, not a source change.
- Docs-only is usually `low`, but judge the content. Whitespace, spelling, reflow and moving pages are `low` even at thousands of lines (#4998: ~1.2k lines of moved docs). A large rewrite where every page must be checked for correctness (e.g. a 63k-line docs refactor) is `high`.

When is it `high`? Any of these:
- **It changes convergence at a few hundred source lines or more** (#4307: router replay, ~500 lines).
- **It combines two hard areas**, e.g. refit + low precision (#3638).
- **It changes what the loss sees or counts, at any size.** Size does not cap complexity here:
  - skipping a logprob pass, even in 16 lines (#4670, CISPO single forward; #5130, #5080 in-loss sequence filtering). Logprobs feed masks in many places, so skipping them spreads.
  - changing which tokens count toward reward lengths or loss masks, even at ~60 lines (#4487).
- **Loss or advantage code at about 200+ source lines**, even in one file. Below that it is `medium`, and `low` only when trivially benign (reindent, reorder arguments, rename). (From the maintainer audit notes, not a single PR.)
- **It runs a core step on a new execution path**, e.g. the weight synchronizer in a separate actor (#5182).
- **It touches vLLM's executor layer**: the DP path, executors, the Ray vs MP backend, or which patches apply in each. That holds even at ~280 lines (#2517), because the reviewer must know how vLLM uses Ray and how our patches fit in.
- **Streaming training in the single controller**, e.g. capping streaming train chunk size (#5036).
- **Startup speedups that add threading or interleaving** (#5001).
- **Caches that must sync or live on the file system**, e.g. persistent compile caches in the launcher, pulled from S3, or baked into the image (#4715).
- **Rollout-checkpoint recovery at a large size**, e.g. ~850 lines restoring router-replay state across prefixes (#4612).

At least `medium`:
- A small change to an existing loss's math, e.g. clamping a ratio before `exp`, is `medium`. It is `high` only at about 200+ source lines, or when it changes what the loss counts or skips. (From the maintainer audit notes, not a single PR.)
- Overlap, async or thread-pool perf features are `medium`, not `high`, unless another `high` rule also matches (#5154).
- Fixes to how training-step metrics are counted or windowed (#5155).
- Refactors of a few hundred source lines, e.g. ~670 lines extracting refit parameter-spec collection (#4764). With that size, a bad design would be rejected, so a human must check it.
- Any change to how a model checkpoint is saved or restored, even at ~50 lines (#4511). Think of backward compatibility, LoRA, and every place a checkpoint can load from (Hugging Face, Megatron-Bridge, Megatron-LM).
- New config surface, even a simple pass-through option (#4481: a configurable NCCL timeout, ~60 lines). The reviewer must think about the config's shape.
- Tracing and telemetry features (OpenTelemetry, NeMo Lens) at about 200+ source lines (#4925). They tend to copy code and pick awkward abstractions. A small one with obvious intent can be `low`.
- Perf changes that persist less state or save it later (rollout or trainer data-plane checkpoints) are `medium` (#5139).
- A sampler behavior change, even at ~20 lines, is `medium`, not `low` (#5079).
- Event logs and ledgers follow size: ~330 lines is `medium` (#5048).
- Chat template changes are at least `medium`, since they decide what tokens training sees (from the audit notes: moving prefix chat template handling out of vLLM into the OpenAI server utilities).
- A source change in the single controller (rollout manager, async utilities, overlap and barriers) is at least `medium`, even when small (from the audit notes).

`low` even in a hard area:
- **A fix for a plain functional failure** (something broken, not a subtle correctness issue), e.g. restoring calls a resumed turn carried over under forest training (#5147). A hard topic does not make every PR in it hard.
- CI workflows and linter or type-checker upgrades, even at a few hundred lines (#5345, #5298).
- Launcher shell scripts and start order (#5019).
- Raising an artificial concurrency cap, e.g. Ray's `max_concurrency` (#5120).
- Tests-only PRs (#4507).
- A pure revert of recently landed commits, even at hundreds of lines (#5302).

## 2. Benign: `yes` | `no` | `unknown`

`yes` means the PR clearly cannot change training results, performance, or what users see.

`yes`:
- Docs, comments, tests only, pure renames, and CI config that does not change what runs.
- Metrics, traces and telemetry fixes (#5049, #4894). Event logs and ledgers (#5048). Exception: fixing how training-step metrics are counted or windowed is `no` (#5155), because people tune runs from those numbers.
- Installing system packages for tests or nightlies (#4789).
- Raising an artificial concurrency cap (#5120).
- A perf refactor that is provably equivalent, e.g. skipping a redundant `masked_select` (#5192).
- *(tentative)* A recipe-only fix that makes a broken run work again, e.g. capping `max_new_tokens` (#5314).

`no`:
- **Any perf change**, since benign covers performance (#4769), except a provably equivalent refactor (#5192, above).
- Launcher changes that alter how every run behaves, even if not what it learns, e.g. raising the fd soft limit (#5058).
- Patching a third-party library's behavior from our side, e.g. patching PyTorch for parallel torch_dist reads (#4714).

## 3. Discuss design first: `yes` | `no` | `unknown`

Should the author have talked with a maintainer before starting the work?

`yes`:
- New config surface, a new worker or actor type, changing who owns some state, cross-module refactors, a new public API.
- A PR big enough that a large refactor is possible, or where the right shape is not obvious, e.g. a first integration of a new serving stack such as Dynamo.
- A new metric whose definition is open, e.g. whether batch age is counted per step or per token (#5049).
- The intended effect depends on a mechanism the author may not understand (#5036: a streaming cap that did nothing at first because no slots opened).
- It adds many environment variables, or a new class that could have been agreed on first (#5001).
- Caches that must sync or live on the file system (#4715).
- Pausing and resuming generation engines around refit (#3929).

`no`:
- Small design choices inside an existing pattern.
- Adding another variant the existing way, e.g. adaptive PPO clip next to the other clip modes (#4750).
- Turning on an option a backend already supports, e.g. Mooncake SSD offload (#4746).
- A non-trivial feature inside an existing subsystem, e.g. snapshotting rollouts at the checkpoint deadline, ~380 lines (#5056).
- Pure optimizations that defer or skip persisting state (#5139).

## 4. Convergence risk: `none` | `low` | `medium` | `high` | `unknown`

Could the change alter learning (loss, advantages, sampling, data order or packing, logprob math, weight sync, precision, retries that drop or duplicate samples)?

- `none`: it cannot reach a training run at all. New training code that is off by default or not yet called is NOT `none`.
- `low`: it touches code a run executes but is unlikely to change learning.

`high`:
- **Every dependency upgrade**: Gym, vLLM, Megatron-Bridge, Transformers, Automodel, uv.lock, 3rdparty submodules, the container (#5005, #4852, #4793, #4753, #5195). Scope: a lockfile change that only rides along with a small feature (one extra package, or a re-resolve that moves a few pins) is not an upgrade (#4894). Neither is a bump of a dev-only tool that no training run imports (linter, type checker, formatter) (#5298).
- **Anything low precision** (FP8, FP4, MXFP8, quantization), including first-time FP8 support (#3638, #3294).
- **Masking, advantages or losses**, even when framed as an optimization (#5130), and skipping work the loss depends on (#5080), unless it is a provably equivalent refactor (#5192), which is `low`.
- **Sampler behavior changes**, e.g. allowing a forced ratio with ready-first sampling (#5079). Keeping or dropping the KV or prefix cache across pause and resume is a sampler behavior change, because it changes how async RL behaves (from the audit notes).
- **Code shared by every run on an engine**, e.g. reworking vLLM worker code or the native vLLM path (#4979, #4905).
- **Core mid-rollout recovery**: checkpointing in-flight rollout state and resuming from it (#4500, #4506). The resumed run must converge like an uninterrupted one.
- **Pausing and resuming engines around refit**, with KV recompute (#3929).

`medium`:
- Recovery that restores one extra piece on top of core recovery, e.g. multimodal prefixes or router-replay state (#4583, #4612).
- Router replay itself (#4307).
- Recipe changes to learning settings (KL weight, clipping, overlong filtering, reward shaping), e.g. aligning DAPO recipes (#5263). Proof includes `convergence`.
- Perf changes that persist less state or save it later (#5139). Adding a new save point instead is `low` (#5056).

`low`:
- A new execution path for a core step (#5182).
- Dtype changes in a training recipe, e.g. FP32 MoE routers (#5187).
- Streaming chunk caps in the single controller (#5036).
- Scaling out a serving layer, e.g. more frontends (#5115).
- Turning on an option a backend already supports (#4746).
- Large refactors that move a subsystem, e.g. rollout checkpointing out of the single controller, ~5.9k lines (#5011).
- First-time support that changes behavior for models or inputs that already worked (#5201).
- Fixes for plain functional failures (#5147).
- Fixes to training-step metric counting (#5155).

`none`:
- CI and dev tooling (#5345, #5298), profiling (#5018), launcher scripts and start order (#5019), launcher process limits (#5058).
- How config objects are built or validated, e.g. `model_validate` (#5306).
- Actor placement, e.g. pinning a CPU-only actor to a node (#5051).
- Event logs and ledgers (#5048), installing system packages (#4789), raising a concurrency cap (#5120).
- When the checkpoint deadline is measured from (#4811).
- First-time support for something that never worked, e.g. token capture for a new model's images (#5201). Adding a branch for the new model inside a shared file is still `none`, as long as models that already worked behave the same. Low precision is the exception (`high`).

## 5. Proof needed: list of `functionality`, `convergence`, `perf`; or `none`; or `unknown`

What evidence should the PR show (W&B runs, nightlies, benchmarks)?

- **Dependency upgrades: all three** (#5005, #5195, #4852).
- **A new model or recipe following the existing pattern:** `convergence`, `perf`, or both (from the audit notes).
- **A recipe switch of `refit_transport` or the MoE backend:** `perf`, since backends are meant to be functionally the same (from the audit notes).
- **Every perf change needs `perf`**, even small or launcher-only ones (#5026, #4713). Perf includes memory (#5036). Scaling out serving needs `perf` to show no loss, not a gain (#5115).
- **A new loss, loss term or algorithm variant needs `convergence` only** (#4750, #5006, #5013). A convergence run also shows it works, so do not add `functionality`. The same holds for a fix or tweak to an existing loss.
- **Skip-work optimizations whose speed gain is obvious need `convergence` only** (#5080, #4670). Masking, advantage or loss optimizations whose gain must be shown need `convergence + perf` (#5130).
- **Low precision:** a new or rewritten path needs `functionality + convergence` (#3638). A perf change to a working path needs `convergence + perf` (#3294).
- **Core recovery, recovery of extra state, router replay, pausing and resuming engines:** `functionality + convergence` (#4500, #4583, #4307, #3929). Convergence shows it still learns the same, or that it helps.
- **vLLM executor-layer changes touching FP8:** `functionality + convergence` (#2517).
- **Persisting less or later:** `functionality + perf`, where functionality shows it is correct as designed (#5139). Shared caches: `functionality + perf` (#4715).
- **Streaming changes in the single controller:** `functionality + perf` (#5036).
- **New execution path for a core step:** `perf` when the point is speed (#5182). Recipe dtype changes for speed: `perf` (#5187).
- **`functionality` only:** metric-counting fixes (#5155), equivalent refactors (#5192), first-time support (#5201), large subsystem moves (#5011), fixes for plain functional failures (#5147), launcher limits (#5058), tests-only PRs (#4507), telemetry features (#4925), a non-trivial feature in an existing subsystem (#5056).
- **A nightly fix needs `functionality`**: show the nightly passes (#4788). A linter or type-checker upgrade also needs `functionality`: show the new gate passes (#5298).
- **Perf changes in the data plane need `functionality + perf`** (#4769): show it still returns the same data.
- **`none`:** docs-only changes; CI-only changes, including changes to test thresholds or checks (#4933, #5055); config construction refactors such as `model_validate` (#5306); pure reverts (#5302).
- If you cannot tell whether extra runs are warranted: `unknown`.

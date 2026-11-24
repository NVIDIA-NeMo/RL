# PR categories

Tag each PR with one or more categories by their key. Each tag is either **main** (what the PR is for) or **aux** (a side area it also touches: a cleanup riding along, a fix bundled in, a side effect). Pairing two main areas is fine (single controller + cleanup, sampler + loss). An aux tag that doesn't seem to belong is a warning sign worth a line in the comment.

Judge from the code, not the description. Categories feed the calls (some pairs make a PR hard), but they don't replace them. Grading rules live only in [rubric.md](rubric.md); this file only says what each category covers.

## Training math: what the model learns from each step

- **`loss`: Loss function.** Loss code, KL, clipping, token weighting. Often paired with `fix`.
- **`advantage`: Advantage calculation.** Advantage or estimator code, reward normalization.
- **`lowprec`: Low precision.** FP8, FP4, MXFP8, quantization, modelopt. It changes numbers quietly, like a loss change.

## Algorithm: how the RL loop runs

- **`singlectl`: Single controller.** Rollout manager, async utilities, overlap and barriers (everything once called "rollout orchestration", except samplers).
- **`sampler`: Samplers / async RL.** Ready-first vs in-order sampling, staleness, and KV or prefix cache kept or dropped across pause and resume.
- **`newalgo`: New algorithm.** A new training method or update rule (e.g. MOPD; LoGRA, which replaces the dense AdamW update with low-rank gradient sketches, is `newalgo` + `perf`). LoRA adapters are not a new algorithm; they go under `backend`.

## Generation and refit

- **`generation`: Generation backend.** vLLM, Dynamo, SGLang, TRT-LLM.
- **`refit`: Refit.** Moving weights from training to generation. A recipe-only switch of a refit setting (e.g. `refit_transport` in a YAML) is `recipe`, not `refit`.

## Fault tolerance: surviving failures without changing results

- **`rolloutckpt`: Rollout checkpointing.** Partial checkpointing of rollouts and recovery from it. Model checkpoints go under `backend`; data-plane checkpoints go under `dataplane`.
- **`resilience`: Resiliency.** What happens when a worker dies: retries, condemning bad workers, health checks, routing around failed generation workers. It can change convergence (are samples dropped, duplicated or retried forever?) and performance.

## Environments: where rewards come from

- **`gym`: Environments / Gym.** NeMo Gym integration, reward environments.
- **`agentic`: Agentic training.** Multi-harness training, context compaction, sub-agents, forest training.

## Data: what training sees and how it gets there

- **`dataplane`: Data plane.** Data-plane interfaces, adapters, token sink, data-plane checkpoints.
- **`dataload`: Data loading and packing.** Sequence packing, dynamic batching, the Energon data loader.
- **`multimodal`: Multimodal.** Image and video inputs, vision encoders. Usually paired with another category.
- **`chattmpl`: Chat template / message handling.** Prefix chat templates, how messages are tokenized, the OpenAI server utilities.

## Models and backends

- **`newmodel`: New model / recipe.** Adds a model or recipe config following the existing pattern.
- **`backend`: Training backend.** Megatron, DTensor / Automodel internals, parallelism and sharding layouts, LoRA adapters, and model checkpointing (saving and loading weights and optimizer state, storage backends, async saves).
- **`recipe`: Recipe change.** Edits an existing recipe or nightly config (YAML) without new code: switching a backend or transport, tuning a setting, changing test thresholds. Also tag the area of the setting. Switching `refit_transport` or the MoE backend is also tagged `perf`.

## Kind of change

- **`fix`: Fix.** Fixes a bug. Usually paired with an area.
- **`perf`: Perf only.** Faster, less memory, offload, faster startup, with no functional change claimed.
- **`config`: New config values.** Adds config knobs.
- **`refactor`: Refactor.** Moves or reshapes code with no behavior change meant (e.g. a config dict becoming a Pydantic model).
- **`cleanup`: Cleanup.** Renames or removes keys, dead code, small internal renames.
- **`breaking`: Large breaking change.** Only huge, obvious ones: deleting a whole backend or a major feature. Small renames never count. Not bad in itself, but maintainers must know.

## Infra and tooling: around the run, not inside it

- **`ci`: CI.** Workflows, test lanes, labels.
- **`testsonly`: Tests only.** Adds or fixes tests, no source change.
- **`deps`: Dependency bump.** uv.lock, pyproject, 3rdparty submodules, the container. A lockfile change riding along with a small feature is not a bump. A dev-only tool bump (linter, type checker) is not a bump either (#5298).
- **`launchers`: Launchers.** ray.sub, k8s, Slurm, launch scripts.
- **`logging`: Logging / metrics.** W&B, telemetry, tracing, profiling.
- **`eval`: Evaluation.** Eval and validation loops, eval-only scripts.
- **`docs`: Docs / skills.** Docs, `skills/`, AGENTS.md.

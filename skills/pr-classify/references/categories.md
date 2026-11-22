# PR categories

Tag each PR with one or more categories by their key. Each tag is either **main** (what the PR is for) or **aux** (a side area it also touches: a cleanup riding along, a fix bundled in, a side effect). Pairing two main areas is fine (single controller + cleanup, sampler + loss). An aux tag that doesn't seem to belong is a warning sign worth a line in the comment.

Judge from the code, not the description. Categories feed the calls (some pairs make a PR hard), but they don't replace them. The notes under a category are rules; [rubric.md](rubric.md) has the full set.

## Training math: what the model learns from each step

- **`loss`: Loss function.** Loss code, KL, clipping, token weighting. Complexity `medium` by default; `high` at about 200+ source lines. Often paired with `fix`.
- **`advantage`: Advantage calculation.** Advantage or estimator code, reward normalization. Same rules as `loss`.
- **`lowprec`: Low precision.** FP8, FP4, MXFP8, quantization, modelopt. It changes numbers quietly, like a loss change. Convergence risk `high`.

## Algorithm: how the RL loop runs

- **`singlectl`: Single controller.** Rollout manager, async utilities, overlap and barriers (everything once called "rollout orchestration", except samplers). A source change here is at least `medium`, even when small.
- **`sampler`: Samplers / async RL.** Ready-first vs in-order sampling, staleness, and KV or prefix cache kept or dropped across pause and resume. Invalidating the KV cache changes async RL behavior, so it belongs here.
- **`newalgo`: New algorithm.** A new training method or update rule (e.g. MOPD; LoGRA, which replaces the dense AdamW update with low-rank gradient sketches, is `newalgo` + `perf`). LoRA adapters are not a new algorithm; they go under `backend`.

## Generation and refit

- **`generation`: Generation backend.** vLLM, Dynamo, SGLang, TRT-LLM.
- **`refit`: Refit.** Moving weights from training to generation. A source change is at least `medium`, even when small. A recipe-only switch of a refit setting (e.g. `refit_transport` in a YAML) is `recipe`, not `refit`.

## Fault tolerance: surviving failures without changing results

- **`rolloutckpt`: Rollout checkpointing.** Partial checkpointing of rollouts and recovery from it. Model checkpoints go under `backend`; data-plane checkpoints go under `dataplane`.
- **`resilience`: Resiliency.** What happens when a worker dies: retries, condemning bad workers, health checks, routing around failed generation workers. It can change convergence (are samples dropped, duplicated or retried forever?) and performance.

## Environments: where rewards come from

- **`gym`: Environments / Gym.** NeMo Gym integration, reward environments.
- **`agentic`: Agentic training.** Multi-harness training, context compaction, sub-agents, forest training. A fix in one of these hard areas can still be `low` when it fixes a plain functional failure.

## Data: what training sees and how it gets there

- **`dataplane`: Data plane.** Data-plane interfaces, adapters, token sink, data-plane checkpoints.
- **`dataload`: Data loading and packing.** Sequence packing, dynamic batching, the Energon data loader.
- **`multimodal`: Multimodal.** Image and video inputs, vision encoders. Usually paired with another category.
- **`chattmpl`: Chat template / message handling.** Prefix chat templates, how messages are tokenized, the OpenAI server utilities. Almost always `medium` to `high` complexity: it decides what tokens training sees.

## Models and backends

- **`newmodel`: New model / recipe.** Adds a model or recipe config following the existing pattern. Proof: `convergence` and/or `perf`.
- **`backend`: Training backend.** Megatron, DTensor / Automodel internals, parallelism and sharding layouts, LoRA adapters, and model checkpointing (saving and loading weights and optimizer state, storage backends, async saves). Model checkpoint save/restore is at least `medium`.
- **`recipe`: Recipe change.** Edits an existing recipe or nightly config (YAML) without new code: switching a backend or transport, tuning a setting, changing test thresholds. Also tag the area of the setting. Switching `refit_transport` or the MoE backend is also `perf`, since backends are meant to be functionally the same.

## Kind of change

- **`fix`: Fix.** Fixes a bug. Usually paired with an area.
- **`perf`: Perf only.** Faster, less memory, offload, faster startup, with no functional change claimed. Never benign; always needs `perf` proof.
- **`config`: New config values.** Adds config knobs. Off by default does not lower the calls, and the config's shape needs design thought.
- **`refactor`: Refactor.** Moves or reshapes code with no behavior change meant (e.g. a config dict becoming a Pydantic model).
- **`cleanup`: Cleanup.** Renames or removes keys, dead code, small internal renames.
- **`breaking`: Large breaking change.** Only huge, obvious ones: deleting a whole backend or a major feature. Small renames never count. Not bad in itself, but maintainers must know.

## Infra and tooling: around the run, not inside it

- **`ci`: CI.** Workflows, test lanes, labels.
- **`testsonly`: Tests only.** Adds or fixes tests, no source change.
- **`deps`: Dependency bump.** uv.lock, pyproject, 3rdparty submodules, the container. Convergence `high`; proof all three. A lockfile change riding along with a small feature is not a bump.
- **`launchers`: Launchers.** ray.sub, k8s, Slurm, launch scripts.
- **`logging`: Logging / metrics.** W&B, telemetry, tracing, profiling. Usually benign, but large tracing features are not automatically low complexity.
- **`eval`: Evaluation.** Eval and validation loops, eval-only scripts.
- **`docs`: Docs / skills.** Docs, `skills/`, AGENTS.md. Judge the content: reflow and moves stay `low`; a rewrite that needs every page checked is `high`.

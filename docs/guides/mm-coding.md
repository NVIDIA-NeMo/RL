# Multimodal coding with NeMo Gym and Megatron

The MM-coding adapter connects Chart2Code, MMSVG Img2SVG/Text2SVG, Image2Struct LaTeX and Design2Code to the existing Megatron learner and vLLM generation path. Single-answer rows use Gym's simple-agent relay. Multi-turn rows use the public OpenCode environment-server interface. Mixed datasets contain both routes. Codex and Hermes are outside this integration's supported routes.

This is a task integration, not a model implementation. It does not add NT4 model registration, change attention/indexer settings, alter the GRPO loss or implement an Automodel learner. An NT4 deployment must supply its independently validated NT4/Megatron/vLLM model recipe and runtime. A successful Super VL/Automodel experiment does not qualify that NT4 deployment.

## Prepare training rows

Source acquisition and approved train/validation split selection remain with the data owner. The adapter accepts prepared `nemo-rl-mm-coding` task records:

```bash
uv run examples/nemo_gym/mm_coding/prepare_mm_coding.py \
  --input /approved/train-tasks.jsonl --output /shared/coding/train.jsonl \
  --reference-root /shared/coding/assets --split train \
  --mode mixed --max-output-tokens 16384
```

Run separately for a disjoint approved validation split. Use `--input-format legacy-row` for source experiment envelopes; their task metadata is preserved, while policy input is rebuilt without old harness instructions or completions. The adapter verifies image bytes, rejects duplicate identities, rejects unsupported harnesses and refuses to overwrite output. It cannot establish that a caller-labeled split excludes held-out evaluation data.

Single-answer rows carry `problem_id=<benchmark>:<source-task-id>` so training does not depend on an eval collector's positional index. Existing datasets must be regenerated or converted with Gym's `resources_servers.mm_coding_training.prepare_task_ids` tool. The Gym revision must support this identity field. OpenCode rows already carry typed `task_id`/`task_input` envelopes; do not flatten them into the single-answer schema. Private reference code belongs in verifier task metadata, never in the policy request.

OpenCode is the only multi-turn harness selected by default or accepted explicitly. Its system instruction explains that only the self-contained final answer file reaches a fresh grading sandbox. Regenerate older rows that lack that instruction; changing a route alone cannot repair an old prompt. Source inputs, sampling limits and the selected Gym routes must agree before training.

## Compose the model and task configurations

Set an **absolute path** to a validated Megatron/vLLM base recipe. This base owns the checkpoint, tokenizer/processor, model architecture, parallelism, precision, attention settings, generation sampling and node allocation. The MM-coding configuration adds SingleController defaults, enables token capture, selects Megatron/vLLM and replaces the Gym environment bindings. It does not select a private NT4 checkpoint or change SSA to dense attention.

```bash
export MM_CODING_BASE_CONFIG=/absolute/path/to/validated-megatron-vllm.yaml
export MM_CODING_TRAIN_DATA=/shared/coding/train.jsonl
export MM_CODING_VALIDATION_DATA=/shared/coding/validation.jsonl
export MM_CODING_REFERENCE_ROOT=/shared/coding/assets
export MM_CODING_MAX_NEW_TOKENS=16384
```

The required output budget must match prepared rows and leave room for input within `policy.max_total_sequence_length`. Choose the base's node allocation explicitly; preparing these files allocates no GPUs. The repository's `examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-video.v1.yaml` is one public configuration used to validate composition. It describes a different model and is not an NT4 training certificate.

Choose `examples/nemo_gym/mm_coding/mm_coding_single.yaml`, `mm_coding_multi.yaml` or `mm_coding_mixed.yaml`. Single permits two images per request; OpenCode multi/mixed permit 32 for retained reference/preview images. The total context limit still applies. SingleController validation is disabled because it does not yet support a validation loop; the explicit validation path prevents accidentally substituting an evaluation set when moving to a trainer that does.

Install a reviewed public Gym revision containing `mm_coding_training`, its session/sandbox-lending APIs, the OpenCode harness-note token-capture support (Gym #5833 or its merged successor), and the training task-ID fix (Gym #6451 or its merged successor). The RL submodule pin alone may not contain all of these. Provision that revision and its dependencies in each deployment-owned driver/Gym/generation interpreter; merely adding a resource root does not upgrade core APIs. Use a separate actor environment cache via `NEMO_RL_VENV_DIR`, prebuild it with `nemo_rl/utils/prefetch_venvs.py`, verify each interpreter with `uv pip check --python <interpreter>`, and retain exact source/dependency pins. Do not modify another run's environments or skip dependency checks.

Run discovery in every relevant interpreter and validate the resolved training configuration in the driver:

```bash
uv run examples/nemo_gym/mm_coding/check_mm_coding_runtime.py --mode mixed \
  --config examples/nemo_gym/mm_coding/mm_coding_mixed.yaml
```

Discovery and schema validation do not prove kernel/model compatibility, credentials, network access, captured-token parity or optimizer correctness. Deploy the remote OpenSandbox provider, pinned OpenCode/runtime files and trusted quality judge according to Gym's MM-coding documentation. Keep grading and registry credentials on the trusted Gym side; do not place them in policy input or candidate environments. Preserve `model_only` task egress and the final-file transfer boundary.

Launch within the approved allocation using the existing entrypoint:

```bash
uv run examples/run_grpo_single_controller.py \
  --config examples/nemo_gym/mm_coding/mm_coding_mixed.yaml
```

Typed multi-turn results need captured model-call tokens, images and logprobs. The legacy `run_grpo_nemo_gym.py` entrypoint is not a replacement for this capture path. The overlay disables multimodal deduplication because captured media and token positions must remain aligned. It keeps Gym's existing reward and infrastructure masking behavior.

## Train and recovery acceptance

Inside an approved allocation, select the matching mode and adapted data, then use a new output directory:

```bash
export MM_CODING_ACCEPTANCE_ROOT=/shared/acceptance/new-attempt
uv run --no-sync bash tests/functional/mm_coding_train_resume.sh mixed
```

The helper runs three training steps followed by a fourth step in a fresh driver process, using the same native checkpoint root and data. It uses `CheckpointManager.get_resume_paths` to detect saved optimizer state, including Megatron's embedded distributed optimizer format. It does not require Automodel's separate optimizer directory. Preserve full model, optimizer/scheduler, dataloader and TQ/replay state; recovery uses the native sibling boundary, not in-flight typed-episode recovery.

Two successful processes and checkpoint discovery are necessary but insufficient acceptance evidence. Audit real native restore logs and optimizer/scheduler counters, a finite nonzero update, unchanged source/model/runtime pins, image/token/logprob bindings, masks and data cursor/replay continuity. Do not label CPU configuration checks, synthetic environment smoke or historical Automodel runs as NT4/Megatron learner acceptance. Report each mode's actual learner/recovery result independently.

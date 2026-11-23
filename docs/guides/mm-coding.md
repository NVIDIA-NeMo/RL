# Multimodal coding with NeMo Gym and Megatron

This integration prepares visual-coding training rows and connects them to an existing Megatron learner with vLLM generation. It supports single-answer tasks, OpenCode multi-turn tasks, and datasets that mix both routes. The task families are Chart2Code, MMSVG Img2SVG/Text2SVG, Image2Struct LaTeX and Design2Code. Codex and Hermes are not supported by this adapter.

## Prepare rows

Provide separate training and validation JSONL files. Each source record has these fields:

| Field | Meaning |
| --- | --- |
| `benchmark` | `chart2code`, `mmsvg_img2svg`, `mmsvg_text2svg`, `image2struct_latex` or `design2code_full`. |
| `task_id` | Source task identity, from 1 to 256 characters. |
| `prompt` | Original task text sent to the policy. |
| `references` | One PNG/JPEG/WebP input image under `--reference-root`; an empty list for Text2SVG. |
| `semantic_prompt`, `ground_truth` | Optional verifier metadata, excluded from policy input. |

The adapter preserves prompt text and image bytes. It checks image bindings, rejects duplicate tasks and refuses to overwrite an output file. Dataset acquisition and split selection remain with the caller. OpenCode instructions require a self-contained final answer file that can run in a fresh grading sandbox.

```bash
uv run examples/nemo_gym/mm_coding/prepare_mm_coding.py \
  --input train-tasks.jsonl --output coding/train.jsonl \
  --reference-root coding/assets --split train --mode mixed
```

Repeat for validation with `--split validation`. Use `--input-format legacy-row` for a flat input envelope containing `task`, `route`, `agent_ref` and `responses_create_params`; only its task metadata and verified image binding are reused. Single-answer outputs remain flat, while OpenCode outputs carry typed task envelopes. NeMo RL assigns `_ng_task_index` when dispatching rows; Gym uses that index when a flat row has no top-level task ID. Original source identities remain in `task.task_id`.

Rows carry no output-token limit. Both routes use the run's `policy.generation.max_new_tokens`, so changing the training output budget does not require rebuilding the data.

## Compose a recipe

Set an absolute path to a validated Megatron/vLLM model recipe. It supplies the model, processor, precision, parallelism, sampling and allocation settings. Choose the matching MM-coding overlay: `mm_coding_single.yaml`, `mm_coding_multi.yaml` or `mm_coding_mixed.yaml` under `examples/nemo_gym/mm_coding/`. The overlays set the following length budgets; size the model runtime and allocation for them before launch.

| Budget | Default | Contents |
| --- | --- | --- |
| `policy.generation.max_new_tokens` | 32,768 | Maximum output for each model call. |
| `policy.max_total_sequence_length` | 65,536 | Input, image tokens, retained multi-turn history and output. |
| `policy.generation.vllm_cfg.max_model_len` | `${policy.max_total_sequence_length}` | The generation server's total context. |

```bash
export MM_CODING_BASE_CONFIG=/absolute/path/to/your-megatron-vllm.yaml
export MM_CODING_TRAIN_DATA=/absolute/path/to/coding/train.jsonl
export MM_CODING_VALIDATION_DATA=/absolute/path/to/coding/validation.jsonl
export MM_CODING_REFERENCE_ROOT=/absolute/path/to/coding/assets
# Optional: the recipe uses this output limit when the variable is unset.
export MM_CODING_MAX_NEW_TOKENS=32768
```

The output budget must leave input space within `policy.max_total_sequence_length`. vLLM caps each call's output to the remaining context. Multi and mixed set OpenCode's context window to that full sequence length and its output limit to `policy.generation.max_new_tokens`. A 32,768-token output limit does not guarantee that every call has that much context left. Single allows two images per request; OpenCode allows 32 to retain reference and preview images. The overlays use SingleController token capture, disable multimodal deduplication because captured rows carry their own media, and preserve Gym's reward and infrastructure masking behavior.

Override the native fields on both the preflight and training commands for a different budget. For example, a shorter smoke can use `policy.max_total_sequence_length=16384 policy.generation.max_new_tokens=8192`; the server and OpenCode context limits follow the total sequence length. Passing a short smoke does not validate the default 32,768/65,536-token budget or checkpoint recovery.

## Install the Gym environment

RL's Gym submodule pin does not contain `mm_coding_training`. Use a Gym commit containing [Gym #6091](https://github.com/NVIDIA-NeMo/Gym/pull/6091), for example `041060103fc3fc441eb17142b97c91d4133f34d2`. Check out that commit under `3rdparty/Gym-workspace/Gym`, or mount it at that path in the container:

```bash
git -C 3rdparty/Gym-workspace/Gym fetch origin 041060103fc3fc441eb17142b97c91d4133f34d2
git -C 3rdparty/Gym-workspace/Gym checkout --detach 041060103fc3fc441eb17142b97c91d4133f34d2
```

Gym's dependencies must match RL's `uv.lock`, because RL starts Gym actors with `uv run --locked`. If a selected Gym revision changes those dependencies, run `uv lock` in RL and rebuild the affected environments first. `NEMO_GYM_EXTRA_ROOTS` adds resource roots; it does not upgrade Gym's core code. Set `NEMO_RL_VENV_DIR` to a separate actor-environment cache, prebuild it with `nemo_rl/utils/prefetch_venvs.py`, and verify each interpreter with `uv pip check --python <interpreter>`. Keep exact source and dependency pins.

Configure the remote provider, renderer, OpenCode runtime and quality judge using [Gym's MM-coding README](https://github.com/NVIDIA-NeMo/Gym/blob/041060103fc3fc441eb17142b97c91d4133f34d2/resources_servers/mm_coding_training/README.md). Keep judge and registry credentials on the trusted Gym side. Preserve the `model_only` task network and final-file transfer boundary.

## Check and launch

The runtime check needs Gym installed in its interpreter. Pass the same `key=value` overrides to the check and the training command:

```bash
overrides=(logger.wandb_enabled=false)
uv run --extra nemo_gym python examples/nemo_gym/mm_coding/check_mm_coding_runtime.py \
  --mode mixed --config examples/nemo_gym/mm_coding/mm_coding_mixed.yaml "${overrides[@]}"
uv run --no-sync python examples/run_grpo_single_controller.py \
  --config examples/nemo_gym/mm_coding/mm_coding_mixed.yaml "${overrides[@]}"
```

The check validates configuration overrides and module discovery. It does not validate model kernels, service credentials or network reachability. Typed multi-turn training needs captured model-call tokens, images and logprobs; use the SingleController entrypoint shown above.

## Save and resume helper

Inside an existing GPU allocation, set `MM_CODING_ACCEPTANCE_ROOT` to a new output directory and run:

```bash
bash examples/nemo_gym/mm_coding/train_resume.sh mixed
```

The helper saves the first training step and stops on a one-second checkpoint deadline. It resumes to step four in a fresh driver process, with the same native checkpoint root, data and total step count. It allows four epochs so a small split with at least one full prompt batch can reach step four. Set `MM_CODING_CHECKPOINT_ROOT` to select a different checkpoint root. Before starting training, the preflight checks both complete save/resume configurations, including their phase-specific logging and checkpoint settings, and rejects overrides that redirect either phase away from the guarded and inspected directory. Each training process reuses the same arguments as its preflight. The helper uses the native checkpoint inspector for embedded Megatron optimizer state. Its per-step checkpoint setting is local to this short run; normal runs retain the base recipe's save interval. This operator helper is not an automated CI training certificate.

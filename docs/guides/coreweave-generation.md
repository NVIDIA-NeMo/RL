# CoreWeave Dedicated Inference

Use an existing CoreWeave deployment to generate rollouts while NeMo-RL uses the training cluster's GPUs for the policy. NeMo-RL exports Hugging Face safetensors checkpoints and waits for CoreWeave to load them on every replica before the next round.

This draft supports synchronous text GRPO with the stock Automodel policy worker and `examples/run_grpo.py`. NeMo-Gym integration is developed in [Gym PR #5560](https://github.com/NVIDIA-NeMo/Gym/pull/5560). The RL-to-Gym wiring is still pending. SingleController, TransferQueue rollouts, Megatron, LoRA, VLMs, quantized exports, draft training, and routing replay are not supported yet. No real-service validation has been run for this port.

## Setup

Prepare a hot-load-enabled `dynamo-vllm` deployment using the [CoreWeave RL Rollouts guide](https://docs.coreweave.com/products/inference/rl-rollouts). The deployment and trainer must use compatible model weights and tokenizer. The completions endpoint must accept integer prompt IDs and return exact generated IDs with sampled-token logprobs, including stop tokens.

Install the optional cookbook client into the driver environment after building NeMo-RL's environments:

```bash
uv pip install --python /path/to/driver/venv/bin/python \
  'cwtrain @ git+https://github.com/coreweave/inference-cookbook.git@41f991fef0d45430d20cfd8bc142c4bfc5070369#subdirectory=libs/cwtrain'
```

The policy workers use NeMo-RL's exporter and do not need cwtrain. Use the frozen driver environment when launching so dependency sync does not remove the optional package. The adapter uses the cookbook's NeMo-RL transport and synchronizer API at the pinned commit; dependency packaging remains a draft review item.

Set the recipe's `CWTRAIN_ROLLOUT_BASE_URL`, `CWTRAIN_ROLLOUT_MODEL`, `CWTRAIN_HOTLOAD_DEPLOYMENT_ID`, `CWTRAIN_HOTLOAD_RUN_ID`, and `CWTRAIN_HOTLOAD_STATE_DIR`. Use a unique run ID and a local directory containing the model checkpoint for `policy.model_name`. The driver and policy rank zero must share the staging directory, including across nodes.

Provide a mounted gateway key file through `CWTRAIN_ROLLOUT_API_KEY_FILE`, a management token through `COREWEAVE_API_TOKEN`, and object-storage credentials as described by cwtrain. Do not put credential values in YAML. The trainer needs access to upload and read its checkpoint prefix; the inference service needs read access. Keep one trainer per deployment.

Launch the recipe from the prepared environment:

```bash
uv run --frozen --no-sync examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen2.5-1.5b-1n8g-automodel-coreweave.yaml \
  policy.model_name=/shared/models/Qwen2.5-1.5B-Instruct
```

## Settings

`policy.generation.coreweave_cfg` uses a typed config. Defaults live in `CoreWeaveConfig`.

| Key | Purpose |
| --- | --- |
| `base_url`, `model` | OpenAI-compatible `/v1` base URL and served model name. |
| `deployment_id`, `run_id` | Deployment to update and unique checkpoint identity prefix. |
| `state_dir` | Shared directory for exports and checkpoint manifests. |
| `max_model_len` | Deployment limit for prompt plus response tokens. |
| `api_key_file` | Optional mounted gateway credential file. |
| `concurrency`, `max_attempts` | Maximum parallel requests and bounded request attempts. |
| `timeout` | Request and hot-load activation timeout in seconds. |
| `full_every` | Full checkpoint interval; 1 sends full checkpoints every round. |
| `upload_workers` | Parallel checkpoint upload workers. |
| `wait`, `verify`, `require_logprobs` | Must remain true for this training path. |
| `prompt_cache_policy` | Must be `reset_all` so new rounds do not reuse old KV state. |
| `importance_sampling` | Cookbook logprob compatibility check; training uses temperature 1, top-p 1, and no top-k. |

The initial sync publishes the loaded/resumed policy before sampling. Failed or partial hot-loads raise and stop the round; they do not commit a new served identity. Retrying an existing checkpoint identity requires matching stored checksums. Never delete checkpoint ancestors that a published delta still needs. Deployment creation and deletion stay outside NeMo-RL.

Metrics count zero generation GPUs in the training cluster. They do not estimate remote GPU capacity. Check export, upload, hot-load, and generation times before choosing a delta interval.

Related design: [NeMo-RL issue #4964](https://github.com/NVIDIA-NeMo/RL/issues/4964). The exporter is adapted from the Apache-2.0 CoreWeave inference cookbook and retains its attribution.

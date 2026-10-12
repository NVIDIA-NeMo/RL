# External vLLM rollout and HF-checkpoint refit

SingleController GRPO can use a vLLM service outside the training Ray cluster.
NeMo-Gym sends OpenAI-compatible rollout requests directly to the service;
the controller exports live Megatron policy weights to shared storage and
reloads the service between policy versions.

## HF-checkpoint transport

Select this transport for managed, non-colocated async vLLM or external vLLM:

```yaml
policy:
  generation:
    refit_transport: hf_checkpoint
    refit_cfg:
      hf_checkpoint:
        checkpoint_dir: /shared/hf_exports
```

Every policy rank participates in Megatron-Bridge export. The synchronizer
publishes each completed export under a unique `version_XXXXXXXX` directory,
pauses generation with cache clearing, reloads all engine workers, and resumes
only after reload succeeds. A failed reload leaves generation paused. Exports
are retained; use a separate directory per training run and remove old exports
after the serving processes have stopped.

The initial implementation supports Megatron policies and non-colocated
generation. Managed vLLM requires `vllm_cfg.async_engine=true`. Quantized
generation, draft-policy refit, and multimodal encoder-cache reset are not
supported by this transport.

## Start an external service

Allocate serving GPUs separately from the training Ray cluster. Use vLLM
0.29.0 for the prefix/capture plugin; it is version-pinned because it wraps
internal renderer and serving APIs. The training environment retains the
repository's own backend dependency pins.

For a two-GPU local smoke, start serving on GPU 1 in the external vLLM
environment:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --served-model-name policy --port 19210 --max-model-len 512 \
  --gpu-memory-utilization 0.7 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Development control endpoints must be reachable by the controller on a private
network. Setup checks health, the served model alias, and `/server_info`.
Refit uses `/pause`, `/collective_rpc` with `reload_weights(weights_path=...)`,
and `/resume`. This example supports one engine deployment, including its
native TP/PP/DP workers. Independent replicas, P/D disaggregation, and router
control fan-out are outside this stack.

Prepare Gym's workplace-assistant data using the existing
[Gym setup instructions](https://docs.nvidia.com/nemo/gym/0.1.0/tutorials/nemo-rl-grpo/setup.html#training-nemo-rl-grpo-setup).
The functional smoke trims the tool list and prepares a small dataset
automatically. For the public config, provide prepared JSONL files:

```bash
export EXTERNAL_ROLLOUT_VLLM_URL=http://127.0.0.1:19210/v1
export EXTERNAL_ROLLOUT_HF_EXPORT_DIR=/shared/hf_exports/run1
export GYM_TRAIN_PATH=/shared/data/workplace_assistant_train.jsonl
export GYM_VALIDATION_PATH=/shared/data/workplace_assistant_validation.jsonl

CUDA_VISIBLE_DEVICES=0 uv run --extra mcore --extra nemo_gym \
  examples/run_grpo_single_controller.py \
  --config examples/configs/grpo_external_vllm_single_controller.yaml
```

The shared export directory must have the same absolute path in both
containers. On multiple hosts, use a service address reachable from Gym and
the controller rather than loopback. The launcher owns service startup and
teardown; NeMo-RL does not terminate external vLLM.

## Multi-turn token modes

TransferQueue carries SingleController training data in all modes. These modes
determine how exact tokens enter that data plane:

| Mode | Prefix source | Token transport |
|---|---|---|
| Ordinary HTTP | Rendered conversation text | Inline Gym responses |
| Exact inline prefix | Gym-provided token IDs | Inline Gym responses |
| External capture | Inline admission or staged-chain lookup | Token deltas staged through the controller bridge; Gym carries commit coordinates |

Ordinary HTTP can run single-turn and text-based multi-turn rollout, but
re-rendering text does not guarantee preservation of earlier generated token
IDs. For exact continuity, build and expose the endpoint plugin before
starting the external service:

```bash
PLUGIN_WHEEL=$(bash tools/external_rollout_vllm/build_prefix_plugin.sh /shared/plugin)
export PYTHONPATH="$PLUGIN_WHEEL:${PYTHONPATH:-}"
export VLLM_PLUGINS=nemo_rl_prefix_api
# Start vllm serve with the same options shown above.
```

Check `/v1/nemo-rl/prefix-token-capability` for activation. Enable inline
prefixes with this override:

```text
++env.nemo_gym.policy_model.responses_api_models.vllm_model.supply_prefix_token_ids=true
```

For external capture, set:

```text
token_capture.enabled=true
++env.nemo_gym.policy_model.responses_api_models.vllm_model.return_token_id_information=false
++env.nemo_gym.policy_model.responses_api_models.vllm_model.supply_prefix_token_ids=false
```

The controller creates an authenticated HTTP bridge and configures the plugin.
The plugin fetches staged prefixes when needed, preserves exact engine input
tokens, and waits for a staging acknowledgement before returning commit
coordinates to Gym. The existing finalizer verifies and assembles the staged
records. Capture uses non-streaming chat completion responses.

## Functional tests

Run inside an isolated two-GPU allocation with the required Megatron, Gym,
TransferQueue, and vLLM environments available. For the external tests,
`EXTERNAL_VLLM_TEST_PYTHON` identifies the vLLM 0.29.0 environment's Python:

```bash
uv run --extra mcore --extra vllm bash tests/functional/grpo_hf_checkpoint_refit.sh

export EXTERNAL_VLLM_TEST_PYTHON=/path/to/vllm-environment/bin/python
uv run --extra mcore --extra nemo_gym bash tests/functional/grpo_external_vllm_gym_single_controller.sh
uv run --extra mcore --extra nemo_gym bash tests/functional/grpo_external_vllm_prefix_gym.sh
uv run --extra mcore --extra nemo_gym bash tests/functional/grpo_external_vllm_capture_gym.sh
```

The external fixtures launch a separate server, wait for readiness, run the
existing small-model SC/Gym training smoke through repeated refits, and clean
up the server. The capture variant also checks that the finalizer ran.

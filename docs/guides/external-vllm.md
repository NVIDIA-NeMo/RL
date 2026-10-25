# External vLLM rollout and NCCL refit

SingleController GRPO can use a vLLM service outside the training Ray cluster.
NeMo-Gym sends OpenAI-compatible rollout requests directly to the service;
the controller transfers live Megatron policy weights directly to serving GPU
workers over NCCL between policy versions. HTTP carries control and metadata.

## NCCL transport

Select the external backend with its native NCCL refit configuration:

```yaml
policy:
  generation:
    backend: remote_vllm
    refit_transport: null
    remote_vllm_cfg:
      base_url: http://serving-host:19210/v1
      served_model_name: policy
      max_model_len: 8192
      refit:
        transport: nccl
        nccl_num_engines: 1
        nccl_engine_world_size: 1
```

Every policy GPU worker and every serving GPU worker joins one NCCL group.
The synchronizer pauses generation with cache clearing, starts the native
weight update, broadcasts HF-named tensors while vLLM receives them, commits
the weight version, and resumes. HF tensor conversion still runs, but checkpoint
files and a shared export directory are unnecessary. A failed update leaves
generation paused; restart both training and serving before retrying.

This path requires a Megatron policy and non-colocated external generation.
ModelOpt quantization, trainer-refitted draft weights, separate KV-scale
transfer, and fleet-health monitoring are unsupported. Set
`nccl_engine_world_size` to the deployment's TP * PP * DP GPU count; the
frontend and controller do not participate in NCCL. See [the refit guide](refit.md)
for deadlines, packing settings, and failure behavior.

## Start an external service

Allocate serving GPUs separately from the training Ray cluster. Use vLLM
0.29.0 for the prefix/capture plugin; it is version-pinned because it wraps
internal renderer and serving APIs. The training environment retains the
repository's own backend dependency pins.

For a two-GPU local smoke, start serving on GPU 1 in the external vLLM
environment:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --served-model-name policy --port 19210 --max-model-len 8192 \
  --gpu-memory-utilization 0.7 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --weight-transfer-config '{"backend":"nccl"}'
```

Development control endpoints must be reachable by the controller on a private
network. Setup checks health, the served model alias, vLLM 0.29.0, the NCCL
transfer backend, and GPU world size. Refit uses `/pause`, the native
`/start_weight_update`, `/update_weights`, and `/finish_weight_update` endpoints,
and `/resume`. This example supports one engine deployment, including its
native TP/PP/DP workers. Independent replicas, P/D disaggregation, and router
control fan-out are outside this stack.

Prepare Gym's workplace-assistant data using the existing
[Gym setup instructions](https://docs.nvidia.com/nemo/gym/main/tutorials/training-tutorials/nemo-rl-grpo/setup/).
The external functional tests keep the full tool list and prepare the public
dataset automatically. The public example uses ten steps, one agent call per
sample, and Reinforce++ without a per-prompt baseline so mixed task rewards
produce actual policy updates even when siblings share a reward. For the public config, provide prepared JSONL files:

```bash
export EXTERNAL_ROLLOUT_VLLM_URL=http://127.0.0.1:19210/v1
export GYM_TRAIN_PATH=/shared/data/workplace_assistant_train.jsonl
export GYM_VALIDATION_PATH=/shared/data/workplace_assistant_validation.jsonl

CUDA_VISIBLE_DEVICES=0 uv run --extra mcore --extra nemo_gym \
  examples/run_grpo_single_controller.py \
  --config examples/configs/grpo_external_vllm_single_controller.yaml
```

The OpenAI base URL must end in exactly `/v1`, without a trailing slash or
path prefix. On multiple hosts, use a service address reachable from Gym and
the controller, and configure mutual NCCL connectivity between training and
serving GPUs. The launcher owns service startup and
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
prefixes by enabling Gym local capture for correlated parent lineage, while
keeping responses inline. Use a fresh capture directory for each run and
increase the agent call limit:

```text
++env.nemo_gym.token_id_capture.enabled=true
++env.nemo_gym.token_id_capture.all_agents=true
++env.nemo_gym.token_id_capture.dir=/shared/gym-prefix-capture/run1
++env.nemo_gym.token_id_capture.rebuild_response=false
++env.nemo_gym.policy_model.responses_api_models.vllm_model.supply_prefix_token_ids=true
++env.nemo_gym.workplace_assistant_simple_agent.responses_api_agents.simple_agent.max_steps=6
```

For external capture, set:

```text
token_capture.enabled=true
++env.nemo_gym.workplace_assistant_simple_agent.responses_api_agents.simple_agent.max_steps=6
++env.nemo_gym.policy_model.responses_api_models.vllm_model.return_token_id_information=false
++env.nemo_gym.policy_model.responses_api_models.vllm_model.supply_prefix_token_ids=false
```

The controller creates an authenticated HTTP bridge and configures the plugin.
The plugin fetches staged prefixes when needed, preserves exact engine input
tokens, and waits for a staging acknowledgement before returning commit
coordinates to Gym. The existing finalizer verifies and assembles the staged
records. Capture uses non-streaming chat completion responses.
External capture currently supports text-only rollouts.

## Functional tests

Run inside an isolated two-GPU allocation with the required Megatron, Gym,
TransferQueue, and vLLM environments available. For the external tests,
`EXTERNAL_VLLM_TEST_PYTHON` identifies the vLLM 0.29.0 environment's Python:

```bash
export EXTERNAL_VLLM_TEST_PYTHON=/path/to/vllm-environment/bin/python
uv run --extra mcore --extra nemo_gym bash tests/functional/grpo_external_vllm_gym_single_controller.sh
uv run --extra mcore --extra nemo_gym bash tests/functional/grpo_external_vllm_prefix_gym.sh
uv run --extra mcore --extra nemo_gym bash tests/functional/grpo_external_vllm_capture_gym.sh
```

The external fixtures launch a separate server, wait for readiness, run at
least ten training steps through repeated refits, and clean up the server.
They require nonzero gradients, trainer parameter-sync timing records, and a
final native vLLM `/weight_info` version of at least ten. The timing records
exclude the NCCL transfer duration.
The prefix variant requires a multi-turn trajectory. The capture variant
requires finalized multi-call rows, token-in continuations, and zero invalid
or poisoned capture rows. Set `EXTERNAL_VLLM_TEST_BASE_URL` to reuse an already
running service, including a service in a separate container; start its plugin
before the prefix or capture tests. That mode preserves the training process
GPU assignment. The caller owns service teardown.

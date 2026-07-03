# Dynamo Integration

NeMo-RL supports both an external Kubernetes `DynamoGraphDeployment` (DGD) and
a fixed Ray-managed Dynamo deployment for Slurm. In both modes NeMo-RL owns
training and sends updated checkpoint-format weights to vLLM over a native NCCL
weight-transfer group.

Install the Dynamo operator and `DynamoGraphDeployment` CRD in the target
cluster before using this integration. `nrl-k8s` renders and applies DGD
resources, but does not install cluster-scoped Dynamo components.

## Architecture

`dynamo_cfg.deployment` selects the lifecycle owner:

- `external` (the default) preserves the DGD behavior described below. The
  operator owns the frontend and workers, and all Ray GPUs remain available to
  training.
- `ray` starts etcd and NATS on the Ray driver, reserves the configured
  inference placement groups, launches one `dynamo.vllm` process per fixed
  model-parallel group, and starts the frontend after the workers. This mode
  requires `policy.generation.colocated.enabled=false`.

The minimal managed configuration is:

```yaml
policy:
  generation:
    backend: dynamo
    dynamo_cfg:
      deployment: ray
      engine_world_size: 1
      dynamo_python: /opt/dynamo_venv/bin/python
      startup_timeout_s: 300
      namespace: null
      frontend_port: 0
      request_timeout_s: 900
      worker_args:
        tool_call_parser: null
        reasoning_parser: null
        exclude_tools_when_tool_choice_none: true
        enable_structural_tag: false
        structural_tag_scope: auto
        structural_tag_schema: auto
        custom_jinja_template: null
        endpoint_types: [chat, completions]
        extra_cli_args: []
      frontend_args:
        router_mode: round-robin
        router_reset_states: true
        extra_cli_args: []
    colocated:
      enabled: false
      resources:
        gpus_per_node: 1
        num_nodes: 1
```

When `namespace` is null, managed mode derives a sanitized value from
`SLURM_JOB_ID`. Service ports are selected automatically. Every worker receives
a unique `DYN_SYSTEM_PORT`, and fixed membership is frozen before NCCL
collective initialization. A dead actor, changed system URL, or changed worker
count aborts the run.

### Argument ownership

| Configuration source | Responsibility |
| --- | --- |
| Generation fields | Request-time temperature, top-p, top-k, stops, and token limits |
| `vllm_cfg` | Standard engine topology, dtype, memory, model length, load format, and eager mode |
| `vllm_kwargs` | Advanced native vLLM engine arguments |
| `dynamo_cfg.worker_args` | Dynamo chat/runtime behavior, including tool and reasoning parsers |
| Managed runtime | Model identity, namespace, discovery, request plane, RL/refit flags, ports, and endpoints |
| `extra_cli_args` | Exact argv escape hatch for otherwise unrepresented options |

Parser defaults are deliberately null. NeMo-RL does not translate
`vllm_cfg.http_server_serving_chat_kwargs.tool_parser` into a Dynamo parser;
the two parser registries have different names and contracts. Managed mode
normalizes `vllm_kwargs` underscores to hyphens, compact-JSON encodes compound
values, omits nulls, emits explicit positive/negative boolean flags, and rejects
duplicates or attempts to override managed and structured options. The final
argv is validated with `dynamo.vllm.args.parse_args` in the configured Dynamo
Python environment before the worker process starts.

Inherited `DYN_*` variables are scrubbed before launch. Only manager-owned
service settings are added back; cache/auth/logging variables, `NCCL_*`, and
explicit non-semantic `vllm_cfg.env_vars` remain available. Resolved argv and
relevant environment values are logged with credentials redacted.

## External Kubernetes deployment

The training Ray cluster and the DGD are separate Kubernetes workloads:

- NeMo-RL sends rollout requests to the DGD frontend's OpenAI-compatible HTTP
  endpoint.
- The DGD frontend `/health` response advertises vLLM workers that registered
  the `rl` endpoint.
- NeMo-RL connects every discovered worker to one NCCL group shared with the
  training policy.
- The DGD lifecycle remains owned by the Dynamo operator. NeMo-RL does not
  create, scale, or delete serving workers.

The generation configuration points to the DGD and describes each worker
engine's internal rank count:

```yaml
policy:
  generation:
    backend: dynamo
    dynamo_cfg:
      dgd_name: my-dgd
      engine_world_size: 1
      request_timeout_s: 900.0
```

`engine_world_size` is the number of vLLM ranks in each discovered worker
endpoint, including tensor- and pipeline-parallel ranks. All worker endpoints
must use the same value.

The DGD vLLM worker must enable the RL routes and native NCCL transfer backend:

```yaml
args:
  - --enable-rl
  - --weight-transfer-config
  - '{"backend":"nccl"}'
```

## Frontend resolution

`dynamo_cfg` supports two frontend forms:

- `dgd_name` derives
  `http://<dgd_name>-frontend.<namespace>.svc.cluster.local:<port>/v1`.
  This form is required for weight transfer because worker discovery uses the
  same frontend's `/health` endpoint.
- `frontend_url` is an explicit rollout URL for deployments outside the
  in-cluster naming convention. It supports generation, but not NCCL refit.

`namespace`, `frontend_port`, and `dyn_system_port` optionally override
the detected Kubernetes namespace, frontend port 8000, and worker admin port
9090.

## Fixed worker fleet

Collective membership is fixed for the lifetime of a training run.

At setup, NeMo-RL filters the frontend health response to entries with the
configured Dynamo namespace, `component: backend`, and `endpoint: rl`. It
deduplicates by `instance_id`, sorts the resulting workers, and records each
worker's system URL.

For (N) workers with `engine_world_size = E`:

```text
inference_world_size = N * E
world_size = train_world_size + inference_world_size
worker[i].rank_offset = train_world_size + i * E
```

Before every update or cache flush, NeMo-RL rediscovers workers and compares the
ordered `(instance_id, system_url)` list with the setup snapshot. A scale,
restart, removal, or address change fails the refit immediately. Restart the
training job to establish a new collective after changing DGD membership.

## NCCL initialization

The policy workers initialize their existing stateless NCCL process group.
Concurrently, NeMo-RL posts `init_weights_update_group` to every Dynamo worker
with `engine_rpc: init_weight_transfer_engine` and:

- the training master address and port;
- the worker-specific rank offset;
- the total training-plus-inference world size.

All initialization futures are awaited together, just like the normal
non-colocated vLLM backend.

## Weight update

`policy.prepare_refit_info()` provides ordered checkpoint metadata. NeMo-RL
serializes it as the vLLM native packed update description:

- `names`;
- `dtype_names`;
- `shapes`;
- `packed: true`.

Each refit launches the policy broadcast and all Dynamo receive operations
concurrently. The Dynamo worker transaction invokes these engine RPCs in order:

1. `start_weight_update(is_checkpoint_format=True)`
2. `update_weights(update_info=...)`
3. `finish_weight_update()`

Dynamo routes those calls through `collective_rpc` to all ranks in each vLLM
engine. NeMo-RL uses vLLM's 1 GiB packed-buffer size and two alternating CUDA
buffers for this path. The normal NeMo-RL vLLM backend retains its existing
configurable packed-buffer behavior.

Any worker or collective error fails the refit. Partial success is treated as
fatal because workers must never continue serving mixed policy versions.

## Generation and cache semantics

Dynamo follows the same scheduling contract as NeMo-RL's normal vLLM backend.
The trajectory collector decides when outstanding generation must drain; the
weight-transfer implementation does not introduce an extra pause.

The distributed-update route therefore allows NeMo-RL to request an unpaused
update and to suppress the route's implicit prefix-cache reset. Existing Dynamo
callers keep the safer defaults: paused updates and automatic cache reset.

When `recompute_kv_cache_after_weight_updates` is enabled, NeMo-RL calls
`DynamoGeneration.invalidate_kv_cache()`, which posts `flush_cache` to every
worker in the fixed fleet. When it is disabled, the integration does not add a
cache flush.

FP8 KV-scale synchronization and speculative-decoding auxiliary weights are
not part of this integration.

## Direct generation

Dynamo supports NeMo-RL's direct synchronous and asynchronous generation paths.
Token-ID requests are sent to the frontend completions endpoint. When
`vllm_cfg.expose_http_server` is enabled, a local token wrapper exposes the
OpenAI chat surface expected by NeMo-Gym and forwards tokenized requests to the
DGD.

## GB300 smoke test

The two-step DTensor TP1 Qwen2.5-1.5B smoke assets are colocated under the
Dynamo examples directory:

- `infra/nrl_k8s/examples/dynamo/V1/grpo_math_1b_dynamo_nccl.yaml`
- `infra/nrl_k8s/examples/dynamo/V1/grpo_math_1b_dynamo_nccl.gb300.infra.yaml`
- `infra/nrl_k8s/examples/dynamo/V1/qwen2_5_1_5b_gb300_nccl.dgd.yaml`

Validate and render the ephemeral workload before launching it:

```bash
RECIPE=infra/nrl_k8s/examples/dynamo/V1/grpo_math_1b_dynamo_nccl.yaml
INFRA=infra/nrl_k8s/examples/dynamo/V1/grpo_math_1b_dynamo_nccl.gb300.infra.yaml

nrl-k8s check "$RECIPE" --infra "$INFRA"
nrl-k8s run "$RECIPE" --infra "$INFRA" --rayjob --dry-run
nrl-k8s run "$RECIPE" --infra "$INFRA" --rayjob --no-wait
```

The RayJob owns an ephemeral RayCluster. The DGD is applied before the RayJob
and receives an owner reference so teardown follows the ephemeral run.

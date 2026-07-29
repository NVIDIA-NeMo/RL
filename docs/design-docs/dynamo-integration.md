# Managed Dynamo Generation

NeMo RL can own a fixed Dynamo vLLM fleet inside the same Ray allocation as
GRPO training. This integration is intended for Slurm. It does not discover or
manage external Dynamo or Kubernetes deployments.

## Runtime ownership

When `policy.generation.backend: dynamo`, the Ray driver starts and owns:

1. an ephemeral etcd server;
2. a NATS server with JetStream enabled;
3. one fixed Ray-scheduled `dynamo.vllm` worker group per inference resource
   group;
4. the Dynamo frontend.

Startup succeeds only after the frontend reports every fixed worker at both
its generation and RL endpoints and advertises the configured model. Shutdown
stops the frontend first, then worker actors, NATS, and etcd, and removes their
temporary state directories.

The managed backend must be non-colocated. `engine_world_size` is the tensor
parallel size multiplied by the pipeline parallel size of each vLLM worker.
Inference GPU resources are declared under
`policy.generation.colocated.resources`.

## Configuration

Defaults live in YAML. A complete minimal runtime section is:

```yaml
policy:
  generation:
    backend: dynamo
    dynamo_cfg:
      engine_world_size: 1
      namespace: null
      frontend_port: 0
      dynamo_python: /opt/dynamo_venv/bin/python
      startup_timeout_s: 600
      request_timeout_s: 900
      etcd_port: 0
      etcd_peer_port: 0
      nats_port: 0
      system_port_base: 29000
      metrics_include_prefixes: null
      metrics_exclude_prefixes: null
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
        tokenizer: default
        tokenizer_cache: false
        tokenizer_cache_bytes: 52428800
        router_mode: round-robin
        router_reset_states: true
        extra_cli_args: []
    vllm_cfg:
      tensor_parallel_size: 1
      pipeline_parallel_size: 1
    colocated:
      enabled: false
      resources:
        gpus_per_node: 1
        num_nodes: 1
```

Zero-valued service ports are allocated automatically. When `namespace` is
null, the runtime derives a sanitized namespace from `SLURM_JOB_ID`. Every
worker group receives a unique `DYN_SYSTEM_PORT`. Inherited `DYN_*` variables
are removed before managed values are added.

`vllm_cfg` owns standard engine topology and dtype fields. `vllm_kwargs` owns
advanced vLLM engine arguments. `dynamo_cfg.worker_args` and
`dynamo_cfg.frontend_args` own Dynamo-specific behavior. Managed model,
namespace, discovery, endpoint, and refit flags cannot be overridden through
the raw argument escape hatches.

## Generation and refit

Both synchronous and asynchronous GRPO allocate a separate inference virtual
cluster for Dynamo. Direct token generation uses the frontend completions
endpoint. NeMo-Gym uses the local token wrapper, which translates chat
requests, preserves multi-turn prefix tokens, and forwards tokenized
completions.

The worker fleet is frozen before collective setup. For worker `i` with engine
world size `E`:

```text
inference_world_size = worker_count * E
worker_rank_offset = training_world_size + i * E
total_world_size = training_world_size + inference_world_size
```

Policy ranks and all vLLM ranks join one stateless NCCL group. Each weight
update drains pending generation first, sends checkpoint-format packed weights,
finishes vLLM's layerwise reload transaction, and then optionally invalidates
the KV cache. A dead actor or changed worker identity fails the update instead
of serving mixed model versions.

## Telemetry

When `vllm_cfg.enable_vllm_metrics_logger` is true, NeMo RL samples each fixed
worker's Prometheus endpoint and writes curated metrics under
`generation_metrics/*`. Include and exclude prefix lists can narrow the
published series. The normal logger routes those samples to enabled sinks,
including W&B, and stops the sampler during shutdown.

## Derived image

Dynamo is not a root Python extra. `docker/dynamo/Dockerfile` derives from the
standard NeMo RL release image and installs `ai-dynamo[vllm]==1.3.0.post1` into
an isolated Python 3.12 `/opt/dynamo_venv`. The normal NeMo RL Ray and vLLM
environments remain unchanged.

Dynamo 1.3.0.post1 pins vLLM 0.23.0. That vLLM release predates PR #44814,
which fixes layerwise reload accounting for composed weight loaders. Without
the fix, a refit can finalize a layer before trailing NemotronH/Mamba2
parameters are loaded. The image build asserts vLLM 0.23.0, runs
`git apply --check`, applies the backport, and records upstream merge commit
`c9e5bf813530fb9ce06024e075da0f520b0718c8` in
`/opt/dynamo_venv/VLLM_BACKPORTS`.

See `examples/slurm/dynamo/README.md` for the two-GPU smoke and the
parameterized six-node SWE/W&B acceptance run.

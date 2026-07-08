# Dynamo examples

These examples require the Dynamo operator and `DynamoGraphDeployment` CRD to
already be installed in the target cluster. `nrl-k8s` does not install them.

Each versioned directory contains the three files needed to run one NeMo-RL
example with Dynamo on Kubernetes:

- `<name>.yaml`: the NeMo-RL recipe.
- `<name>.<platform>.infra.yaml`: the Ray and Kubernetes topology. Its
  `dynamoGraphDeployment.manifest` field references the DGD file in the same
  directory.
- `<name>.dgd.yaml`: the DynamoGraphDeployment (DGD), including the frontend
  and vLLM worker configuration.

The current examples are:

- `V1`: Qwen2.5 1.5B DTensor GRPO on the math task.
- `V2`: Llama 3.1 8B Instruct Megatron async GRPO.
- `V3`: Qwen2.5 1.5B Instruct DTensor GRPO on the sliding-puzzle task.
- `V5`: Nemotron Nano v2 9B Megatron GRPO on the workplace-assistant task.

All four examples use Dynamo direct generation and vLLM's native NCCL weight
transfer on GB300. See each infra file's header for validation and launch
commands.

## WandB generation metrics

Dynamo worker metrics can be sampled from each worker's
`DYN_SYSTEM_PORT/metrics` endpoint and logged through the existing
`generation_metrics/*` WandB panels. Enable the existing generation metrics
logger in the NeMo-RL recipe:

```yaml
policy:
  generation:
    dynamo_cfg:
      # Optional. Omit for the curated Dynamo/vLLM metric set.
      metrics_include_prefixes:
        - dynamo_component_gpu_cache_usage
        - dynamo_component_inflight_requests
        - dynamo_work_handler_queue_depth
        - dynamo_component_requests_total
        - dynamo_work_handler_time_to_first_response
        - vllm:generation_tokens
        - vllm:prompt_tokens_total
        - vllm:inter_token_latency
    vllm_cfg:
      enable_vllm_metrics_logger: true
      vllm_metrics_logger_interval: 0.5

logger:
  wandb_enabled: true
```

Metrics sampling requires the `dgd_name` configuration path so NeMo-RL can
discover worker addresses through the DGD frontend. The supplied DGD manifests
already expose `DYN_SYSTEM_PORT=9090`. Set `metrics_include_prefixes: []` to
collect every metric family, or `metrics_exclude_prefixes: []` to disable the
default `python_` and `process_` exclusions. Collecting every family can
create a large number of per-worker WandB plots.

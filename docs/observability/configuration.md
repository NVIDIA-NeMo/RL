# Configuration

Telemetry is configured by the `telemetry:` block of your run config. Keep it there: a run's telemetry settings should be recoverable from the file that describes the run, not from whatever happened to be in a shell.

Two things do belong in the environment, because they describe *where* you are running rather than *what* you are measuring: the standard [`OTEL_EXPORTER_OTLP_*`](#standard-otel-sdk-variables) endpoint/protocol/headers, and `OTEL_SERVICE_NAME`.

## The `telemetry:` config block

`telemetry:` is an optional top-level field of every algorithm's `MasterConfig`. It is **documented here, not baked into the exemplar configs** — add it to your own run config.

```yaml
telemetry:
  enabled: false              # master switch; when false, every site is a ~0-cost no-op
  service_name: nemo-rl       # service.name reported to the backend
  span_groups: default        # preset (default | per_step | all) or a comma-separated group list
  traces_enabled: true        # emit trace spans
  metrics_enabled: true       # emit the rl.* metric instruments
  logs_enabled: false         # bridge Python logging to OTel logs (trace-correlated)
  exporter: otlp              # otlp | console
  vllm_native_tracing: false  # opt in to vLLM's own OTLP tracing (gRPC by default, one span per request — see vllm-tracing.md)
```

The defaults above are the field defaults of `TelemetryConfig` (`nemo_rl/telemetry/config.py`). The endpoint, headers, and protocol are **not** in this block — they come from the standard `OTEL_EXPORTER_OTLP_*` env vars (see below).

Every process that enables telemetry exports — driver, workers, and singleton actors alike. There is no rank-filtering setting; see [Which ranks export](#which-ranks-export).

`service_name` maps onto the standard `OTEL_SERVICE_NAME` (lens reads it unprefixed), so setting either works.

For the full config model, field semantics, and validation rules, see [lens: configuration](https://github.com/NVIDIA-NeMo/Lens).

### How the settings reach the workers

Ray actors do not inherit the driver's Python objects, so on the driver `init_telemetry_driver` projects the block into `NEMO_RL_OTEL_*` environment variables *before* `init_ray()`; the resulting environment is snapshotted into the Ray `runtime_env` and every worker rebuilds the same config from it.

These variables are a transport, not a second configuration interface. They are listed here so that a `NEMO_RL_OTEL_*` name in a log or a `ps` output is recognisable, and because two of them have no `telemetry:` equivalent:

| Variable | Meaning |
|---|---|
| `NEMO_RL_OTEL_RUN_ID` | Correlates the driver and every worker to one run. Generated from `SLURM_JOB_ID` or a random hex string when unset. |
| `NEMO_RL_OTEL_USER_ID` | Optional user/team label, read by lens. |

The projection uses `os.environ.setdefault`, so a variable already present in the environment wins over the YAML value. That is deliberate for the two above, which a job scheduler supplies. For every other setting, prefer the config: a NeMo-RL toggle set in a shell leaves no trace of who set it or why, and splits a run's configuration between a file and an environment with nothing recording which half came from where. The resolved settings are logged once at init for exactly this reason, and a hydra-style `++telemetry.<field>=<value>` override covers the one-off case without leaving the config record.

## Standard OTel SDK variables

Endpoint, protocol, and headers are honoured by the OTel SDK directly:

| Variable | Example |
|---|---|
| `OTEL_SERVICE_NAME` | `nemo-rl` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` or `http/protobuf` |
| `OTEL_EXPORTER_OTLP_HEADERS` | `<header>=<value>,<header>=<value>` (e.g. auth headers your backend requires) |

Pick the protocol to match your backend: a local collector or Jaeger typically speaks gRPC on `:4317`; a direct-to-SaaS OTLP endpoint typically speaks `http/protobuf` on `:443`. See [Observability Stack](observability-stack.md).

## Which ranks export

**Every process that enables telemetry exports.** nemo-lens has no notion of rank: it does not select among ranks and does not sample by rank, so NeMo-RL has no `export_strategy`, `export_rank`, `export_sample_rate`, or `sampler_enabled` setting. Each process instead labels itself, and narrowing the fleet down is a decision you make with those labels:

- **Filter in the collector.** Every span and metric carries `nv.dl.rank` and `nv.dl.world_size` as resource attributes. Drop or keep whichever ranks you want in an OpenTelemetry Collector processor — the collector is also the only place that sees the whole fleet at once, which is where a policy about it belongs.
- **There is no per-rank switch.** `telemetry.enabled` is one job-wide value that every worker inherits through the Ray `runtime_env`, so it cannot keep some ranks quiet while others export. Filter on `nv.dl.rank` in the collector instead. Setting it false does disable the whole job cheaply — a process with telemetry off gets an empty span-group set, so `is_span_group_enabled()` is `False` everywhere and no span objects are created at all.

`RANK` is **group-local**: the policy group and the generation group each number their workers from zero, so `nv.dl.rank` is only unambiguous together with the `rl.worker_group` attribute that names which group it belongs to. Filter on the pair.

If you previously set `export_strategy` or one of its companions, the key still parses — `TelemetryConfig` allows extra keys — but it no longer does anything. It is not projected into the worker environment either, so nothing downstream reads it.

## Run identification

Every run gets an id that flows to all backends as the **`nemo.run.id`** resource attribute, shared by the driver and every worker. Note the spelling: `run_id` is the config field you set, `nemo.run.id` is the attribute key lens emits and the one your backend indexes.

**Priority order:**

1. `NEMO_RL_OTEL_RUN_ID` (explicit, highest priority).
2. `SLURM_JOB_ID` (auto-detected on SLURM clusters).
3. Auto-generated 12-character hex id (fallback).

The run id is written to the environment on the driver **before** `init_ray()`, so every worker inherits the same value and correlates to the same run. This is also how vLLM's native spans are correlated back to the RL run — see [vLLM Tracing](vllm-tracing.md).

Filter by `nemo.run.id` in your backend to isolate a specific run.

## Resource attributes

Each process sets stable-for-the-run values on its OTel `Resource` — `init_telemetry_driver` on the driver, `init_telemetry_worker` in each Ray actor — so they appear on every span/metric as backend "Process" tags:

| Attribute | Source |
|---|---|
| `nemo.run.id` | the run id resolved above (`NEMO_RL_OTEL_RUN_ID` / `SLURM_JOB_ID` / generated). Emitted by lens from the `run_id` config field — see [Run identification](#run-identification) |
| `nv.dl.campaign.stage` | always the literal `RL`. Set by every NeMo-RL process, driver and worker, so a backend collecting several stages of a model's lifecycle (pretrain → SFT → RL) can select this stage without matching on service names |
| `rl.algorithm` | the `algorithm="<algo>"` passed to `init_telemetry_driver` |
| `rl.model` | `policy.model_name` |
| `nemo.precision` | `policy.precision` |
| `nv.dl.provider.name` | always the literal `nemo-rl`: which training framework produced the telemetry, for a backend collecting several. Separate from the service name, which a user may rename |
| `nv.dl.run.uuid` | the same run id as `nemo.run.id`. Duplicated on purpose — a backend joining RL runs to Megatron ones groups on the `nv.dl.*` spelling |
| `nv.dl.topology.size.tp` | `policy.megatron_cfg` / `automodel_cfg` TP size |
| `nv.dl.topology.size.pp` | `policy.megatron_cfg` PP size |
| `nv.dl.topology.size.dp` | derived: `cluster.num_nodes × cluster.gpus_per_node ÷ (TP × PP × CP)`. Omitted when `policy.generation.colocated.enabled` is false, because generation then holds ranks the policy does not have, and when the product does not divide the world size |
| `nv.dl.training.config.global_batch_size` | `policy.train_global_batch_size` |
| `nv.dl.training.config.micro_batch_size` | `policy.train_micro_batch_size` |
| `nv.dl.training.config.sequence_length` | `policy.max_total_sequence_length` |
| `nv.dl.training.config.optimizer` | `policy.megatron_cfg.optimizer.optimizer` (`adam`), else `policy.optimizer.name` (`torch.optim.AdamW`) — the two backends spell it differently |
| `nv.dl.training.config.recompute_granularity` | `policy.megatron_cfg.recompute_granularity`, only when `activation_checkpointing` is on. Megatron ignores the granularity otherwise, so reporting it would describe recompute that is not happening |
| `nv.dl.training.target.train_iters` | `<algo>.max_num_steps`, when positive. `rm` spells "train for one epoch" as `-1`, which is omitted rather than reported as a target |
| `nv.dl.software.torch`, `.cuda`, `.nccl`, `.transformer_engine` | versions as loaded in this process, not as pinned in the lockfile. Each is omitted when unavailable — CUDA and NCCL on a CPU-only build, transformer-engine when it is not installed |
| `nv.dl.rank`, `nv.dl.world_size` | this process's rank and group size (`RANK` / `WORLD_SIZE`, or `0` / `1` for the driver and singleton actors) |
| `nv.dl.local_rank` | worker processes only: `LOCAL_RANK`, which `RayWorkerGroup` exports beside `RANK`. Identifies a device on one node, which is what you have when the complaint came from `nvidia-smi` rather than from the job |
| `rl.worker_group` | worker processes only: the worker group's `name_prefix` (`lm_policy`, `vllm_policy`, ...), from `NRL_WORKER_GROUP`. The `NemoGym` actor reports the literal `nemo_gym` — it is built from the environment registry rather than by `RayWorkerGroup`, so nothing sets the env var for it |

The config-derived rows are driver-only: a worker is handed the propagated environment, not the `MasterConfig`. Attribute construction is best-effort: a missing config key simply omits that attribute; it never raises. Plus auto-detected host / GPU / SLURM / Kubernetes attributes from lens's resource detection.

Every key above that lens declares is taken from `nemo.lens.semconv` rather than written as a string, so a typo is an `ImportError` instead of silent data loss. Note the two spellings of the parallelism sizes: `nv.dl.topology.size.*` is the resource-scoped one lens types through its `DL_RESOURCE_TYPES` registry, while the `dl.tensor_parallel.*` family is for span and metric attributes.

## Typical configurations

Each example puts the NeMo-RL settings in the config and only the destination in the environment. The `++` form is a hydra-style CLI override: it is applied to the config and echoed into the run's log, so a one-off stays as traceable as an edit to the YAML.

### Console exporter (no backend)

```bash
uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml \
  ++telemetry.enabled=true ++telemetry.exporter=console
```

Spans and metrics print to stdout — a quick dry run with no backend to stand up.

### Direct to an OTLP backend (http/protobuf)

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://<your-otlp-endpoint>:443
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_HEADERS="<header>=<value>"   # any auth headers your backend requires
uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml \
  ++telemetry.enabled=true
```

See [Observability Stack](observability-stack.md) for the full backend-export setup.

### Per-step granularity

```yaml
telemetry:
  enabled: true
  span_groups: per_step
```

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

`per_step` adds the per-step phases (rollout, generation, reward, advantage, policy update) on top of the run-level `job` span, so the whole run is one trace. See [Span Groups](span-groups.md).

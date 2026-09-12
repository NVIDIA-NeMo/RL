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
  export_strategy: single_rank # single_rank | all_ranks | sampled | first_rank_per_node
  export_rank: -1             # for single_rank: which rank exports (-1 = last rank)
  export_sample_rate: 1.0     # for sampled: fraction of worker ranks that export
  sampler_enabled: false      # drop spans at the SDK level using export_sample_rate
  traces_enabled: true        # emit trace spans
  metrics_enabled: true       # emit the rl.* metric instruments
  logs_enabled: false         # bridge Python logging to OTel logs (trace-correlated)
  exporter: otlp              # otlp | console
  vllm_native_tracing: false  # opt in to vLLM's own OTLP tracing (gRPC-only — see vllm-tracing.md)
```

The defaults above are the field defaults of `TelemetryConfig` (`nemo_rl/telemetry/config.py`). The endpoint, headers, and protocol are **not** in this block — they come from the standard `OTEL_EXPORTER_OTLP_*` env vars (see below).

The driver always exports (it hosts the training loop and the metrics logger); `export_strategy` / `export_rank` govern the Ray **worker** ranks.

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

## Export strategy

`export_strategy` controls which **worker** ranks actually send telemetry:

- `single_rank` (default) — only the rank named by `export_rank` (`-1` = last rank).
- `all_ranks` — every worker exports.
- `sampled` — a deterministic hash of the rank selects `export_sample_rate` of the ranks. The same rank and rate always give the same outcome, so the exporting set is stable across restarts.
- `first_rank_per_node` — the first local rank on each node exports (reads `LOCAL_RANK`).

`export_sample_rate` applies to `sampled`; it has no effect under the other strategies. `sampler_enabled` is independent of `export_strategy` but asks the same kind of question: it installs lens's rank-aware sampler on the TracerProvider, which hashes the rank against `export_sample_rate` once at startup and then keeps or drops *every* span on that rank. A rank has to clear both filters to emit anything, so leaving the sampler on with a low rate can silence a rank the strategy selected.

The driver is independent of both — it always exports, and its rank sampler is disabled for the same reason (`_unrank` in `nemo_rl/telemetry/setup.py`): a synthetic rank 0 is not a member of the population the filters are selecting from. Singleton actors such as the async trajectory collector are exempt on the same grounds. Non-exporting ranks get an empty (`frozenset()`) span-group set, so `is_span_group_enabled()` is `False` everywhere and no span objects are created at all. See [lens: sampling](https://github.com/NVIDIA-NeMo/Lens) for the detailed semantics.

`RANK` is **group-local**: the policy group and the generation group each number their workers from zero. So `export_rank: 3` selects rank 3 *of every worker group*, and each group's spans carry an `rl.worker_group` attribute to tell them apart.

## Run identification

Every run gets a `run_id` that flows to all backends as a resource attribute and is shared by the driver and every worker.

**Priority order:**

1. `NEMO_RL_OTEL_RUN_ID` (explicit, highest priority).
2. `SLURM_JOB_ID` (auto-detected on SLURM clusters).
3. Auto-generated 12-character hex id (fallback).

The `run_id` is written to the environment on the driver **before** `init_ray()`, so every worker inherits the same value and correlates to the same run. This is also how vLLM's native spans are correlated back to the RL run — see [vLLM Tracing](vllm-tracing.md).

Filter by `run_id` in your backend to isolate a specific run.

## Resource attributes

`init_telemetry_driver` sets stable-for-the-run values on the OTel `Resource`, so they appear on every span/metric as backend "Process" tags:

| Attribute | Source |
|---|---|
| `rl.algorithm` | the `algorithm="<algo>"` passed to `init_telemetry_driver` |
| `rl.model` | `policy.model_name` |
| `nemo.precision` | `policy.precision` |
| `dl.tensor_parallel.size` | `policy.megatron_cfg` / `dtensor_cfg` TP size |
| `dl.pipeline_parallel.size` | `policy.megatron_cfg` PP size |
| `dl.rank`, `dl.world_size` | set automatically by lens |
| `rl.worker_group` | worker processes only: the worker group's `name_prefix` (`lm_policy`, `vllm_policy`, ...), from `NRL_WORKER_GROUP` |

Attribute construction is best-effort: a missing config key simply omits that attribute; it never raises. Plus auto-detected host / GPU / SLURM / Kubernetes attributes from lens's resource detection.

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

`per_step` makes each training step its own root trace (rollout, generation, reward, advantage, policy update). See [Span Groups](span-groups.md).

# Profile Gym HTTP and token capture on a CPU workstation

`tools/http_profile` compares the checkout's actual Gym HTTP/capture middleware
with opaque forwarding prototypes and NRL's direct capture implementation. The
default modes run on a CPU workstation without PyTorch; the opt-in `direct_native`
arm uses CPU tensors. No mode requires CUDA, a tokenizer, a model, or a training job.

The experiment models the chat-completions traffic an external SWE harness such
as `harbor_agent_general` produces: independent rollout conversations, four
sequential turns, code and terminal text, and a tool schema. The latency benchmark
does not launch Harbor, containers, or an SWE task. A separate
[Harbor credential smoke](#harbor-credentials-without-an-inference-proxy) below
runs actual OpenCode agents in Docker against a fake model. The production Harbor
adapter uses the generic rollout endpoint shim described in
[Direct token capture](direct-token-capture.md) for credential provisioning.
A conversation ID identifies one rollout, not its entire GRPO group.

## What is measured

The client, HTTP backend, and optional proxy run in separate processes over
loopback TCP. Servers use uvicorn/uvloop/httptools. The backend returns deterministic
OpenAI-compatible completions; synthetic engine token buffers are prepared before
measurement. All modes use the same backend fixture and client validation.

| Mode | Response path | Capture |
| --- | --- | --- |
| `direct` | Client → backend | None; HTTP baseline |
| `direct_native` | Client → backend with rollout API key | Production capture owner, engine bridge and native tensor codec; prebuilt engine outputs and memory storage |
| `gym` | Client → actual Gym `VLLMModel` → backend | Disabled |
| `gym_capture` | Actual Gym external-capture middleware and `VLLMWorkerCaptureHandler` | Actual `RolloutTokenCapture`, in-memory staging sink, actual fsync-backed `FileLineageStore`; awaited before response |
| `opaque` | Request/response byte streams through an aiohttp proxy | None |
| `opaque_inline` | Opaque forwarding | Exact block-tree insertion on backend response path |
| `opaque_thread` | Opaque forwarding | Bounded enqueue to one tree-owning thread |
| `opaque_process` | Opaque forwarding | Bounded enqueue to one tree-owning process |

The opaque proxy adds a rollout header from the URL. It never parses or re-encodes
JSON or SSE. This is a benchmark adapter with a minimal header set, not a general
production reverse proxy. Native capture occurs inside the backend from its owned
token buffers; text at the proxy is insufficient to recover exact training tokens.

The legacy and proposed capture arms have different downstream contracts. The
legacy arm captures and verifies chained deltas; the prototype stores every call's
actual context and distinct generated action. This is a measurement of the proposed
contract, not an implementation-equivalence test. In particular, distributed
staging durability and existing ledger custody semantics are absent from the native
prototype. Shared-prefix training, including [PR #4537](https://github.com/NVIDIA-NeMo/RL/pull/4537), is not a dependency.

For the legacy `token_capture.transport: gym` vLLM capture path, the switches are
`env.should_use_nemo_gym=true`, `policy.generation.backend=vllm`,
`policy.generation.vllm_cfg.async_engine=true`,
`policy.generation.vllm_cfg.expose_http_server=true`, `data_plane.enabled=true`,
`data_plane.impl=transfer_queue`, and `token_capture.enabled=true`, within the
single-controller recipe. Setup derives the capture directory, credentials and
generation backend. The Gym environment installs `token_id_capture.enabled=true`,
`all_agents=true`, `external_staging=true`, `rebuild_response=false` and a shared
`FileLineageStore`, and disables `policy_model`'s `return_token_id_information`.
Those are existing settings, not switches for the proposed queue/tree path. The
CPU harness constructs this external-capture middleware directly without starting
the rest of the training runtime. See
[Gym setup](../nemo_rl/environments/nemo_gym.py) and
[capture validation](../nemo_rl/algorithms/single_controller_utils/setup.py).

## Run

From the repository root, with the Gym checkout present:

```bash
uv venv --python 3.13 /tmp/nrl-http-profile-venv
uv pip install --python /tmp/nrl-http-profile-venv/bin/python \
  -r tools/http_profile/requirements.txt
uv run --no-project --python /tmp/nrl-http-profile-venv/bin/python \
  python -m tools.http_profile.benchmark \
  --output results/http_profile/baseline
uv run --no-project --python /tmp/nrl-http-profile-venv/bin/python \
  python -m tools.http_profile.summarize results/http_profile/baseline
```

Use an isolated environment rather than syncing the GPU project's dependencies.
The requirements pin the CPU harness dependencies. `environment.json` records
all installed versions, interpreter, platform, commits, options, and hashes of the
benchmark sources. This environment is a CPU experiment, not Gym's production
dependency lock. Keep generated
results, profiles, traces, and environment metadata in the ignored `results/`
directory; do not commit them.

Defaults: initial contexts of 2,048/8,192/32,768 synthetic tokens, 1/8/32 concurrent
rollouts, 128 measured requests per case, three repeats, four turns per rollout,
128 generated tokens and 128 new context tokens per turn. Text uses approximately
four bytes per synthetic token. Actual request/response byte counts are reported.
Each case warms up separately. Context/concurrency/repeat order is shuffled with a
fixed seed within each mode; modes run in command-line order. Use longer bursts
and reordered modes when assessing small differences.

Every response's content is checked. Capturing runs must account for every call
after a drain barrier, with no rejected records, resolver errors, or missing
commits. A failed queue run writes a `*.failure.json` diagnostic and raises instead
of reporting successful throughput. Output directories must be fresh.

`measurements.jsonl` preserves individual runs and latency samples; `results.json`
contains the completed run set. `summary.csv` and `summary.json` report medians of
per-run percentiles/throughput/CPU, the range of p50 across repeats, queue peaks,
and drain time. They do not pool percentiles or claim statistical confidence bounds.
First-body-byte latency measures arrival at the client, not engine TTFT. Server CPU
sums proxy, backend, and, when separate, resolver process CPU through capture drain.
Client CPU is recorded separately. Queue peaks bound pending payloads, not total
tree memory. Completed trees remain resident until case reset.

To check streaming and controlled slow consumers:

```bash
uv run --no-project --python /tmp/nrl-http-profile-venv/bin/python \
  python -m tools.http_profile.benchmark \
  --modes direct gym_capture opaque_process --sizes 8192 --concurrency 1 8 \
  --requests 64 --stream --backend-delay-ms 16 \
  --output results/http_profile/streaming
uv run --no-project --python /tmp/nrl-http-profile-venv/bin/python \
  python -m tools.http_profile.benchmark \
  --modes opaque_inline opaque_thread opaque_process --sizes 8192 \
  --concurrency 1 --requests 32 --rounds 1 --resolver-delay-ms 5 --trace \
  --output results/http_profile/slow-resolver
```

The backend divides a streaming delay across eight SSE content chunks. Gym's
compatibility route requests a complete upstream response then replays it as SSE;
that arm consequently uses one backend sleep instead. Scheduler granularity means
the elapsed total is not exactly the configured delay.

## Perfetto traces and CPU profiles

Run tracing separately from speed measurements:

```bash
uv run --no-project --python /tmp/nrl-http-profile-venv/bin/python \
  python -m tools.http_profile.benchmark \
  --modes gym_capture opaque_thread opaque_process --sizes 32768 \
  --concurrency 4 --requests 16 --rounds 1 --trace \
  --output results/http_profile/traces
```

Open any `*.trace.json` with **Open trace file** in [Perfetto](https://ui.perfetto.dev).
These use Perfetto's supported [Chrome JSON format](https://perfetto.dev/docs/getting-started/other-formats).
No tracing daemon is needed. Tracks include client and server request spans,
first response body, Gym ledger resolution and append/fsync, awaited legacy capture,
native enqueue, resolver queue wait and work, and pending records/bytes counters.
`resolver work` includes tree insertion and any explicitly configured test delay.
Flow arrows connect opaque client → proxy → backend and enqueue → resolver.
Gym currently does not forward the diagnostic header, so its proxy → backend flow
is absent; those backend spans remain visible independently.

Each concurrent request has its own logical track to avoid false nesting on the
event-loop thread. These are wall-time slices, not CPU scheduling traces. Threads
and processes use the same host monotonic clock; cross-host traces will require
clock alignment. Traces contain synthetic identifiers, paths, byte counts, and
timing, not prompt/token contents. Trace storage is in-memory until each case ends;
use small diagnostic runs. The ordinary serving benchmarks have no tracing
middleware. Instrumented timings are deliberately rejected by `summarize`.

For function-level profiles, replace `--trace` with `--profile`, optionally
`--profile-clock wall` (default `cpu`). Yappi records backend and proxy functions,
including backend threads and builtins. It emits `.pstat` and text rankings by
inclusive and exclusive time. The separate resolver process is not Yappi-instrumented;
use `opaque_thread` to profile the same tree implementation. Its total CPU is still
counted in process-mode benchmark results. Per-call profiling heavily perturbs
Python loops; use it to locate work, never to calculate speedups.

## Serialization and ownership in the experiment

Let `P` be prompt tokens, `G` generated tokens, `D` newly added prompt tokens,
`H` model calls in a rollout, `V` vocabulary size, `L` MoE layers, `K` selected
experts per token/layer, and `E` experts per layer. Let `B` be actual request JSON
bytes, and `R` response text/JSON bytes. Token counts do not uniquely determine
UTF-8 bytes; the fixture chooses `B = Θ(P)` and `R = Θ(G)` deliberately.

| Boundary/work | Payload/work size | Location |
| --- | --- | --- |
| Client JSON encode | `Θ(B)` | Prebuilt outside timing for every arm; client still transmits `B` |
| Gym request parsing, typed validation, object dumps, outgoing JSON encoding | Each traversal `Θ(B + H)` including capture's staging-key chain | Critical path; real Gym functions |
| Opaque forwarding | `Θ(B + R)` bytes transferred; bounded HTTP chunks; no application JSON conversion | Critical path; extra socket hop still exists |
| Backend HTTP parse | `Θ(B + H)` | Critical path in every mode; common stub substitutes for real vLLM ingress |
| Legacy prefix reconstruction/validation | `Θ(P)` token integers | In-memory sink replaces TQ fetch; backend waits |
| Legacy delta/mask/logprob construction and digest encoding | Root `Θ(P + G)`; continuation delta `Θ(D + G)`; cumulative hash remains `Θ(P + G)` | Backend waits; real staging code encodes tokens as 8-byte integers and floats as float32; record validation recomputes its digest |
| Capture admission/commit JSON | `Θ(H)` keys plus bounded per-call metadata | Backend/Gym HTTP; no full token arrays on this configured external-capture path |
| Gym history fingerprints/context digest | `O(B + R)` per history scan | Request path, including awaited ledger worker threads |
| Gym custody ledger JSON encode/decode | `Θ(H)` chain metadata per row, `O(H²)` accumulated chain-key bytes across a rollout | Actual file read/cache and append/fsync; no token arrays in custody rows |
| Backend response JSON encode | `Θ(R + H)` legacy, `Θ(R)` native | Critical path; common stub encode plus legacy receipt |
| Gym response decode/validation/dumps/JSON or SSE replay | `Θ(R + H)` before stripping metadata, then `Θ(R)` | Critical path |
| Native thread enqueue | `O(1)` references to immutable owned buffers and fixed metadata | Critical path; no tree insertion or token scan awaited |
| Native process enqueue/IPC | Producer admission `O(1)`; feeder pickles/copies `4(P + G) + 4G` bytes, consumer unpickles | Feeder/consumer off coroutine; still CPU, GIL, copies and pipe traffic |
| Resolver tree insertion | Expected `O(P + G)` scan/hashing of exact 16-token blocks; distinct per-call logprobs `4G` bytes | Resolver thread/process; atomic under a single tree owner |
| Dense materialization | `Θ(P + G)` token bytes per call | After capture; allows training without shared-prefix execution |
| Client response parse | `Θ(R)` | After individual latency sample, but part of closed-loop throughput |

For the real worker, packed exact token IDs need `Θ((P+G) log V)` bits and
selected router IDs need `Θ((P+G)LK log E)` bits (or fixed-width integer storage).
Selected IDs scale with `K`; full router score vectors would scale with `E`.
Base64 expands raw bytes by approximately 4/3 and costs a linear scan/copy each
way. Router tensors, their JSON/base64 conversions, GPU-to-host ownership transfer,
real vLLM response materialization, Ray, TransferQueue serialization, media, and
training materialization are **not measured by this harness**. Their costs must be
added when integrating the real backend; the native buffers here already exist.

## Queue and training contract

`CaptureQueue.submit` reserves bounded record/byte capacity and submits an owned,
immutable call record. It does not await insertion. Admission still acquires short
locks; it is not lock-free, and multiprocessing uses a Python feeder thread. Queue
overflow records an invalid rollout. Worker errors are retained and checked at a
drain barrier. This prototype is memory-only and has no crash recovery.

`--resolver-workers N` shards `opaque_process` by a stable hash of the rollout ID.
Each process still exclusively owns its assigned trees. The configured queue
record/byte budget is divided across processes; a hot shard can fill before others.
The reported aggregate peak for multiple shards is the sum of individual peaks,
an upper bound rather than a simultaneous sample. Per-shard details are retained.

One resolver owns a rollout tree. Out-of-order parents, subagent branches, repeated
identical outputs with distinct call IDs, and compacted contexts can coexist
without determining which prior generation produced an input. Exact contexts,
generated-action identities, prompt boundaries and per-action logprobs remain
independent of tree interning. Tests reconstruct every stored context densely.
Production records will also need policy/tokenizer identity, complete sampling and
training metadata, and optional routed-expert/media buffers.

For production, mark a rollout trainable only after its admitted calls have all
resolved and its reward is available. Close admission after harness/subagents join;
then seal/drain that rollout. Compute its GRPO advantage before expanding to calls.
Tree sharing must not deduplicate distinct actions or count copied assistant text
as newly generated actions. Dense consumption can ship independently of optional
shared-prefix execution. Sustained service needs enough resolver capacity and an
explicit overflow policy; fast HTTP with a growing capture backlog is insufficient.

## Sustained capture load

Longer bursts help distinguish HTTP response latency from capture capacity.
Check rejected records, queue peaks, and post-response drain time as well as
throughput. A run that overflows its capture queue is a failed run. Shard resolver
work by rollout when comparing multiple processes, and keep the total queue
budget fixed across comparisons.

```bash
uv run --no-project --python /tmp/nrl-http-profile-venv/bin/python \
  python -m tools.http_profile.benchmark \
  --modes opaque opaque_process --sizes 32768 --concurrency 32 \
  --requests 2048 --rounds 3 --resolver-workers 2 \
  --output results/http_profile/sustained-sharded
```

## Harbor credentials without an inference proxy

The launch controller can provision one ordinary provider API key per rollout,
register `key -> rollout` on the backend once, then pass the key through Harbor's
`AgentConfig.env`. The agent sends normal requests directly to the model endpoint
with `Authorization: Bearer <key>`. It needs no conversation field or custom
header. Subagents using the same provider configuration belong to the same rollout;
this key identifies the whole tree, not a particular generation or branch.

Harbor 0.22.0 (the minimum accepted by this checkout's `harbor_agent_general`)
supports three useful configuration routes for OpenCode:

| Route | Harbor configuration | OpenCode configuration | Tradeoff |
| --- | --- | --- | --- |
| Provider environment list | `harbor_agent.env.OPENAI_API_KEY` | `provider.nrl.env: [OPENAI_API_KEY]` | Ordinary credential variable; explicitly bind it to the custom provider |
| Config environment reference | `harbor_agent.env.NRL_ROLLOUT_API_KEY` | `provider.nrl.options.apiKey: "{env:NRL_ROLLOUT_API_KEY}"` | Recommended: explicit provider binding, key stays out of the config overlay |
| Inline config value | No key environment variable | `provider.nrl.options.apiKey: <rollout-key>` | Works, but Harbor does not redact arbitrary secrets nested in `kwargs`; they persist in job/config logs |

The OpenCode settings live under `harbor_agent.kwargs.opencode_config`. In
standalone Harbor YAML, the equivalent fields are `agents[].env` and
`agents[].kwargs.opencode_config`. The CLI's repeatable `--ae` / `--agent-env`
sets the same environment mapping. `${VAR}` host-environment templates are also
supported (escape the interpolation as `\${VAR}` when passing through Gym).
Neither a static YAML value nor a shared CLI environment automatically creates a
different key per trial: inject the key into each rollout's own agent config.
The Gym wrapper already launches one Harbor job/trial per rollout.

An example **launch-time template**, with values filled by the controller:

```yaml
harbor_agent:
  name: opencode
  model_name: nrl/policy
  env:
    NRL_ROLLOUT_API_KEY: <unique-key-for-this-rollout>
  kwargs:
    opencode_config:
      enabled_providers: [nrl]
      small_model: nrl/policy
      provider:
        nrl:
          npm: "@ai-sdk/openai-compatible"
          options:
            baseURL: http://model-server:8000/v1
            apiKey: "{env:NRL_ROLLOUT_API_KEY}"
          models:
            policy:
              name: policy
              limit:
                context: 32768
                output: 4096
```

Use the actual served model name and supported limits. Setting `small_model`
routes auxiliary model use through the same provider too. An agent configured
with another provider/key for a subagent must receive the rollout binding there
as well; inheritance is not universal across arbitrary agents.

Use a custom provider name such as `nrl` for Chat Completions. OpenCode 1.18.18's
built-in `openai` provider explicitly selects the Responses API; overriding its
SDK with `@ai-sdk/openai-compatible` fails before sending a request. The smoke
uses `nrl` for all three routes.
See [OpenCode's pinned provider implementation](https://github.com/anomalyco/opencode/blob/v1.18.18/packages/opencode/src/provider/provider.ts),
[provider configuration](https://opencode.ai/docs/providers/), and
[environment references](https://opencode.ai/docs/config/).

### Required change at Gym's worker handoff

Harbor redacts sensitive environment values during normal serialization. The
current `harbor_agent_general.run()` sends `job_config.model_dump(mode="json")`
to Ray, which turns a literal per-rollout key into a redacted value. When a
literal matches the host environment, Harbor substitutes a host-variable
template instead; that requires the correct environment on the receiving worker.

For internal transport, preserve credentials with Harbor's explicit option:

```python
job_config.model_dump(mode="json", context={"redact_sensitive_env": False})
```

Use that only for the worker handoff; normal persisted configs should still use
Harbor's redaction. Copy the agent config before adding a rollout's environment
value, and do not mutate shared `os.environ` between concurrent rollouts. The
smoke tests reproduce the redaction problem and use the corrected transport;
they do not launch a Ray cluster. The generic Gym endpoint shim in this change
also applies the correction to the production Harbor adapter.

The backend must accept these issued credentials and map them to a rollout inside
its existing HTTP server. A static unrelated vLLM API key does not provide that
mapping. `token_capture.transport: direct` enables NRL's backend integration and
registration lifecycle. After registration, attribution is an in-process key
lookup; no Gym forwarding hop or remote per-request identity lookup is needed.
Native staging uses a bounded pool and drains at rollout completion. Calls that
actually generate tokens receive distinct capture IDs, including retries.

### Reproduce the Docker smoke

This is separate from the HTTP timing environment:

```bash
uv venv --python 3.13 /tmp/nrl-harbor-key-venv
uv pip install --python /tmp/nrl-harbor-key-venv/bin/python \
  -r tools/http_profile/requirements-harbor.txt
uv run --no-project --python /tmp/nrl-harbor-key-venv/bin/python \
  python -m pytest --noconftest tools/http_profile/test_harbor_keys.py -q
uv run --no-project --python /tmp/nrl-harbor-key-venv/bin/python \
  python -m tools.http_profile.harbor_key_smoke \
  --image "$HARBOR_SMOKE_IMAGE" \
  --output results/http_profile/harbor-keys
```

Set `HARBOR_SMOKE_IMAGE` to a Linux image with OpenCode, bash and `stdbuf`
installed. The test requires Docker Desktop's `host.docker.internal` route and
binds a temporary host HTTP listener. It replaces
only Harbor's installer with a version check to reuse the image; `Job.create`,
trial environment propagation, OpenCode's production `run()`, real HTTP/SSE,
tool execution, verification, and teardown run normally. All credentials are
fresh synthetic test keys. The JSON round trip reproduces the worker payload,
without exercising Ray transport itself. No vLLM, GPU, training, real model,
compaction, or subagent execution is tested here.

Each credential mode runs two concurrent jobs, with a server barrier requiring
their first HTTP requests to overlap. The server injects one HTTP 503 per rollout
to exercise real client retries, then requests a bash tool call and a follow-up
completion. It validates each request's key against an independent task marker,
checks for unwanted conversation fields/headers, and verifies the file written by
the agent. An unregistered key is rejected before JSON parsing. `summary.json`
records observations and assertion failures; the command exits nonzero on a failed
trial or missing tool round trip.

## Tests

```bash
uv run --no-project --python /tmp/nrl-http-profile-venv/bin/python \
  python -m pytest --noconftest tools/http_profile -q
uv run --no-project --python /tmp/nrl-http-profile-venv/bin/python \
  ruff check tools/http_profile
```

`--noconftest` keeps the CPU tests independent of the repository's Ray/PyTorch test
fixtures. Coverage includes reordered branches and dense reconstruction, distinct
identical generations, non-waiting submission, explicit overflow and duplicate-ID
failure reporting, drain/reset, trace flow metadata and concurrent ASGI tracks.


## Production native capture

The `direct_native` arm exercises NRL's production native-output bridge, bounded
capture pool, and tensor codec. It replaces the engine with prebuilt outputs and
TransferQueue with memory; it has no Gym model proxy. It currently benchmarks
non-MoE payloads. The fake engine runs on the HTTP event loop, so the comparison
excludes the production worker's HTTP-to-engine loop handoff as well as vLLM/GPU
execution, tokenization, real TQ transfer, and routing-array costs.

Install the production arm's additional CPU tensor dependencies before running it:

```bash
uv pip install --python /tmp/nrl-http-profile-venv/bin/python \
  -r tools/http_profile/requirements-native.txt
uv run --no-project --python /tmp/nrl-http-profile-venv/bin/python \
  python -m tools.http_profile.benchmark \
  --modes direct direct_native gym_capture --sizes 32768 \
  --concurrency 1 32 --requests 128 --rounds 3 \
  --output results/http_profile/direct-native
```

The harness alternates mode order between rounds. Registration is included in
case throughput, outside individual request latencies. Finalization/drain is
reported separately. Background staging still competes for the GIL, so compare
server CPU and sustained throughput alongside request latency. Keep power and
thermal conditions consistent across runs.

Use a separate run with `--trace` for Perfetto diagnostics. The `native tensor
staging` spans use the resolver thread IDs. Instrumented timings must not be used
for speedup claims.

See [Direct token capture](direct-token-capture.md) for configuration, generic
Gym agent integration, training selection, and asymptotic payload costs.

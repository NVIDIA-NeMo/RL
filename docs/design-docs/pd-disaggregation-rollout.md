# NemoRL<>TRTLLM Disaggregation Rollout  (Experimental)

> **Experimental**: PD disaggregation is wired for the TensorRT-LLM backend on the  
> NeMo-Gym rollout path only.

NeMo RL owns the *replica* (which engines exist, how they are placed, and their
lifecycle) and the *refit safety* (what happens when weights change mid-request). It
delegates both the *KV handshake* and the *routing inside a replica* to the inference
backend's own disaggregation front-end — for TensorRT-LLM, `OpenAIDisaggServer`.

## Design



### Topology

A **replica** is a self-contained disaggregated fleet — M context engines plus K
generation engines — fronted by F `OpenAIDisaggServer` instances
(`num_frontend_workers`, default 1). N replicas are created, and each exposes F URLs:

```
                               replica 0  OpenAIDisaggServer fe_0 (one URL) --.
                                          OpenAIDisaggServer fe_{F-1} (URL) --+
NeMo-Gym --session affinity-->                                                |-- ctx_0 .. ctx_{M-1}
                                                                              `-- gen_0 .. gen_{K-1}

                               replica 1  OpenAIDisaggServer fe_0 (one URL) --.
                                          ...                                 `-- ...
```

Every frontend of a replica is handed the *same* two address pools, so F changes nothing
about the engine layout — it only widens the relay in front of it. The frontend is a
single-process ASGI relay that terminates HTTP, strips the Gym-only request fields,
re-serializes the body, asks the routers, and makes two engine calls per turn. All of that
is CPU on one event loop under one GIL, so with long multi-turn prompts a single frontend
becomes the replica's turn-throughput ceiling well before its GPUs saturate. F > 1 spreads
that work over F processes.

`TrtllmGeneration.dp_openai_server_base_urls` reports all N×F frontend URLs as one flat
list, so NeMo-Gym sees "N×F instances" exactly as it sees N without disaggregation. Its
existing per-session affinity (`responses_api_models/vllm_model/app.py::_resolve_client`,
a `session_id -> client` map seeded by `sha256(session_id) % len(clients)`) binds a
trajectory to one frontend — and therefore to one replica — for its whole lifetime. Since
each replica contributes exactly F of the N×F slots, replicas stay uniformly loaded.

That stickiness is a **correctness** requirement, not just load-spreading. Each frontend
holds its own `ctx_router` state in-process, so two frontends of the same replica have no
shared view of which context engine holds a given conversation's prefix. If a trajectory's
later turn reached a different frontend, that frontend would route it to a context engine
that never saw the prefix — correct output, but a full re-prefill and the whole point of
`ctx_router: conversation` lost.

Replicas are disjoint: no engine belongs to two of them. That is what lets every frontend  
keep its routing state in-process and removes any need for cross-replica coordination —  
`coordinator_url` is `None`, even with F > 1.

**Everything below the replica boundary is opaque to NeMo RL.** Once a request reaches a
replica's `OpenAIDisaggServer`, that server alone decides which context engine serves the
prefill, which generation engine serves the decode, whether a context leg is needed at
all, and how the KV handshake between them is carried. NeMo RL never sees an individual
engine on the request path; it only composes each replica's two pools and selects the
router policies through `DisaggServerConfig` (see [Configuration](#configuration)).

The cost of that boundary is that context affinity is no longer free. When a replica had
exactly one context engine, Gym's choice of URL *was* the choice of context engine. Now
`ctx_router` has to provide it, which is why it must be configured with a stateful policy
while `gen_router` need not be.

Affinity is therefore established at three levels, and NeMo RL controls only the first:

| Level | Decides | Driven by | State |
| --- | --- | --- | --- |
| Gym → frontend | which replica (and which of its F frontends) | `sha256(session_id)`, sticky | in Gym's `session_id -> client` map |
| frontend → context engine | which of the replica's M context engines | `ctx_router` | in that frontend's process |
| context engine → ADP rank | which attention-DP rank holds the prefix | `ConversationParams.conversation_id` relayed on the request | none — a function of the id |

The third row is why `trtllm_http_server` forwards the conversation id (Gym's session id,
or the one the disagg service stamps onto `disaggregated_params` for its ctx/gen legs) to
`llm.generate_async`: under attention-DP each rank owns a separate KV pool, so without the
id a turn lands on an effectively random rank and re-prefills its history.

### Forming a replica

`TrtllmGeneration` lays engines out replica by replica, contexts before generations, so  
membership is positional and needs no negotiation. Once every engine has initialised and  
reported its address, it builds the address pools for each replica — `type='ctx'` entries  
for that replica's context engines, `type='gen'` for its generation engines — and starts F  
`DisaggServerActor`s against them, collecting one URL each. Those N×F URLs become  
`dp_openai_server_base_urls`.

Each frontend is a CPU-only Ray actor in its own process rather than a thread inside an
engine worker: it is the request hot path for the whole replica, and sharing a process
with an engine would couple the replica's routing latency to that one engine's load. Three
properties are worth noting, all of which exist so a crashed frontend can come back at the
*same* URL and Gym's sticky clients recover after their 5xx retries:

- **Serving starts in `__init__`, not in `start()`.** Ray re-runs `__init__` on actor
restart but never replays method calls. `start()` only waits for `/health` and asserts the
serving thread is still alive.
- **The node pin is hard.** Frontends spread round-robin over their own replica's engine
nodes (`addrs[base + fe_idx % per_replica]`), so a frontend sits beside engines it talks
to, and restarts are unlimited.
- **Ports are deterministic**: `frontend_base_port + replica_idx * num_frontend_workers +
frontend_idx`. Both indices are needed — several replicas can share a node, and an offset
carrying only `frontend_idx` would have every replica's frontend 0 ask for the same port.
That failure is silent: `uvicorn`'s bind failure calls `sys.exit(1)`, which only unwinds
the daemon thread, and the `/health` probe is then answered 200 by whichever frontend won
the bind. Hence the liveness assert in `start()`.

Every engine — context and generation alike — therefore has to run its own HTTP server,
because an address is the only way `OpenAIDisaggServer` can reach one. That is the sole
thing it is given about an engine.

NeMo RL still created those engines and still holds their Ray actor handles, so refit does
not go through the disagg server at all: `TrtllmGeneration` drives the engines directly, as
it does without disaggregation. The same is true of sleep/wake, prefix-cache reset and
profiling. Delegation is confined to the request path; the control plane is unchanged.

### Node affinity

Placement matters at two scales, and both are preferences rather than hard requirements.

**An engine's TP group should fit inside one node.** Tensor-parallel collectives are the
most bandwidth-hungry traffic in the system. When a per-role TP is wider than a node, the
engine should at least stay inside one *segment* — the `segment_size` nodes of a single
NVLink domain that `RayVirtualCluster` aligns placement to.

**A replica should fit inside one node, or failing that one segment.** The ctx→gen KV
transfer happens entirely within a replica, so its cost is set by the slowest link any of
that replica's engines has to cross. A replica straddling a segment boundary pays network
bandwidth on every handoff, and it pays it per turn.

Engines are laid out replica by replica with same-role engines contiguous, which keeps each role's GPUs adjacent.

Which of the two boundaries actually costs depends on the platform:

- **HGX, 8 GPUs per node** — the NVLink domain *is* the node, so crossing one drops
straight to the network. The node boundary is what matters.
- **GB200 NVL72** — Ray sees 4 GPUs per node and 18 of those nodes share one NVLink
domain, so crossing a node is cheap. The segment is the boundary worth respecting.



## Refit safety

Disaggregation does not introduce a new class of staleness. The aggregated path already  
tolerates decoding KV that was built under previous weights: the only cache operation after  
a weight update is `reset_prefix_cache()`, which clears the reusable prefix cache but not  
the KV held by in-flight requests. 

`drain=True` is the one guarantee that does weaken. It blocks until `active_requests` and
`waiting_queue` are empty, and a context-only request whose KV has not been pulled yet is
in neither — once its forward finishes the engine releases its scheduler slot and parks it
in a separate `_requests_in_transfer` map, keeping only the KV blocks. So draining both
engines only empties their scheduler queues; a context engine can still be holding KV
blocks awaiting transfer, which a generation engine then decodes under the new weights.
This affects the synchronous path, where `drain=True` otherwise means "nothing holds KV
from the previous weights"; the in-flight path already accepts that.

## Configuration

```yaml
policy:
  generation:
    backend: trtllm
    trtllm_cfg:
      tensor_parallel_size: 2
      expose_http_server: true          # every engine needs a disagg-capable endpoint
      disaggregation:
        enabled: true
        num_context_engines: 2          # M — per replica
        num_generation_engines: 3       # K — per replica, independent of M
        cache_transceiver_backend: UCX  # DEFAULT | UCX | NIXL | MOONCAKE | MPI
        ctx_router: conversation        # stateful: keeps a trajectory on one context engine
        gen_router: load_balancing      # stateless: placed locally, no coordinator
        num_frontend_workers: 2         # F — disagg servers per replica; N*F <= 256
        frontend_base_port: 17300       # port = base + replica_idx * F + frontend_idx
        # Per-role engine overrides, merged over trtllm_cfg. Any TRT-LLM kwarg
        # goes here; TP and the MoE split are the ones that usually differ,
        # and each role must satisfy moe_tp * moe_ep == its own TP.
        ctx_trtllm_kwargs:
          tensor_parallel_size: 4
          moe_tensor_parallel_size: 2
          moe_expert_parallel_size: 2
        gen_trtllm_kwargs:
          tensor_parallel_size: 2
          moe_tensor_parallel_size: 1
          moe_expert_parallel_size: 2
      trtllm_kwargs:
        kv_cache_config:
          enable_block_reuse: true      # see below
```

GPU budget:

```
replica width  = M * context_tp + K * generation_tp
inference GPUs = num_replicas * replica width
```

`num_frontend_workers` does not appear here: frontends are CPU-only actors
(`num_cpus=1, num_gpus=0`), so F never changes the GPU budget or the engine layout.

`TrtllmGeneration._plan_engines()` turns this into a per-engine `(role, width)` list — the
single source of truth everything else derives from:

```
[ctx_tp × M, gen_tp × K,   ctx_tp × M, gen_tp × K, ...]
 \______ replica 0 _____/  \______ replica 1 ...
```

Same-role engines are contiguous within a replica, which keeps each role's block of GPUs  
adjacent. The replica *count* is derived from the inference cluster's size rather than
configured — `num_replicas = inference_GPUs / replica_width`, the same rule the DP-shard
count follows without disaggregation — so `replica_width` must divide the inference GPU
count.

Each frontend's `DisaggServerConfig.node_id` must be set explicitly, and must be distinct
across *all* of them — not just across replicas. Its default is `uuid.getnode() % 256`,
documented as assuming one disagg server per machine, and we run N×F of them. It keys the
snowflake request-id mint (`process_id` is hardwired to 0 without a coordinator, and
`time.monotonic()` shares an origin across processes on one node), so a collision would
let two frontends mint the same `ctx_request_id`. We use
`replica_idx * num_frontend_workers + frontend_idx`; the field is 8 bits, hence the
asserted `num_replicas * num_frontend_workers <= 256`.

## **Prerequisites**

- **`ChatCompletionResponseChoice` needs a `token_ids` field.** Prompt token ids and
per-token logprobs already come back on the standard response, but generated token ids do
not — the chat choice carries only each token's decoded text. `CompletionResponseChoice`
already has the field; the chat one needs the same.


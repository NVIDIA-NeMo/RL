# NemoRL<>TRTLLM Disaggregation Rollout  (Experimental)

> **Experimental**: PD disaggregation is wired for the TensorRT-LLM backend on the  
> NeMo-Gym rollout path only.

NeMo RL owns the *replica* (which engines exist, how they are placed, and their
lifecycle) and the *refit safety* (what happens when weights change mid-request). It
delegates both the *KV handshake* and the *routing inside a replica* to the inference
backend's own disaggregation front-end — for TensorRT-LLM, `OpenAIDisaggServer`.

## Design



### Topology

A **replica** is a self-contained disaggregated fleet — M prefill engines plus K
decode engines — fronted by F `OpenAIDisaggServer` instances
(`num_frontend_workers`, default 1). N replicas are created, and each exposes F URLs:

```
                               replica 0  OpenAIDisaggServer fe_0 (one URL) --.
                                          OpenAIDisaggServer fe_{F-1} (URL) --+
NeMo-Gym --session affinity-->                                                |-- prefill_0 .. prefill_{M-1}
                                                                              `-- decode_0 .. decode_{K-1}

                               replica 1  OpenAIDisaggServer fe_0 (one URL) --.
                                          ...                                 `-- ...
```

Every frontend of a replica is handed the *same* two address pools, so F changes nothing
about the engine layout — it only widens the relay in front of it. The frontend is a
single-process ASGI relay that terminates HTTP, strips the Gym-only request fields,
re-serializes the body, asks the routers, and makes two engine calls per turn.

Chat-template rendering and tokenization are the expensive part of serving a turn on
the CPU side, and they scale badly with conversation length: `build_spliced_prompt_ids`
runs over the *whole* conversation, so a turn-30 request re-renders and re-tokenizes all
30 turns. By default that work lands in the prefill adapter, which is one process per
prefill engine — it can only be grown by adding prefill engines, i.e. GPUs.

Disaggregation makes a cheaper place available. `frontend_tokenize` moves the rendering
and tokenization onto the frontends, which attach `prompt_token_ids_b64` to the prefill leg
so the adapter does no template work at all (a sample is shadow-validated prefill-side under
`NRL_TRTLLM_TOKENIZE_SHADOW_RATE`, since the ids have to be bit-identical to what the
adapter would have produced). Frontends are CPU-only actors with no GPU in their budget,
so raising F spreads the tokenization cost over F processes at no GPU cost — this is what
F is for, and what to size it against.

`TrtllmGeneration.dp_openai_server_base_urls` reports all N×F frontend URLs as one flat
list, so NeMo-Gym sees "N×F instances" exactly as it sees N without disaggregation. Its
existing per-session affinity (`responses_api_models/vllm_model/app.py::_resolve_client`,
a `session_id -> client` map seeded by `sha256(session_id) % len(clients)`) binds a
trajectory to one frontend — and therefore to one replica — for its whole lifetime. Since
each replica contributes exactly F of the N×F slots, replicas stay uniformly loaded.

That stickiness is a **correctness** requirement, not just load-spreading. Each frontend
holds its own `ctx_router` state in-process, so two frontends of the same replica have no
shared view of which prefill engine holds a given conversation's prefix. If a trajectory's
later turn reached a different frontend, that frontend would route it to a prefill engine
that never saw the prefix — correct output, but a full re-prefill and the whole point of
`ctx_router: conversation` lost.

Replicas are disjoint: no engine belongs to two of them. That is what lets every frontend  
keep its routing state in-process and removes any need for cross-replica coordination —  
`coordinator_url` is `None`, even with F > 1.

**Everything below the replica boundary is opaque to NeMo RL.** Once a request reaches a
replica's `OpenAIDisaggServer`, that server alone decides which of its prefill engines
serves the prefill, which of its decode engines serves the decode, whether a prefill leg is
needed at all, and how the KV handshake between them is carried. NeMo RL never sees an individual
engine on the request path; it only composes each replica's two pools and selects the
router policies through `DisaggServerConfig` (see [Configuration](#configuration)).

The cost of that boundary is that context affinity is no longer free. When a replica had
exactly one prefill engine, Gym's choice of URL *was* the choice of prefill engine. Now
`ctx_router` has to provide it, which is why it must be configured with a stateful policy
while `gen_router` need not be.

Affinity is therefore established at three levels, and NeMo RL controls only the first:

| Level | Decides | Driven by | State |
| --- | --- | --- | --- |
| Gym → frontend | which replica (and which of its F frontends) | `sha256(session_id)`, sticky | in Gym's `session_id -> client` map |
| frontend → prefill engine | which of the replica's M prefill engines | `ctx_router` | in that frontend's process |
| prefill engine → ADP rank | which attention-DP rank holds the prefix | `ConversationParams.conversation_id` relayed on the request | none — a function of the id |

The third row is why `trtllm_http_server` forwards the conversation id (Gym's session id,
or the one the disagg service stamps onto `disaggregated_params` for its prefill/decode legs) to
`llm.generate_async`: under attention-DP each rank owns a separate KV pool, so without the
id a turn lands on an effectively random rank and re-prefills its history.

### Forming a replica

`TrtllmGeneration` lays engines out replica by replica, contexts before generations, so  
membership is positional and needs no negotiation. Once every engine has initialised and  
reported its address, it builds the address pools for each replica — TRT-LLM still spells  
the two legs ctx/gen on the wire, so `type='ctx'` entries for that replica's prefill  
engines and `type='gen'` for its decode engines — and starts F  
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

**An engine should fit inside one node.** An engine's width is whatever its parallelism
multiplies out to — tensor parallelism alone on the TRT-LLM path, which asserts `pp == 1`;
TP × PP on a backend with pipeline stages. Tensor-parallel collectives are the
bandwidth-hungry part, so TP is the dimension that most wants to stay node-local, while
pipeline stages only pass activations between neighbours and tolerate a slower link. When
an engine is wider than a node it should at least stay inside one *segment* — the
`segment_size` nodes of a single NVLink domain that `RayVirtualCluster` aligns placement
to.

**A replica should fit inside one node, or failing that one segment.** The prefill→decode KV
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

**TODO — re-prefill a mid-decode request after an in-flight update.** With
`in_flight_weight_updates`, a request that is already decoding keeps decoding against KV
built under the old weights; the correct handling is to send it back to a prefill engine
and re-prefill its prefix under the new ones. That is not implemented, so the tokens a
request emits across an update boundary are produced from a mix of weights.

Multi-turn bounds how far this propagates. `reset_prefix_cache()` runs on every engine
after the update, so the next turn's prefill cannot reuse any pre-update block and
re-prefills the whole conversation under the new weights — the staleness is confined to
the turn that was in flight, rather than being carried forward by the prefix cache for
the rest of the trajectory.

## Configuration

```yaml
policy:
  generation:
    backend: trtllm
    # The layout. Backend-agnostic, so it lives on the generation config: any
    # backend that grows disaggregation describes its fleet with these.
    disaggregation:
      enabled: true
      num_prefill_engines: 2          # M — per replica
      num_decode_engines: 3           # K — per replica, independent of M
      num_frontend_workers: 2         # F — disagg servers per replica; N*F <= 256
      frontend_tokenize: true         # render + tokenize on the frontends
      frontend_base_port: 17300       # port = base + replica_idx * F + frontend_idx
    trtllm_cfg:
      tensor_parallel_size: 2
      expose_http_server: true        # every engine needs a disagg-capable endpoint
      # What each engine of a role looks like, merged over the keys above.
      # Only the knobs that may legitimately differ between prefill and decode:
      # TP, the MoE split, gpu_memory_utilization, max_batch_size,
      # max_num_tokens, trtllm_kwargs. precision / max_model_len / the parsers
      # are shared by the whole replica and are rejected here. Each role must
      # satisfy moe_tp * moe_ep == its own TP.
      prefill_engine:
        tensor_parallel_size: 4
        moe_tensor_parallel_size: 2
        moe_expert_parallel_size: 2
      decode_engine:
        tensor_parallel_size: 2
        moe_tensor_parallel_size: 1
        moe_expert_parallel_size: 2
      # How TRT-LLM implements the layout: what the OpenAIDisaggServer process
      # itself needs. Passed straight through to DisaggServerConfig.
      disagg_server:
        ctx_router: conversation      # stateful: keeps a trajectory on one prefill engine
        gen_router: load_balancing    # stateless: placed locally, no coordinator
        gen_tokids_ctxbytes: true     # relay ids as base64 int32, not a 30k-int array
        gen_strip_message_history: true
    trtllm_kwargs:
      # The KV transceiver is an ordinary AsyncLLM argument, not a disagg
      # schema field. Omitting it means backend=DEFAULT, which is what makes
      # an engine report itself as disaggregated at all.
      cache_transceiver_config:
        backend: NIXL                 # DEFAULT | UCX | NIXL | MOONCAKE | MPI
      kv_cache_config:
        enable_block_reuse: true      # see below
```

GPU budget:

```
replica width  = M * prefill_width + K * decode_width
inference GPUs = num_replicas * replica width
```

`num_frontend_workers` does not appear here: frontends are CPU-only actors
(`num_cpus=1, num_gpus=0`), so F never changes the GPU budget or the engine layout.

`TrtllmGeneration._plan_engines()` turns this into a per-engine `(role, width)` list — the
single source of truth everything else derives from:

```
[prefill_width × M, decode_width × K,   prefill_width × M, ...]
 \___________ replica 0 ___________/  \____ replica 1 ...
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


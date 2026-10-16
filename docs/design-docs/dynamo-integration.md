# Managed Dynamo generation design

The managed Dynamo backend owns a fixed vLLM fleet inside the Ray allocation.
It is deliberately narrower than Dynamo itself: there is no external-runtime,
Kubernetes, DGD, multi-node engine-group, or non-vLLM mode.

## Ownership and placement

Constructing `ManagedDynamoRuntime` is inert. Its explicit `start()` method
allocates ports, launches etcd and NATS JetStream, creates one Ray-managed
`dynamo.vllm` process per model-parallel group, and starts the frontend. A
worker group must fit on one node. Its world size is derived from vLLM tensor
parallelism times pipeline parallelism; expert parallelism must be either one
or equal to tensor parallelism.

Startup completes only after the frontend sees the same fixed membership at
the generation and RL endpoints and advertises the configured model. Worker
handles are recorded before readiness checks so partial startup failures can
be torn down. Shutdown is idempotent and guards the frontend, worker pool,
NATS, etcd, and temporary state independently.

## Generation state

Both GRPO trainers use `DynamoGeneration.generate_async()` against the managed
frontend. NeMo-Gym traffic passes through a process-local token wrapper. It
uses the policy tokenizer, preserves caller `nvext.extra_fields`, adds Dynamo
engine metadata, and translates rendered multi-turn prefixes back to the exact
caller token IDs.

Serialized rollout copies contain only frontend URLs, immutable worker admin
endpoints, and the prepared refit metadata. They cannot own or stop services.
That is enough for the SingleController actor to issue weight updates and for
AREAL-style post-refit cache invalidation; Magistral keeps its existing
driver-side invalidation lifecycle.

## Weight refit

Every engine loads `DynamoWorkerExtension`
(`nemo_rl/models/generation/vllm/dynamo_worker_extension.py`), a subclass of the
vLLM backend's `VllmInternalWorkerExtension`. `DynamoRefitChannel` drives it
through Dynamo's per-worker `/engine/*` RL routes, which forward an
`engine_rpc` name to vLLM `collective_rpc`; Dynamo needs no NeMo RL code of its
own. Setup calls go to `init_weights_update_group`, whose server deadline is
`dynamo_cfg.control_timeout_s`, and refits go to
`update_weights_from_distributed` with `allow_unpaused=True` and
`reset_prefix_cache=False`. A refit is one `collective_rpc` per engine. Under
the SingleController, `pause_generation_for_refit` first freezes every engine
through Dynamo's `pause_generation` route with vLLM's `keep` mode, exactly as
the vLLM backend pauses its own engines: frozen requests continue on the new
weights after the resume, and with `clear_cache` they recompute their KV. The
legacy GRPO loop instead drains its trajectories before the refit.

Each engine is one DP shard of `E = TP x PP` refit ranks
(`get_refit_fleet_shape`), and engine `i` gets rank prefix `i * E`; the
extension adds its local rank. Both transports work this way:

- **Collective** (`refit_transport: null`): `CollectiveWeightSynchronizer`
  broadcasts packed full tensors over the NeMo RL `model_update_group`.
- **NCCL reshard** (`refit_transport: nccl_reshard`):
  `NcclReshardWeightSynchronizer` sends each training shard straight into the
  engines' shards, plus a packed broadcast for the remaining parameters.

The fleet is fixed, so any refit membership other than every engine is
rejected. Three properties of the HTTP hop shape the extension's additions:

- Dynamo discards the RPC's return value, so the `nrl_*` entry points raise
  rather than return False, and an abort comes back as `RefitAborted`.
- The admin server caps request bodies at 2 MiB, so refit metadata is staged in
  chunks (`nemo_rl/utils/staged_payload.py`) and reassembled in the engine. It is
  kept there, so a pickled client copy needs only the endpoints.
- Engine RPC tasks never retry: re-posting a collective call to a dead worker
  could never complete.

Every refit has a deadline: GRPO uses `dynamo_cfg.control_timeout_s`, the
SingleController `async_rl.generation_fleet_health.refit_timeout_s`. When it
passes, the extension and the trainers each abort their own communicators and
raise, so neither side hangs in NCCL.
KV-cache invalidation stays outside the generic synchronizer because GRPO's
cache mode determines where it runs.

## Prefill/decode disaggregation

`policy.generation.disaggregation` assigns each engine a role, `prefill` or
`decode`, in launch order: replicas are tiled prefill engines first, as for
TRT-LLM. Roles share one engine shape, so each engine is still a full `TP x PP`
replica of the policy and both refit transports need no role-specific code.
Launch order is also refit rank order.

Each role runs `dynamo.vllm --disaggregation-mode <role>` with the NIXL
connector, on a node-local NIXL side-channel port. Dynamo registers decode
engines as the `backend` component and prefill engines as `prefill`. The
frontend serves nothing until both are live and never falls back to
aggregated serving, so startup waits for the expected count of each component.
Unlike TRT-LLM's per-replica frontends, one Dynamo frontend routes across the
whole fleet.

The pre-refit drain is ordered. vLLM queues new requests on a paused engine. A
prefill engine's `wait` drain also keeps stepping until decode has pulled every
prompt KV it handed off. If decode were paused first, a request prefilled just
before the drain would sit on the paused decode engine without pulling. The
prefill engine's KV lease (30 s by default) would then expire and its blocks
would be freed and cleared, and after resume decode would read those blocks;
NIXL does not check a lease on a normal read. So prefill engines pause first,
while decode is still running. Decode engines pause next, which drains every
handed-off request. Resume runs decode first, so no new handoff waits on a
paused decode engine.

Caches are cleared through `pause_generation(clear_cache=True)`, which fails
loudly when vLLM cannot reset its prefix cache. Dynamo's `flush_cache` route
and `reset_prefix_cache=True` on the update route ignore that failure, so
neither is used.

## Dependencies

`DynamoVllmWorker` runs in the `dynamo` extra's actor venv: the `vllm` extra
(vLLM 0.29.0, nccl4py, `nemo_rl`) plus the base `ai-dynamo==1.5.1` package. The
actor launches `dynamo.vllm` with its own interpreter and the driver resolves
the same venv for `dynamo.frontend`, so the extension always finds the vLLM it
was written for. `BUILD_DYNAMO=1` prebuilds that venv and installs etcd and
NATS into the image.

See [Managed Dynamo generation on Slurm](../guides/dynamo-generation.md) for
build, configuration, and launch instructions.

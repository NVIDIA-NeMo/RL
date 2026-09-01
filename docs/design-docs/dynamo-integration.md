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

Serialized rollout copies contain only frontend URLs and immutable worker
admin endpoints. They cannot own or stop services. Those endpoints are enough
for AREAL-style post-refit cache invalidation; Magistral keeps its existing
driver-side invalidation lifecycle.

## Weight refit

Dynamo uses `CollectiveWeightSynchronizer`. If each engine has world size `E`,
worker `i` starts at rank `training_world_size + i * E`. The policy sender uses
vLLM's peer initialization and its fixed packed-transfer geometry: two 1-GiB
buffers. The isolated vLLM environment validates the same constants before a
worker starts.

Generation is drained before refit. The worker then runs vLLM's native
`start_weight_update`, `update_weights`, and `finish_weight_update` transaction.
KV-cache invalidation stays outside the generic synchronizer because GRPO's
cache mode determines where it runs.

## Dependency isolation

`BUILD_DYNAMO=1` adds a Python 3.12 `/opt/dynamo_venv` to the standard image.
It contains only `ai-dynamo[vllm]==1.3.0.post1`, its pinned vLLM 0.23.0, etcd,
and NATS. NeMo-RL's normal Ray and engine environments are unchanged; the
standard NeMo-RL vLLM environment currently uses vLLM 0.25.1.

vLLM 0.23.0 predates PR #44814, which fixes layerwise reload accounting for
composed loaders. The installer asserts the exact vLLM version, checks and
applies the backport, and records upstream merge commit
`c9e5bf813530fb9ce06024e075da0f520b0718c8` in
`/opt/dynamo_venv/VLLM_BACKPORTS`. Remove the backport only after Dynamo pins a
vLLM release containing that fix. At that point delete the patch, application
logic, marker assertion, and backport text rather than rebasing the patch.

See [Managed Dynamo generation on Slurm](../guides/dynamo-generation.md) for
build, configuration, and launch instructions.

# Fresh-main Lightning baseline

Base: NeMo-RL main `8661b4753a19031626f4014512755061404aeb2e` (2026-10-06).
Updated to main `8bf6bd4bafc05ca29cf34981ad9c378fa38db72d` on Oct7,
including #3724. Dependency files, actor extras and recursive submodule pins
are unchanged relative to the Oct6 base.
Bridge: `ec835530efeff71b55ec015f36bcc0c33b8b52b7`.
Initialize all recursive submodules before archiving source. The only pending
production delta is #4111's model-parallel rank lookup during early NCCL source-map
construction (`23f8d5595928`), with its regression and grouped-test fixture.
No historical integration snapshot or automatic per-layer padding patch is used.

The previous BF16 baselines failed in a scalar expert-group ALLREDUCE after
initial refit and generation. A per-layer HybridEP padding collective is a
candidate, not a proven call-site attribution. These recipes instead explicitly
enable main's existing packed-input prepadding before model forward.

Both recipes use GBS512 (64 prompts x 8 generations), max length4096,
training TP2/CP2/PP1, rollout TP4/EP4, FlashInfer TRTLLM and rollout CUDA Graphs.
Sync uses 32 training GPUs/EP32; Async-1off uses 16 training GPUs/EP16 and
16 rollout GPUs with NCCL reshard. Policy/reference logprobs remain enabled.
This is a separate approved Lightning recipe, not a main performance-folder recipe.

The BF16/BF16 Sync and Async-1off jobs completed20 updates on October7.
For a same-code rollout comparison, `sync-mxfp8.yaml` and `async-mxfp8.yaml`
inherit the matching BF16 recipes and change only rollout precision and its
exclusions: routed experts in the middle layers use MXFP8; first2 and last6
layers, attention, Mamba, shared experts, router, MTP and the output head stay
BF16. Submit these arms with `ROLLOUT_PRECISION=mxfp8`. Training stays BF16.
The same pinned image, model, GBS, topology and logprob settings are retained.

Stages: immutable nightly import, recursive source preparation, GPU environment
and resolved-config gate, then matched 20-step Sync/Async BF16 baselines.
The gate rejects mismatches in lockfile, dependency declaration or actor extras.
Submodule source is deliberately overlaid at the recorded main pins, and import
paths are checked. Do not relabel this as a container-native run.
Builds/caches stay node-local;
only image, source tar, checkpoints and durable results go on shared storage.
Existing images, experiments and completed measurements are preserved.

## Nightly dependency drift

The downloaded Oct6 nightly failed the dependency gate: it contains Bridge
`1f8873bb`, MCore `6a366090` and resiliency0.6, rather than this main snapshot.
Its installed Mamba is already2.3.2.post1 despite older lockfile metadata.
This failure occurred before model initialization. It is not a new refit failure.

`align_image.sbatch` preserves that base image, replaces source with the pinned
archive, and syncs the driver and required policy/generation/Async helper actor
environments with `uv sync --locked --inexact`. Build/cache writes stay inside
the node-local writable container; the final image is saved only after imports,
GPU operations and regressions pass. The new fingerprint is recorded after
actual dependency installation, never as a substitute for installation.

The first aligned build completed installation and four-role GPU imports, but
pytest exited5 with no tests: the repository excludes `mcore` tests unless
`--mcore-only` is passed. Both gate commands now select the mcore lane explicitly;
an empty selection remains an error. The exported candidate is not a validated
image and is not manually renamed. A new build reuses its installed dependencies,
runs the corrected tests, and exports to a new immutable filename only on success.
It also checks native editable imports outside the source tree with PYTHONPATH
unset, separately from the intentional source-overlay benchmark imports.

The next build selected11 tests but their common Ray fixture failed before any
test body: dashboard MetricsHead returned EOF. Ray2.58 derives its Unix socket
under the session directory; the observed path was109 bytes, exceeding Linux's
107-byte limit. Both validation jobs now set a short, job-scoped RAY_TMPDIR in
/tmp, as the existing training launcher already does. Build/JIT caches stay in
/raid/scratch. A probe calls Ray's own path validator to reproduce the long-path
failure and accept the short path before the actual tests run. No dashboard or
test is disabled, and no model/refit production code is changed by this fix.

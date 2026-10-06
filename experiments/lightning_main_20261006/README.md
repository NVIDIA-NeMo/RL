# Fresh-main Lightning baseline

Base: NeMo-RL main `8661b4753a19031626f4014512755061404aeb2e` (2026-10-06).
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

Stages: immutable nightly import, recursive source preparation, GPU environment
and resolved-config gate, then matched 20-step Sync/Async BF16 baselines.
The gate rejects mismatches in lockfile, dependency declaration or actor extras.
Submodule source is deliberately overlaid at the recorded main pins, and import
paths are checked. Do not relabel this as a container-native run.
Builds/caches stay node-local;
only image, source tar, checkpoints and durable results go on shared storage.
Existing images, experiments and completed measurements are preserved.

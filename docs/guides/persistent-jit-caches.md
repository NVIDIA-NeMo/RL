# Persistent JIT Compile Caches

vLLM and the Megatron trainer compile kernels when they start (TorchInductor, Triton, FlashInfer, CUDA), which takes minutes on large models. This feature keeps the compiled files in a shared folder, so the next job with the same setup skips most of that work.

## Turn it on

Set two variables in the environment the Ray nodes start with:

| Variable | What it is | Example |
|---|---|---|
| `NRL_COMPILE_CACHE_SHARED_DIR` | Folder that keeps the caches between jobs. It must be visible inside the container on every node, e.g. add it to `MOUNTS`. | `/lustre/.../compile_cache` |
| `NRL_COMPILE_CACHE_LOCAL_DIR` | Fast storage on each node that the compilers use during the job. | `/tmp/nemo_rl_compile_cache` |

Each worker type has its own cache: `vllm` (the generation engines) and `megatron` (the trainer), in `<shared>/vllm` and `<shared>/megatron`. To keep one type somewhere else, set `NRL_COMPILE_CACHE_VLLM_DIR` or `NRL_COMPILE_CACHE_MEGATRON_DIR`.

Before a vLLM engine compiles, NeMo RL copies the `vllm` cache onto every node the engine runs on (once per node) and waits for it. The Megatron workers copy the `megatron` cache onto their nodes and wait for it in the same way when they start. The compilers then use `<local>/<type>`; each vLLM engine uses its own folder, `<local>/vllm_engines/<n>`, filled from the node's copy.

Writing new files back is the launcher's job. With `ray.sub`, start it in `SETUP_COMMAND`, which runs on every node before `ray start`. Start the copies there too, so they overlap Ray start-up and the workers find them done. A node does not know its role at that point, so every node copies both types:

```bash
export SETUP_COMMAND='
SYNC=/opt/nemo-rl/nemo_rl/utils/cache_sync.sh   # the checkout in the NeMo RL image
rm -rf "$NRL_COMPILE_CACHE_LOCAL_DIR"   # node storage may outlive the job
(bash $SYNC seed-type vllm; bash $SYNC seed-type megatron) >>"$(dirname "$0")/cache_seed_$(hostname -s).log" 2>&1 &
pidfile=/tmp/nemo_rl_cache_sidecar_$(basename "$(dirname "$0")").pid   # once per node, even if ray start is retried
if ! kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null; then
  bash $SYNC sidecar-types 60 "$(dirname "$0")/ENDED" vllm megatron >>"$(dirname "$0")/cache_sync_$(hostname -s).log" 2>&1 &
  echo $! >"$pidfile"
fi'
```

Both lines that start a command must end in `&`: `ray.sub` runs `SETUP_COMMAND` in the foreground, so a command without it keeps `ray start` from running. The setup script is written to the job's log folder, so `$(dirname "$0")/ENDED` is the marker `ray.sub` writes when the job ends; the write-back makes one last pass then.

`NRL_COMPILE_CACHE_FORCE_FRESH=1` starts every node empty but still writes back.

## What it covers

Covered: vLLM generation engines (also those spanning several nodes), the Megatron trainer, and the external vLLM pools (`POOL_CACHE_DIR`, see `tools/external_gym_vllm/README.md`).

Not covered: the DTensor/Automodel trainer, SGLang, and kernels compiled in the last minute of a job (write-back only takes files that have not changed for `NRL_COMPILE_CACHE_MIN_AGE_MIN` minutes, default 1).

The cache only helps a job that compiles the same thing: same model, parallelism, `max_model_len`, batch and CUDA graph sizes, GPU type, container image and vLLM version. Anything else compiles as usual; the compilers key their files by these inputs, so a job never uses a wrong entry.

## How it behaves

- Files on the shared folder are never replaced or deleted, only added. A file appears there whole or not at all (written under a temporary name, then renamed).
- Nodes take turns writing back (a lock folder next to each type's cache). A node that finds it busy tries again at the next pass; the last pass waits up to 2 minutes, then logs how many files it could not publish.
- A failed copy or write-back (for example a filesystem error) only logs a warning: the job compiles what is missing. A missing script, `rsync` or `flock`, or only one of the two variables set, stops the job at start.
- The cache only grows. A job that compiles something again (a cold run, a node without the copy) adds a second copy under new file names, and old entries stay. Reset it when it gets large.

## Reset

Delete `<shared>/<type>` (or the whole shared folder) between jobs. The next job compiles everything and writes it back.

## Replacing the script

`NRL_COMPILE_CACHE_SYNC_SCRIPT` points NeMo RL at another script, for example one that copies from object storage. NeMo RL only calls `<script> seed-type <type>`, with the variables above in the environment. The script must:

1. copy the type's cache to `<local>/<type>` and return when the copy is done;
2. be safe to call from several processes on one node at once, copying only once;
3. exit 0 when there is nothing to copy, and when a copy fails but the job can go on;
4. exit non-zero only for errors in the setup; NeMo RL then stops.

Writing back stays with the launcher.

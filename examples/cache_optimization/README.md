# Cache Persistence for Init Optimization.



See [`launch_with_caches.sh`](launch_with_caches.sh) for an example 21 nodes script.

---

## Environment Variables

### vLLM Compile Caches

| Variable | Purpose |
|----------|---------|
| `VLLM_CACHE_ROOT` | Root directory for vLLM compile caches (torch.compile, Triton kernels, GPU P2P cache). Default: `~/.cache/vllm`.|

### FlashInfer Caches

| Variable | Purpose |
|----------|---------|
| `FLASHINFER_CUBIN_DIR` | Directory for pre-compiled FlashInfer `.cubin` GPU binaries. |
| `FLASHINFER_WORKSPACE_BASE` | Directory for FlashInfer JIT workspace artifacts. |

### DeepGEMM Cache

| Variable | Purpose |
|----------|---------|
| `DG_JIT_CACHE_DIR` | Directory for DeepGEMM JIT-compiled FP8 GEMM kernels. |

### UV Package Manager

| Variable | Purpose |
|----------|---------|
| `UV_CACHE_DIR` | UV package manager download and build cache directory. |
| `UV_LINK_MODE=symlink` | Tells UV to symlink packages instead of copying them. Faster installs, less disk usage. |

### NeMo Gym Virtual Environment

| Variable | Purpose |
|----------|---------|
| `NEMO_GYM_SKIP_VENV_IF_PRESENT` | Default `1` (skip enabled). NeMo Gym hashes the dependency file and config into a marker; when it matches, `uv pip install` is skipped entirely. Set to `0` to force reinstall. |

---

## CUDA Graph Optimization

vLLM captures CUDA graphs at startup for each unique batch size it may
encounter. By default this can include a large number of sizes, which slows
down initialization. Two knobs control this:

- **`max_num_seqs`** -- limits the maximum batch size for the engine. A lower
  value means fewer CUDA graphs need to be captured.
- **`compilation_config.cudagraph_capture_sizes`** -- explicitly lists which
  batch sizes to capture CUDA graphs for (e.g., `[1,2,4,8,16]`).

These apply to both the **policy vLLM engine** and any **NeMo Gym judge
models**. Pass them as Hydra overrides on the launch command:

```bash
# Policy vLLM engine
++policy.generation.vllm_kwargs.max_num_seqs=16
++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes='[1,2,4,8,16]'

# NeMo Gym judge models (repeat for each judge)
++env.nemo_gym.genrm_model.responses_api_models.vllm_model.server_args.max_num_seqs=16
++env.nemo_gym.genrm_model.responses_api_models.vllm_model.server_args.compilation_config.cudagraph_capture_sizes='[1,2,4,8,16]'

++env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.server_args.max_num_seqs=16
++env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.server_args.compilation_config.cudagraph_capture_sizes='[1,2,4,8,16]'

++env.nemo_gym.safety_judge_model.responses_api_models.vllm_model.server_args.max_num_seqs=16
++env.nemo_gym.safety_judge_model.responses_api_models.vllm_model.server_args.compilation_config.cudagraph_capture_sizes='[1,2,4,8,16]'
```

---

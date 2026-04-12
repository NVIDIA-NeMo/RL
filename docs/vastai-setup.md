# Vast.ai Development Environment Setup

This guide walks through setting up a NeMo RL development environment on Vast.ai for running unit and integration tests. This is particularly useful for testing the `cpu_serialize` weight transfer path, which was designed for containers that lack `CAP_SYS_PTRACE` / `--ipc=host` (the default on Vast.ai).

## Machine Selection

### Template Configuration

- **Docker Image**: `nvcr.io/nvidia/nemo-rl:v0.5.0`
- **Container Disk**: 120 GB minimum (60 GB will cause `No space left on device` during dependency builds)

### GPU Requirements

Different tests have different VRAM requirements:

| Test Type | Minimum GPU | Notes |
|-----------|-------------|-------|
| Unit tests (`test_utils.py`) | 1x GPU, any VRAM | Basic transport protocol tests |
| `cpu_serialize` integration tests | 2x RTX 4090 (24 GB) | Sufficient for `cpu_serialize` path |
| `cuda_ipc` integration tests | 2x GPU with ≥40 GB VRAM (L40, A40, A100) | `cuda_ipc` requires `--ipc=host` which Vast.ai does not provide; these tests will fail on Vast.ai regardless of VRAM |

> **Note**: Vast.ai containers do not have `CAP_SYS_PTRACE` / `--ipc=host`, so `cuda_ipc` weight transfer will not work. This is the reason the `cpu_serialize` fallback path exists. The `cuda_ipc` regression tests can still be run if the underlying ZMQ transport works, but CUDA IPC handle sharing across processes may fail depending on the container configuration.

## Initial Setup

### 1. SSH into the Machine

Vast.ai provides SSH connection details on the instance page. Typically:

```bash
ssh -p <port> root@<host> -L 8080:localhost:8080
```

### 2. Clone the Repository

```bash
cd /workspace
git clone https://github.com/<your-fork>/RL.git
cd RL
git checkout <your-branch>
```

### 3. Fix Python Version Constraint

The NeMo RL container ships Python 3.12, but `pyproject.toml` may require a newer version. Fix this:

```bash
# Check current Python version
python3 --version

# If pyproject.toml requires a version higher than what's installed:
sed -i 's/requires-python = ">=3\.[0-9]*\.[0-9]*"/requires-python = ">=3.12"/' pyproject.toml
sed -i 's/requires-python = ">=3\.[0-9]*\.[0-9]*"/requires-python = ">=3.12"/' 3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/distributed/fsdp/pyproject.toml
```

> **Note**: Do not commit these changes. They are local workarounds for the Vast.ai container environment.

### 4. Set Up HuggingFace Token

Some tests download gated models. Set your HF token:

```bash
export HF_TOKEN=<your-token>
# Or persist it:
huggingface-cli login
```

### 5. Install Dependencies

```bash
pip install -e ".[dev,test]" --break-system-packages
```

## Running Tests

### Key Flags

Every test command should include these flags:

- `-o "addopts="` — Overrides `pyproject.toml` default addopts (which includes `--testmon`, not available in the container)
- `--timeout=0` — Disables timeout (first run compiles flash-attn, mamba-ssm, etc. in Ray worker venvs, which can take 10+ minutes)
- `-v -s` — Verbose output with stdout

Always stop Ray before each test to avoid stale process issues:

```bash
ray stop --force
```

### Full Test Command Template

```bash
ray stop --force && pytest <test_path> -v -s -o "addopts=" --timeout=0
```

### Unit Tests (Transport Protocol)

```bash
ray stop --force && pytest tests/unit/models/policy/test_utils.py -k "stream_weights" -v -s -o "addopts=" --timeout=0
```

### Integration Tests (cpu_serialize)

```bash
# Async refit (production path) — most important test
ray stop --force && pytest tests/unit/models/generation/test_vllm_generation.py::test_vllm_async_refit_cpu_serialize -v -s -o "addopts=" --timeout=0

# Direct ZMQ weight update
ray stop --force && pytest tests/unit/models/generation/test_vllm_generation.py::test_vllm_direct_zmq_weight_update_cpu_serialize -v -s -o "addopts=" --timeout=0
```

### Regression Tests (cuda_ipc)

```bash
ray stop --force && pytest tests/unit/models/generation/test_vllm_generation.py::test_vllm_weight_update_and_prefix_cache_reset -v -s -o "addopts=" --timeout=0
```

> **Note**: On GPUs with ≤24 GB VRAM, this test may fail with a buffer sizing assertion (`Parameter model.embed_tokens.weight too large for buffer`). This is a pre-existing limitation of the `0.3 * free_memory` buffer calculation on small-VRAM GPUs, not a regression.

## Troubleshooting

### Ray Worker Venv Issues

Ray creates isolated virtual environments for workers at `/opt/ray_venvs/`. These can become stale or have version mismatches.

**Symptom**: `ModuleNotFoundError: No module named 'megatron'` in Ray workers, or `ValueError: runtime_env_agent_port must be an integer between 1024 and 65535`.

**Fix**: Delete stale venvs and force rebuild:

```bash
rm -rf /opt/ray_venvs/*
NRL_FORCE_REBUILD_VENVS=true pytest <your_test> -v -s -o "addopts=" --timeout=0
```

If Megatron is still not found after rebuild, add `.pth` files to the worker venvs:

```bash
# Find the site-packages directories in worker venvs
find /opt/ray_venvs/ -name "site-packages" -type d

# For each one, add a .pth file pointing to the main venv's packages
echo "/opt/nemo_rl_venv/lib/python3.12/site-packages" > /opt/ray_venvs/<worker_venv>/lib/python3.12/site-packages/nemo_rl_main.pth
```

### Stale Ray Processes

**Symptom**: `ConnectionError: GCS connection timed out` or tests hang at startup.

**Fix**:

```bash
ray stop --force
# If that doesn't work:
pkill -9 -f ray
```

### Disk Space

**Symptom**: `No space left on device` during pip install or venv builds.

**Fix**: Clear caches:

```bash
rm -rf /root/.cache/pip /root/.cache/huggingface
du -sh /root/.cache/*  # Check what's using space
```

> **Important**: Be careful not to delete packages that pip depends on (like `pluggy` or `urllib3`). If pip breaks, reinstall them:
> ```bash
> python3 -m ensurepip --upgrade
> pip install --no-cache-dir pip setuptools wheel --break-system-packages
> ```

### Worker Venv Build Timeout

**Symptom**: Test appears to hang during first run (compiling flash-attn, mamba-ssm, causal-conv1d).

**Fix**: This is normal on first run. Use `--timeout=0` and wait 10-15 minutes. Subsequent runs reuse the compiled venvs.

## Test Results

Test results are saved to `tests/unit/unit_results/` with timestamps:

```bash
# List all test results
ls -lt tests/unit/unit_results/

# View a specific result
cat tests/unit/unit_results/<timestamp>.json

# Check exit status (0 = passed)
jq '.exit_status' tests/unit/unit_results/<timestamp>.json
```

## Known Limitations

1. **No CUDA IPC on Vast.ai**: Vast.ai containers lack `CAP_SYS_PTRACE` / `--ipc=host`, so `cuda_ipc` transport will not work. Use `NRL_MODEL_UPDATE_TRANSPORT=cpu_serialize` instead.

2. **Buffer Sizing on Small GPUs**: The default buffer calculation (`free_memory * 0.3`) may produce buffers too small for the largest model parameters on GPUs with ≤24 GB VRAM. This affects both `cuda_ipc` and `cpu_serialize` paths equally and is a pre-existing framework limitation.

3. **First-Run Compilation**: Ray worker venvs compile native extensions (flash-attn, mamba-ssm) on first use. Always use `--timeout=0` for the first run on a new machine.

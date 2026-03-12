# Gym Branch Review: `feature/nvidia-IF-bench-validators-integrations`

## Changes Made to the Gym Submodule

Two files modified, both backwards-compatible.

### 1. `resources_servers/turing_vif/app.py` — Dynamic Judge URL Discovery

**Problem:** When using a separate judge model (e.g., Qwen3-235B) with Turing VIF, the judge URL had to be hardcoded in the config as `judge_base_url: http://127.0.0.1:8000/v1`. However, NeMo-Gym assigns dynamic ports to all server instances at startup, so port 8000 is never actually used. The previous `vllm_model` server type with `spinup_server: true` silently accepted but never acted on those fields — no vLLM process was ever started, causing `Cannot connect to host 127.0.0.1:8000` errors at runtime.

**Fix:** Added an optional `judge_server_name` config field to `TuringVIFResourcesServerConfig`. When set, `_get_judge_client()` discovers the judge URL at runtime via `get_server_url()` from the NeMo-Gym server registry, which resolves the dynamically assigned host and port. When not set, the existing `judge_base_url` / `policy_base_url` fallback behavior is preserved.

This enables use of `local_vllm_model` (which actually spins up a vLLM instance via Ray) as the judge server type, with automatic URL resolution.

**Diff:**
```python
# New config field (TuringVIFResourcesServerConfig)
judge_server_name: Optional[str] = Field(
    default=None,
    description="NeMo Gym server instance name for the judge model. "
    "When set, the judge URL is discovered automatically from the "
    "server registry, and judge_base_url is ignored.",
)

# Updated _get_judge_client()
if self.config.judge_server_name:
    from nemo_gym.server_utils import get_server_url
    base_url = get_server_url(self.config.judge_server_name) + "/v1"
else:
    base_url = self.config.judge_base_url or getattr(
        self.config, "policy_base_url", "https://api.openai.com/v1"
    )
```

**Backwards compatibility:** Fully backwards-compatible. The new field defaults to `None`, preserving the existing behavior for all current users. The default `turing_vif.yaml` config (which uses `${policy_base_url}` interpolation) is unaffected.

### 2. `nemo_gym/profiling.py` — Lazy Import Fix

**Problem:** Top-level imports of `gprof2dot` and `pydot` cause `ModuleNotFoundError` on Ray worker nodes where these profiling dependencies are not installed. These modules are only needed when `dump()` is called, not at import time.

**Fix:** Moved the imports inside the `dump()` method so they're only loaded when actually needed.

**Backwards compatibility:** No behavioral change. The modules are imported at the same point in the execution flow — just deferred from module load time to first use.

## Experiment Configs (NeMo-RL side, not in Gym)

The following changes were made in the NeMo-RL experiment configs to work with the corrected Gym code:

- Switched judge model server type from `vllm_model` (proxy-only, `spinup_server` was a no-op) to `local_vllm_model` (actually starts vLLM via Ray)
- Replaced hardcoded `judge_base_url` with `judge_server_name: local_vllm_model` for dynamic URL discovery
- Used proper `local_vllm_model` config fields: `vllm_serve_kwargs` and `vllm_serve_env_vars`
- Switched dataset loading from inherited `OpenMathInstruct-2` to `NemoGymDataset` with `nemo_gym_data_processor`
- Fixed `cluster.num_nodes` to use policy-only node count (judge nodes managed separately by NeMo-Gym)

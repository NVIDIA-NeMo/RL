# TRT-LLM Async GRPO Runs

Sanity checks and experiments for the `shuyix-nemo-rl-trtllm` image + async GRPO recipe.

## Runs

| Date | Job | Nodes | Image | Recipe | Purpose | Log | Status |
|------|-----|-------|-------|--------|---------|-----|--------|
| 2026-07-15 | 2391178 | lyris[0217-0218] | shuyix-nemo-rl-trtllm-20260714-aarch64.squashfs | grpo-qwen3-8b-2n4g-megatron-trtllm-async-1off | Sanity check new sqsh: TRT-LLM 1.3.0rc21, Qwen3-8B-Base, async 1-off math | logs/grpo-qwen3-8b-2n4g-megatron-trtllm-async-1off/run-20260715.log | KILLED — stale Ray venvs (stdout buffered, silent hang) |
| 2026-07-15 | 2391178 | lyris[0217-0218] | shuyix-nemo-rl-trtllm-20260714-aarch64.squashfs | grpo-qwen3-8b-2n4g-megatron-trtllm-async-1off | Restart with NRL_FORCE_REBUILD_VENVS=true + python -u (unbuffered); rebased branch needs fresh Ray worker venvs | logs/grpo-qwen3-8b-2n4g-megatron-trtllm-async-1off/run-20260715-rebuild.log | DEAD — Ray GCS killed |
| 2026-07-15 | 2391922 | lyris[0253-0254] | shuyix-nemo-rl-trtllm-20260714-aarch64.squashfs | grpo-qwen3-8b-2n4g-megatron-trtllm-async-1off | New alloc. python -u + NRL_FORCE_REBUILD_VENVS=true + editable nemo-rl on both nodes | logs/grpo-qwen3-8b-2n4g-megatron-trtllm-async-1off/run-20260715-v2.log | RUNNING |

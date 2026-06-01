# SWE2 Direct — Multi-turn Perf Benchmark Plan

**Branch:** `zhiyul/swe2-perf-bench` off `origin/main` (`e94d33c88`)
**Worktree:** `.claude/worktrees/swe2_perf_bench/`
**Goal:** Benchmark the multi-turn agentic RL system end-to-end. Skip Stage 1 (SWE1 pivot); go straight to Stage 2 (SWE2 e2e) from the base HF model. Accuracy is not the target — system performance is.

## Reference recipe

`examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml` lives on `origin/binhu/swe-rl-qwen3-thinking` (1 commit ahead of main, +692 lines, doc/yaml only — no Python).

We do not need to cherry-pick: the yaml can be referenced from binhu's branch directly when launching, or copied in locally. Decide at launch time.

## What's already on `main`

Confirmed:
- NeMo-Gym submodule pin `1a4912e231b...` already includes:
  - `responses_api_agents/swe_agents/configs/swebench_openhands_training.yaml`
  - `resources_servers/single_step_tool_use_with_argument_comparison/...` (not needed for SWE2-only, but present)
  - all three agent harnesses (Codex, OpenCode, OpenHands/CodeAct)
  - `apptainer_memory_limit_mb` config knob
  - `chat_template_kwargs` plumbed into the `/tokenize` endpoint (`vllm_model/app.py:397`) — i.e. `truncate_history_thinking=false` reaches the tokenizer

So: **no Gym version bump needed.**

## What's missing on `main`

1. **`docker/Dockerfile`: apptainer + deps** (~11 lines, available as the Dockerfile chunk of `4872d6c77` on `origin/nliang/qwen3-swe-training`):
   - `libfuse3-3 uidmap squashfs-tools fakeroot` packages
   - apptainer 1.3.1 .deb install
   - `singularity` symlink to `/usr/bin/apptainer`
   - Skip the Python part of that commit (`vllm_worker_async.py` `_resolve_chat_template`) — already reverted in nliang's next commit and unneeded since binhu's recipe inlines the chat template directly into the yaml.

2. **External infra:**
   - Per-instance Apptainer `.sif` images for SWE-bench Verified (~hundreds of GB; one `.sif` per instance). Fetch script `examples/nemo_gym/download_swe_images.py` exists on `origin/super-v3` (not on main).
   - `swe2.jsonl` dataset from `nvidia/Nemotron-RL-Super-Training-Blends`.
   - 24-node cluster booking (16 train + 8 generation, non-colocated).

## Why SWE2-direct is fine *for perf*

The "reward sparsity" argument that justifies running SWE1 first only matters when the goal is **learning signal / accuracy**. For perf benchmarking the rollouts still fire, the sandboxes still spin up, the training step still runs — the gradient just isn't very useful, which we don't care about.

### The one real trap: trajectory length distribution bias

From a base model, expect a **bimodal** trajectory length distribution:
- **Short tail** (1–3 turns): format failures. Agent emits malformed tool call → env errors out → trajectory dies.
- **Long tail** (~200 turns / 30 min timeout): agent flails productively but never resolves.
- **Middle** (10–80 productive turns): sparse — that's what a trained agent does.

Implications:
- Short rollouts under-exercise the multi-turn stack — you're benchmarking sandbox startup + 2 vLLM calls.
- Long rollouts are dominated by timeout, not productive agent behavior — exercises the system but in a flat-line pattern.
- **Capture the turn-per-sample histogram alongside throughput** and flag the distribution shape.

Optional mitigation: run a quick ~50-step SWE1 first (single node, no sandbox, hours not days) just to nudge formatting past gross failures. Gives a more representative perf workload without committing to a full Stage 1.

## Launch configuration

```bash
uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe2.yaml \
  policy.model_name=Qwen/Qwen3-30B-A3B-Thinking-2507 \
  data.train.data_path=/path/to/swe2/train-split.jsonl \
  data.validation.data_path=/path/to/swe2/val-split.jsonl \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter='[/path/to/sif/sweb.eval.x86_64.{instance_id}.sif]' \
  checkpointing.save_period=999999 \
  grpo.val_period=999999 \
  grpo.val_at_start=false \
  grpo.max_num_epochs=1
```

Key overrides vs the recipe defaults:
- `policy.model_name` → base HF model (recipe placeholder is `/path/to/swe1_checkpoint_hf`).
- Disable checkpointing/validation during the perf measurement window so steady-state numbers aren't contaminated by large intermittent I/O bursts.

## Cluster topology (recipe defaults)

- **Train pool:** 16 nodes (`cluster.num_nodes: 16`), Megatron parallelism `TP×EP×CP×PP = 4×8×4×2`.
- **Generation pool:** 8 nodes, non-colocated (`policy.generation.colocated.enabled: false`).
- **Total:** 24 nodes.
- `agent_max_turns: 200`, `swebench_agent_timeout: 1800s`, `concurrency: 768` parallel sandboxes.
- `max_total_sequence_length: 131072`, sequence packing + CP.

## What to measure

Most useful steady-state numbers (after 1–2 warmup steps, then 5–10 measurement steps):

| Metric | Why |
|---|---|
| steps/hour | headline throughput |
| rollout latency P50/P95/P99 | watch for bimodal distribution |
| turns-per-sample mean + histogram | core multi-turn-system signal |
| vLLM gen tokens/s | rollout generation throughput |
| weight-sync time | in-flight update overhead per step |
| sandbox spin-up + tear-down latency | Apptainer overhead per rollout |
| train-pool GPU util | bottleneck check |
| gen-pool GPU util | bottleneck check |
| train-pool idle waiting on rollouts | **the** signal for async GRPO quality |

## In-repo work checklist

- [ ] Cherry-pick or hand-apply Dockerfile chunk from `4872d6c77` (Dockerfile only — skip the `vllm_worker_async.py` part).
- [ ] Rebuild docker image with apptainer.
- [ ] Either (a) cherry-pick binhu's `06f9783e4` for the yaml, or (b) launch with `--config` pointing at the yaml on binhu's branch (decide at launch time).
- [ ] Verify config keys exist on current `main`: `force_on_policy_ratio`, `penalize_malformed_thinking`, `invalid_tool_call_strategy`, `truncated_importance_sampling_*`, `async_grpo.in_flight_weight_updates`. Any missing keys will silently no-op or hard-fail at config-load time.

## Out-of-repo work checklist

- [ ] Build / download per-instance `.sif` images for SWE-bench Verified.
- [ ] Stage `swe2.jsonl` dataset.
- [ ] Provision 24 nodes (16 train + 8 gen).
- [ ] HF / W&B / model-cache auth on the cluster.

## Open questions to resolve before launch

1. Does any config key in the binhu recipe (`force_on_policy_ratio`, `penalize_malformed_thinking`, `invalid_tool_call_strategy`, etc.) require code that landed *after* binhu's branch base (`9e0cdfdd9`) and isn't on current `main`? Grep before launching.
2. Run a quick SWE1 warm-up (~50 steps, single node) first, or accept the bimodal trajectory distribution as-is?
3. Where to host the `.sif` images? They are large — local lustre vs `/ephemeral` vs shared dataset path.

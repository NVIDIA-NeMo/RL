# GRPO Qwen3-32B Eagle3 Resume Context (2026-02-28)

## Canonical resume script
- /home/scratch.shaunakj_other/tmp/grpo_commands/grpo32b_eagle3_resume_from_latest_ckpt.sh

## Context note location
- /home/scratch.shaunakj_other/Development/RL/docs/grpo32b_eagle3_resume_context_2026-02-28.md
- Previous temporary location: /home/scratch.shaunakj_other/tmp/grpo_commands/grpo32b_eagle3_resume_context_2026-02-28.md

## Experiment identity
- Run family: grpo-32b-eagle3-temp0p6-s4-seed124-train-normal100-ckpt10-scratchhome-live-2026-02-28-030829
- Config file: /home/scratch.shaunakj_other/tmp/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.2026-02-25-235814.seed124.normal100.local.yaml
- Main model: Qwen3-32B snapshot 9216db5781bf21249d130ec9da846c4624c16137
- Spec decode draft model: RedHatAI/Qwen3-32B-speculator.eagle3 snapshot e5756763c9b3bef3cc260cab70b76008fb42a81b
- Data: /home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl

## Checkpoints
- Checkpoint dir: /home/scratch.shaunakj_other/results/grpo-32b-eagle3-temp0p6-s4-seed124-train-normal100-ckpt10-scratchhome-live-2026-02-28-030829
- Existing checkpoints: step_10, step_20
- Last observed earlier training progress in logs: generated train_data through step_26 under exp_001

## Last known good runtime context
- Successful Ray session: /home/scratch.shaunakj_other/t/ray/session_2026-02-28_03-09-01_438227_1557241
- Node/shape in successful run: umbriel-b200-068, 8 GPUs (B200), large shared memory/object store
- Ray started runtime_env_agent successfully and proceeded into training

## Additional relaunch findings on 2026-02-28
- Login/service node relaunches failed on ipp1-1334 (no GPU env) with runtime env agent timeout / connection refused.
- GPU-node relaunches on umbriel-b200-081 reached vLLM initialization but one attempt failed with:
  - `torch.distributed.DistNetworkError`
  - `EADDRINUSE` on vLLM EngineCore TCP rendezvous port (example: 41545)
- This is distinct from the login-node failure and appears to be a startup port-collision condition.

## Script hardening added on 2026-02-28
- Added pre-launch check for `CUDA_VISIBLE_DEVICES`.
- Unset inherited distributed env vars before launch to avoid stale rendezvous bindings:
  - `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `NODE_RANK`
  - `AVAILABLE_ADDR_LIST`, `AVAILABLE_PORT_LIST`
- Added local Ray cleanup (`ray stop --force`) before each run attempt.
- Added retry loop (`max_attempts=3`) with inter-attempt cleanup:
  - `ray stop --force`
  - `pkill -f 'VllmAsyncGenerationWorker|EngineCore_DP0|run_grpo.py'`

## How to use on a new node
1. Start a GPU node allocation first.
2. Run:
   - bash /home/scratch.shaunakj_other/tmp/grpo_commands/grpo32b_eagle3_resume_from_latest_ckpt.sh
3. If running from a normal shell and you want detached logging:
   - nohup bash /home/scratch.shaunakj_other/tmp/grpo_commands/grpo32b_eagle3_resume_from_latest_ckpt.sh > /home/scratch.shaunakj_other/logs/grpo-32b-eagle3-temp0p6-s4-seed124-train-normal100-ckpt10-scratchhome-live-2026-02-28-030829/run-resume-$(date +%F-%H%M%S).log 2>&1 &

## Resume semantics
- No explicit `resume_from` flag is required in this setup.
- The code loads the latest `step_*` checkpoint from `checkpointing.checkpoint_dir` automatically.

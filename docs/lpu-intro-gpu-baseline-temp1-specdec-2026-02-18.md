# LPU Intro Presentation Draft: GPU Baseline (SpecDec temp=1)

Date: 2026-02-18

## Slide: GPU Baseline Before LPU

- Workload baseline: `Qwen3-32B` target + `Qwen3-1.7B` draft, GRPO 5-step, `temp=1`, `max_new_tokens=4096`, 8 GPUs.
- No-spec baseline: `56.53s` avg generation, `288.23` E2E tok/s (steps 2-5).
- Best GPU speculative setting: `s=2` at `41.27s` generation, `340.50` tok/s (`1.37x` generation speedup vs no-spec).
- Sensitivity to larger windows: `s=7` (`67.88s`) and `s=10` (`82.20s`) are slower than no-spec.
- Why degradation happens: TAR drops sharply (`79.04%` at `s=1` to `33.29%` at `s=10`), so draft work is increasingly wasted.
- Critical system note: verifier/target is updated each step, while draft-model weights are not transferred in this setup.
- This is the control point for LPU introduction and comparison.

## Optional Slide: Why Large Windows Hurt in This Regime

- Spec decode gives gains only while acceptance remains high.
- At `temp=1`, target/draft disagreement rises, and larger windows amplify mismatch cost.
- As window grows, wasted draft tokens and recovery overhead increase faster than accepted-token savings.
- Practical conclusion for this setup: tune for local optimum; `s=2` is best in this sweep.

## 2-Minute Talk Track

`0:00-0:25`

"Before discussing LPU, here is our current GPU control baseline on the exact workload we care about: 32B target, 1.7B draft, temp=1, and 4k decode."

`0:25-0:50`

"On GPUs, speculative decoding helps only in a narrow region. No-spec is 56.53 seconds generation; best is window s=2 at 41.27 seconds, about 1.37x faster."

`0:50-1:20`

"As we increase speculative window, performance collapses. s=7 and s=10 are slower than no-spec. Acceptance falls from about 79% down to about 33%, so we pay draft cost without enough accepted tokens."

`1:20-1:45`

"One key systems reason this matters: in our loop, verifier weights are refreshed every step, but draft-model weights are not transferred. That increases draft-target mismatch over training and hurts larger windows more."

`1:45-2:00`

"So this GPU profile is our baseline for introducing LPU: compare against no-spec and s=2, and evaluate whether LPU improves throughput while reducing this window-size fragility."

## Speaker Notes (Q&A Ready)

- Q: "Do speculative windows always help?"
  - A: No. In this run, windows `1-4` help; `7` and `10` regress.
- Q: "What metric explains the regression?"
  - A: Token Acceptance Rate collapse with larger windows.
- Q: "Is draft synchronized with verifier each step?"
  - A: Not in this setup. Refit/update path updates target model weights; draft-model weights remain static.

## Source Reports

- Main metrics report:
  `specdec-window-sweep-temp1-vs-nospec-grpo-5steps-4k-2026-02-18.md`
- Repro + artifacts:
  `repro-specdec-window-sweep-vs-nospec-grpo-5steps-4k-2026-02-18.md`

## Code Pointers (for systems claim)

- Refit/update calls in GRPO:
  `nemo_rl/algorithms/grpo.py:1046`
  `nemo_rl/algorithms/grpo.py:1049`
  `nemo_rl/algorithms/grpo.py:1061`
  `nemo_rl/algorithms/grpo.py:1062`
- NeMo vLLM update target (main model):
  `nemo_rl/models/generation/vllm/vllm_backend.py:183`
  `nemo_rl/models/generation/vllm/vllm_backend.py:236`
- Draft-model proposer behavior (no embedding/lm_head sharing in draft_model mode):
  `.venv/lib/python3.12/site-packages/vllm/v1/spec_decode/draft_model.py:68`
  `.venv/lib/python3.12/site-packages/vllm/v1/spec_decode/draft_model.py:73`
- vLLM draft-model path used by speculative config:
  `.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py:458`
  `.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py:459`

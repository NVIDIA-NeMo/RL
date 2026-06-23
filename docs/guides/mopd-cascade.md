# Multi-Teacher On-Policy Distillation (MOPD) — Cascade reproducer

This guide documents an internal experiment used to verify the correctness of NeMo-RL's
multi-teacher on-policy distillation (MOPD): a Nemotron-3-Nano student is distilled from
several larger teachers, with a **different teacher routed per environment**. The recipe
reproduces a reference Nemotron-Cascade MOPD run (originally trained on a separate codebase)
to within eval noise across math, instruction-following, and human-preference benchmarks.

## Algorithm

MOPD runs through the NeMo-Gym GRPO entrypoint with three switches:

| Knob | Value | Role |
|---|---|---|
| `grpo.adv_estimator.name` | `opd` | on-policy distillation advantage (per-token teacher/student log-prob gap) |
| `on_policy_distillation.enabled` | `true` | enable the teacher worker group |
| `on_policy_distillation.non_colocated_teachers.enabled` | `true` | teachers served on their own nodes (vocab must match the student) |
| `grpo.async_grpo.enabled` | `true` | overlap rollout + teacher logprob scoring with training |

Teachers are routed **per agent / environment** via
`on_policy_distillation.teacher_model_by_agent_name` (any agent without an explicit entry
falls back to `default_teacher`). The student, reference, and each distinct teacher are
loaded once and shared across the agents that route to them
(`deduplicate_shared_teacher_checkpoints`). Entrypoint: `examples/nemo_gym/run_grpo_nemo_gym.py`.

## Recipes

| Config (`examples/nemo_gym/`) | Launcher | Notes |
|---|---|---|
| `mopd_cascade_replicate_16n8g_98k.yaml` | `launch_mopd_cascade_98k.sh` | Faithful replication of the reference run (98k-token seqs) |
| `mopd_cascade_replicate_6n8g.yaml` | `launch_mopd_cascade_replicate.sh` | Smaller variant |

All run the same MOPD stack (async GRPO + OPD + non-colocated teachers, Megatron student with
sequence packing) on the nano-v3 NeMo-Gym environment set (math, code, MCQA, workplace,
instruction-following, structured outputs). 16 nodes × 8×H100 (cw-dfw): student + non-colocated
teachers + vLLM generation.

### Teacher routing

The defining feature of the recipe is that **each environment routes to a different teacher** —
a math-specialized teacher for `math_with_judge`, an instruction-following teacher for the
instruction / structured-output / workplace / MCQA / code environments, and a preference-tuned
(RLHF) teacher for open-ended generation — configured per agent in
`on_policy_distillation.teacher_model_by_agent_name`. See the recipe yaml for the exact per-agent
checkpoint assignments.

## Running

```bash
# cw-dfw (H100), 16 nodes. Override NUM_NODES / paths in the launcher as needed.
bash launch_mopd_cascade_98k.sh                   # full replication (98k-token seqs)
bash launch_mopd_cascade_replicate.sh             # smaller variant
```

## Results (correctness check)

The internal replication reproduces the reference Nemotron-Cascade MOPD run within eval noise,
confirming the ported algorithm is correct. The student improves substantially over its starting
checkpoint and approaches the (preference-tuned) teacher ceiling.

| Benchmark | Reference | This repo (reproduced) |
|---|---|---|
| AIME25 (avg@64) | 92.34 | 93.3 |
| IFBench (prompt / instruction, strict) | 81.74 / 83.83 | 81.3 / 83.1 |
| ArenaHard v2 (hard-prompt / creative, win-rate, no style control) | 85.5 / 71.0 | 84.1 / 69.8 |

Per-model breakdown — the MOPD student against its teachers and starting checkpoint:

| Model | AIME25 | ArenaHard (hard / creative) | IFBench (prompt / instruction) |
|---|---|---|---|
| Math teacher (AceMath) | 93.3 | 71.1 / 59.7 | 49.7 / 55.2 |
| Preference (RLHF) teacher | — | 87.5 / 80.9 | — |
| Starting checkpoint | 90.0 | 74.1 / 42.5 | 82.3 / 83.4 |
| **MOPD (step 65)** | **93.3** | **84.1 / 69.8** | **81.3 / 83.1** |

Teachers are specialists (the math teacher anchors AIME, the RLHF teacher anchors ArenaHard) and
were each evaluated on the benchmarks they anchor. The MOPD student matches the math teacher on
AIME and approaches the RLHF teacher on ArenaHard while retaining strong instruction-following —
i.e. it absorbs the strengths routed in from each teacher.

Evaluation was run with external harnesses, not this repo:
[NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills) (`ns eval`) for AIME25 and IFBench, and
[arena-hard-auto](https://github.com/lmarena/arena-hard-auto) v2.0 (GPT-4.1 judge) for ArenaHard.

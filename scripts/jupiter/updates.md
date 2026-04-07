# Distillation Config Updates

## Online Teacher Updates (`teacher_update_period`)

By default the teacher is frozen for the entire run. Setting `teacher_update_period` copies the student's current weights into the teacher every N training steps, enabling online self-distillation.

### Usage

Add to a launch script or config override:

```bash
distillation.teacher_update_period=50
```

Omitting the key (or setting it to `null`) keeps the default frozen-teacher behaviour -- existing configs are unaffected.

### How it works

After the student completes a training step, if `(step % teacher_update_period) == 0`:

1. Student weights are saved to `<checkpoint_dir>/_teacher_sync/`.
2. Teacher workers load those weights via `load_checkpoint`.
3. Teacher is offloaded back to CPU until needed for the next step's logit computation.

On **resume from checkpoint** with `teacher_update_period` set, the teacher is automatically re-synced from the student's checkpoint weights so training continues consistently.

### Files changed

- `nemo_rl/algorithms/distillation.py` -- config field, training loop logic, resume sync
- `nemo_rl/models/policy/lm_policy.py` -- added `Policy.load_checkpoint()` method

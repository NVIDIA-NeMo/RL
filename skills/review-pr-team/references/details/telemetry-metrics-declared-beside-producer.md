# telemetry-metrics-declared-beside-producer

> NeMo-RL telemetry must not name algorithm logger keys; each teed metric is declared next to the function that produces its key (not in nemo_rl/telemetry/, not per training loop), and RL owns its attribute names rather than importing lens semconv constants

When a PR adds OTel/telemetry metric plumbing, check which way the knowledge flows. `nemo_rl/telemetry/` may provide the tee and a registration call, but it must not list algorithm logger keys (`reward`, `loss`, `kl_penalty`, `critic/loss`, `vllm/*`, `timing/setup`, …). Each row is declared **beside the function that produces the key**:
- loss metrics go in `loss_functions.py`;
- `critic/loss` goes beside `_compute_critic_metrics`;
- `vllm/*` goes beside `compute_engine_step_metrics`;
- keys every loop logs (`loss`, `grad_norm`, `lr`, `reward`, …) go in the torch-free `algorithms/metric_utils.py`, as constants the producers also use.

Declare by producer, not by training loop. The single controller serves GRPO, PPO, OPD and SFT v2 and builds its `train` dict from the same producers as `grpo.py`/`ppo.py`, so a per-loop list would either duplicate rows or leave the shared loop uncovered.

Also: do not recommend importing `nemo.lens.semconv` constants for RL-owned attribute names. NeMo-RL defines its own, and docs that tell contributors to use lens's constants should be cleaned up.

**Why:** The user objected on #3655 ("ideally algorithm is decoupled from telemetry"). The same map came back as `_TRAIN_SCALARS` in #4052, and a text-search drift test (`test_every_teed_logger_key_is_emitted`) stood in for derivation: it passes if the word appears as any dict key anywhere. The user also rejected a staged ask to use lens's `GYM_ROLLOUT_BATCH_SIZE`, because it "introduces a hop and reliance on lens to define properties in RL".
**How to apply:** grep new telemetry modules for literal logger keys and prefixes. For each, name the producer function and give the table of homes. Treat an AST/grep "drift test" guarding a copy as a sign that derivation was possible. Related: [derive-key-lists-dont-restate-them](derive-key-lists-dont-restate-them.md).

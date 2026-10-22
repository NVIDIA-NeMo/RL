# Does the failure really happen

A real mechanism is not a real bug until the bad state is reachable and something reads it.

- **A race needs three things: wrong order, open gates, and the resource usable in the window; grep the other side for `suspend|pause|drain|wait_until`.** The refit suspends the engine first, so the #3397 "race" could not happen.
- **For a deadlock, map the full wait graph, including whether the waiter's need was already met before the block.** On #3480 the needed increment happened before the save, so it was a one-poll delay.
- **For "a raise skips cleanup and leaks X", write out the second entry line by line.** An `if not hasattr` attach plus a stateless loader made the retry work (#3566).
- **Finding a `raise` in upstream code does not prove it fires; read what the guard measures and look for an earlier filter.** `is_ready` was per-sample, and the controller pre-filtered fields (#3087). In-repo fakes can be stricter than production.
- **When a finding is about what a value means (inclusive bound, unit), read the consumer, not the validator.** Gym's `randint` is inclusive, so `65536` reaches `bind()` (#3369). Details: [verify-the-consumer-not-just-the-validator](details/verify-the-consumer-not-just-the-validator.md).
- **Before "this config change costs users", check the block is live.** Exemplar `mcore_generation_config` is inert under `backend: "vllm"` (#3331); `moe_aux_loss_coeff` is never read when load balancing is `"none"` (#4108).
- **Before citing a correction function as evidence, evaluate it under the config the feature forces.** `_scale_mtp_param_grads` returns early at factor 1.0 under token-level loss (#4171).

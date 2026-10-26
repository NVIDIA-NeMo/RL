# research-project-extension-points

> How a NeMo-RL research project is meant to plug into core — worker_extension_cls_fqn + subclassing a *Impl worker, not AbstractPolicyWorker

When reviewing anything under `research/`, the question "are they copy-pasting core
instead of extending it?" resolves against **one** sanctioned mechanism:

1. Subclass an undecorated `*Impl` worker (`DTensorPolicyWorkerV2Impl`,
   `MegatronPolicyWorkerImpl`) and re-wrap it with
   `@ray.remote(runtime_env=get_runtime_env_for_policy_worker(...))`.
2. Register the subclass FQN in `ACTOR_ENVIRONMENT_REGISTRY`.
3. Pass it to core's `Policy` as `worker_extension_cls_fqn=...`.

`research/template_project/template_project/worker_extension.py` +
`single_update.py` demonstrate exactly this. The `*Impl` classes are deliberately
left undecorated so research code can inherit them; the `@ray.remote` class in core
is a thin subclass.

Two things that are NOT the mechanism, and that I wrongly named first:
- `resolve_policy_worker_cls` — a narrow ModelOpt swap keyed on `quant_cfg`, looking
  up a hardcoded `POLICY_WORKER_OVERRIDES` dict. Cannot plug in a third-party class.
- `AbstractPolicyWorker` — mostly the refit path (NCCL collectives, reshard, ZMQ,
  SGLang IPC) plus a token-shaped `get_reference_policy_logprobs`.

**Why:** On PR #3408 (flow_grpo) the user asked twice whether the project used "the Impl
extension mechanism we show in the example research project." I answered the first
time from a grep and named the wrong hooks, so the answer had to be retracted.

**How to apply:** Before judging a research project's reuse, read
`research/template_project/` end to end — it IS the spec for what reuse should look
like. Then ask whether the `*Impl` base is actually inheritable: if the base's
`__init__` builds something the project cannot use (an HF causal LM on a DTensor
mesh, when the project loads a diffusers pipeline with no sharding), NOT extending it
is correct and is not a finding. Check for real duplication instead —
`configure_worker`, dist-init, LoRA/checkpoint helpers — and confirm a shared helper
exists in core before calling something a copy. See
`read-the-example-dont-grep-it` (see [claims-searches-and-numbers](../claims-searches-and-numbers.md)).

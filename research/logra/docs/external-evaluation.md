# Evaluating saved policies with an external evaluator

`configs/cua_prod4_external_eval.yaml` preserves the native training recipe and
requests an independent evaluation of each saved policy every 50 updates.
Set `CUA_EXTERNAL_EVALUATION_DIR` to a dedicated directory containing an immutable
`protocol.json` descriptor of the external inference, desktop, and scoring setup.

Rollout generation continues across external-evaluation boundaries.
The trainer waits for checkpoint finalization, then atomically writes
`requests/step-XXXXXX.json`. The request pins the checkpoint path and protocol
content hash. It is a request, not an evaluation result. Repeated requests for the
same policy and protocol are idempotent; a conflicting protocol fails explicitly.
Evaluation needs a saved positive-step checkpoint, so initial evaluation at step
zero is unsupported. On resume, `training.eval_at_start=true` can evaluate the
existing checkpoint in the previous run directory before continuing. A later
missing checkpoint fails explicitly; it never falls back to an older policy.

A separate controller must export the saved policy with AutoModel's official
`tools/offline_hf_consolidation.py`, validate the export, and run the external
evaluator. Native distributed safetensor fragments are not directly loadable as
a normal Hugging Face checkpoint. Export to a separate directory and keep the
source checkpoint unchanged. The controller owns job submission, completion,
scoring, and checkpoint retention until the export has completed.

This option does not modify the loss, optimizer, rollout collection, or policy
synchronization. Legacy inline evaluation remains the default when
`training.external_evaluation_dir` is unset. Results from different evaluation
protocols must be reported separately.

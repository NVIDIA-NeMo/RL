# infer-from-the-scheduler-before-adding-a-user-var

> Before endorsing or documenting a new env var users must set (or a new on/off flag), check whether Slurm or an already-required var already supplies it

When a PR adds a variable every user must export, or an on/off flag, first ask whether the value can be inferred. Slurm exports a lot to every job and step (`SLURM_CLUSTER_NAME`, `SLURM_JOB_ID`, …), and an already-required setting often implies the switch.

**Why:** On PR #4375 the review asked the author to *document* `CLUSTER_NAME`, and to *validate* `RAY_ALLOY_ENABLED` as 0|1. The user pushed back on both. Asking everyone to set a variable is tedious and breaks across launch scripts; `SLURM_CLUSTER_NAME` already holds the cluster's `ClusterName` (confirmed with a probe job: it is set in both the batch script and the `srun` step). The flag was not needed at all, because export can turn on whenever the required `LOKI_PUSH_URL` is set. One trap: "any non-empty value means on" is wrong for a flag whose documented default is `=0`.
**How to apply:** For each new user-facing variable, grep the scheduler's output env vars and the PR's other required vars before writing a "document it" or "validate it" ask. Prefer `${X:-$SCHEDULER_VALUE}` as an override with a default, or key the feature on a setting users must provide anyway. Probe a real cluster when the answer depends on site config (the lesson in `failed-search-is-not-a-negative-fact` (see [claims-searches-and-numbers](../claims-searches-and-numbers.md)) applies to env vars too).
**Seen:** #4375 C5 CLUSTER_NAME (user); #4375 C3 RAY_ALLOY_ENABLED flag (user)

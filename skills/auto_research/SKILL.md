---
name: auto-research
description: Autonomous NeMo RL experiment driver for accuracy-focused recipe search in this repository. Use when coding agents should inspect recipe inheritance and backend code paths, run repeatable experiments, and record each hypothesis on its own git branch under a shared prefix with a TSV experiment log.
---

# Auto Research

Run iterative NeMo RL experiments in this repository with accuracy as the objective and git as the ledger.

Treat dependencies as ready, but choose the runtime deliberately. Use the recipe's authoritative metric as the source of truth. Keep changes small, reproducible, and simple. Preserve unrelated user work.

## Workflow

1. Inspect the current git state and identify unrelated user changes before branching.
2. Use a shared branch prefix. Prefer a user-provided one; otherwise create a suggestive default such as `autoresearch/2026-03-24-dapo-qwen2p5`.
3. Read the target recipe, its parents, and the relevant code paths in `examples/run_grpo.py`, `nemo_rl/models/`, `nemo_rl/algorithms/`, and `docs/`.
4. Translate any user stop rule into explicit values you can monitor, such as `target_experiments`, `campaign_deadline`, `per_experiment_timeout`, or `target_metric`.
5. Verify required data, checkpoints, runtime inputs, and the launcher.
6. Create an untracked TSV log and per-experiment log directory.
7. Run a baseline first on `<prefix>/baseline` if none exists.

For GPU, CPU-heavy, distributed, or long-running work, use the relevant runtime skill before launching: `slurm` on a Slurm head node or `launch-nemo-rl` for nrl-k8s/Kubernetes. Run locally only for light inspection, dry runs, and short non-GPU checks.

## Branching

- Put every experiment on its own branch under the shared prefix.
- Keep every branch, even for failed or weak ideas.
- Put at least one commit on each branch for the hypothesis.
- Add follow-up fix commits on the same branch when a rerun is justified.
- Never stash, reset, or overwrite unrelated user changes silently. If dirty files overlap the experiment, use a separate worktree or ask before proceeding.

See `references/git-workflow.md` for the exact pattern.

## Loop

1. Pick one concrete hypothesis.
2. Create a branch such as `autoresearch/2026-03-24-dapo-qwen2p5/prompt-compact-schema`.
3. Edit the smallest set of files needed.
4. Commit the hypothesis.
5. Before launching the run, check the monitored stop conditions. Do not stop early unless one is already clearly met.
6. Identify the authoritative metric source from the recipe or logging code, then run with a unique log path:

```bash
LOG_DIR=reports/auto_research/<campaign>/<experiment>
mkdir -p "$LOG_DIR"
uv run <entrypoint> > "$LOG_DIR/run.log" 2>&1
```

7. If the user gave a per-experiment wall-clock limit, enforce it explicitly. Prefer a recipe-level timeout when one already exists; otherwise wrap the command with an external timeout. If both exist, honor the tighter limit.
8. Extract the primary metric with a command appropriate for the actual log format. If extraction is empty, inspect the last log lines and the recipe's logging path before marking the run.
9. Record branch, parent commit, commit, recipe, metric name, metric value, memory, elapsed time, launcher or job id, command, log path, status, and description in the TSV, along with enough timing or count information to evaluate the stop rule.
10. Periodically print user-facing progress updates during the campaign. Include the current branch, latest known result, attempted experiment count, remaining experiment count if applicable, remaining campaign time if applicable, and whether any stop condition has been met yet.
11. Re-check the monitored stop conditions after the experiment completes and state the result explicitly, for example `stop condition not yet met: 17/24 attempted, 6h12m remaining` or `stop condition met: 24/24 attempted`.
12. Mark the result as `keep`, `discard`, or `crash`, then move to the next branch unless a user-specified stop condition has been clearly met.

For count-based stop rules, count attempted ideas, not only successful or fully completed runs.

For campaign time budgets, convert the user limit into an absolute deadline at the start of the campaign and keep checking remaining time.

For per-experiment budgets, enforce a timeout on every run and treat overruns as failures.

Examples:
- `do 50 experiments`: stop only after 50 attempted experiment rows exist in the TSV
- `10h total, 1h each`: enforce a 1 hour limit per run and stop when the 10 hour campaign budget is reached, or when there is not enough remaining budget to start another 1 hour run
- `50 experiments or 10h total, 1h each`: monitor all three values, never exceed the per-run cap, and stop only when one campaign-level stop trigger is clearly reached

## Priorities

Prefer ideas with high expected accuracy gain and low complexity cost:
- correctness and backend compatibility
- prompt and rollout formatting
- batch, sequence, and precision layout
- optimizer and scheduler tuning
- reward shaping, clipping, or scaling
- dataset mix or validation changes
- synchronous versus asynchronous execution based on hardware

All else equal, prefer simpler wins and avoid brittle hardware-specific hacks.

## Stop

If the user gives explicit stopping conditions, they override the generic rule. Do not stop because the search feels sufficient; stop only when the requested count, deadline, budget, or target condition has been clearly met.

During the campaign, explicitly inform the user whether the stop condition has been met. If not, report the remaining count, remaining time, or other remaining threshold in concrete terms.

If the user does not give explicit stopping conditions, run the baseline plus up to three low-risk experiments, then summarize the best result and ask before continuing.

## References

- `references/git-workflow.md` for branch, dirty-worktree, parent-commit, and baseline rules.
- `references/exploration-ideas.md` for turning symptoms into concrete hypotheses.
- `references/experiment-log-template.md` for the TSV schema and reproducibility fields.

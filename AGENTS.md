# NeMo-RL

NeMo-RL is an RLHF training framework built on Ray and PyTorch (FSDP2 / Megatron-Core). It supports algorithms like GRPO, DPO, and SFT for LLMs and VLMs.

## Skills

Coding guidelines and operational procedures are organized as Claude skills in
two locations:

- `skills/` — customer-facing operational skills (launch-nemo-rl, auto-research, brev-etiquette, docs)
- `.agents/contributor-skills/` — contributor-facing development guidelines (testing, linting, CI/CD, review, etc.)

All skills are symlinked into `.claude/skills/` for unified discovery.
**Always read the relevant `SKILL.md` before starting any task it
covers — skills are mandatory context, not optional background reading.**

**Workflow — mandatory order for every task:**
1. **Pull information first.** Read the commit, PR, error log, file, or
   whatever artifact the task is about. Do not reason about it yet.
2. **Select and invoke the skill.** Based on what you just read, identify
   the relevant skill and invoke it before forming any answer or plan.
3. **Answer or implement.** Only after the skill is loaded, use its context
   to reason, diagnose, or write code.

Never skip or reorder these steps. Do not wait for the user to name the right
skill keyword — infer it from the artifact you read.

## Code Review

Use `/review-pr <pr-number>` for interactive local PR review.

When reviewing code, follow these principles:

- **Be concise and actionable.** Focus on bugs, logic errors, missing tests, outdated docs, and guideline violations. Lead with the issue, then a concrete suggested fix (name the file/function; give the snippet/schema when known).
- **Do NOT flag:** style/formatting (linters handle it), minor naming suggestions, *subjective* architecture debates, or performance unless there is a clear measurable issue.
- **DO flag concrete maintainability smells that have a clear fix** (these are not "architectural opinions"): side-channel/out-of-band state (e.g. setting an attribute the caller reads via `hasattr`), `dict[str, Any]` for known-field configs, inconsistent return types across siblings, and silent-on-misconfiguration paths. Raise them as low-severity suggestions.
- **High confidence only — but verify the premise.** Only flag issues you are confident about; if unsure, skip it. When a finding hinges on how an external tool/API/env-var behaves, verify it against source/docs (with a permalink) before flagging — don't inflate confidence on an unverified premise.
- **Verify upstream API usage.** When code calls into megatron-bridge, megatron-lm, automodel, or gym APIs, look up the actual API to verify correct usage. Evaluate each such call with scrutiny — don't assume the author got the signature, return type, or semantics right.
- **Compare new components to their nearest internal analog.** A new worker group, estimator, or config block should mirror the established pattern (backend dispatch, override hooks, guards, typing, return shape) or justify diverging.
- It is perfectly acceptable to have nothing to comment on. Say "LGTM" if so.

## Kubernetes / nrl-k8s

For launching, monitoring, stopping, and debugging NeMo-RL recipes on Kubernetes, see the skill at @skills/launch-nemo-rl/SKILL.md.

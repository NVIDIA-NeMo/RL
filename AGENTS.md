# NeMo-RL

NeMo-RL is an RLHF training framework built on Ray and PyTorch (Automodel / Megatron-Core). It supports algorithms like GRPO, DPO, and SFT for LLMs and VLMs.

## Skills

Coding guidelines and operational procedures are organized as agent skills in `skills/`, one directory per skill with a `SKILL.md`. `.claude/skills` and `.agents/skills` are symlinks to `skills/`, so every agent discovers the same set. **Always read the relevant `SKILL.md` before starting any task it covers — skills are mandatory context, not optional background reading.**

**Workflow — mandatory order for every task:**
1. **Pull information first.** Read the commit, PR, error log, file, or whatever artifact the task is about. Do not reason about it yet.
2. **Select and invoke the skill.** Based on what you just read, identify the relevant skill and invoke it before forming any answer or plan.
3. **Answer or implement.** Only after the skill is loaded, use its context to reason, diagnose, or write code.

Never skip or reorder these steps. Do not wait for the user to name the right skill keyword — infer it from the artifact you read.

Any task that writes or changes code also loads the `contributing` skill and reads every `skills/contributing/references/*.md` file. A rule's `Contributing Details:` link has step-by-step instructions; read it when the rule applies. A rule's `Review Details:` link points to the review lesson it came from; reading it is not needed to write code.

## Code Review

For formal GitHub reviews, comment `/review` or `/review mode=strict` on the PR. See CONTRIBUTING.md for service integration prerequisites. The existing repository review guidance and local interactive commands below are unchanged.

Use `/review-pr-team <pr-number>` for interactive local PR review by a coordinated team of specialized agents (requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`). Related: `/review-pr-team-with-html <pr-number>` adds a public explainer page built with `/pr-html-concise-public`, and `/review-pr-team-verify-followup <pr-number>` checks the author's fixes after a published review and scans the new commits for major issues they introduced.

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

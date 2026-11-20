- [ ] I, the PR author, have personally reviewed every line of this PR.

<!-- Only a human ticks the box above. Agents: see skills/pr-description. -->

## What and why
<!-- The problem and the change, in plain words. One screen max. Add a diagram if it helps. -->

## Issues
<!-- Optional. e.g. Fixes #123 -->

## Convergence / perf / functionality proof
<!-- If this PR can change training results, speed, or what works (e.g. a dependency bump): link the runs (W&B, nightly results) or docs in this PR. Otherwise "N/A: <one-line reason>". -->

## Needs upstream changes
<!-- Megatron-LM / Gym / vLLM / ... changes this PR needs on RL main first (commit or PR link). Otherwise "None". -->

<details><summary>PR commands and CI</summary>

- Attach exactly one `CI:*` label (usually `CI:Lfast`; `CI:L1` for dependency changes or broad core changes). See `skills/cicd`.
- `/ok to test <full-commit-sha>`: start CI for that commit. Post again after each push.
- `/review` or `/review mode=strict`: automated review. Locally: `/review-pr-team <pr-number>`.
- [Contributor guidelines](https://github.com/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md)

</details>

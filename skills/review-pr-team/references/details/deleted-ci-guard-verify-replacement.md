# deleted-ci-guard-verify-replacement

> When a PR deletes a CI check, verify what it actually guarded AND that the claimed replacement runs under the same trigger conditions (always-on vs gated)

A PR that deletes a CI workflow usually claims an existing mechanism replaces it. Verify two separate things: (1) what bug class the deleted check actually guarded (read the deleted YAML + the script it ran — it may guard something narrower than its name suggests), and (2) whether the claimed replacement has the same *trigger profile* — in NVIDIA-NeMo/RL, `cicd-main.yml` jobs are gated behind copy-pr-bot `/ok to test` and can be skipped via `image_tag` overrides, so "the container build catches it" is weaker than an ungated `pull_request` workflow.

**Why:** In PR #3235, the deleted `mbridge-deps-sync.yml` was always-on/pre-ok-to-test; its replacement (`uv sync --locked` in the Docker build) was gated and skippable. The devil's advocate also found the reverse error: the "gap" was partly pre-existing (the deleted check never validated lock freshness at all) — so both the PR's claim and the finding needed refinement.

**How to apply:** for deleted checks, diff trigger conditions (`on:`, path filters, gating) between the deleted workflow and its replacement; check branch protection required-status-checks for orphaned entries (`gh api repos/<repo>/branches/main/protection/required_status_checks`); and classify the gap as PR-introduced vs pre-existing before assigning severity.

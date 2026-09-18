# Sharing branch validation — 2026-09-18

- Focused RL suite: **64 passed**.
- Gym compact diagnostics suite: **3 passed**. The intentional filesystem-error
  test logged its expected warning; pytest also noted an unused asyncio option
  because plugin autoload was disabled.
- Changed/added Python files: parsed successfully. Shell scripts: `bash -n` passed.
- Both repository diffs: `git diff --check` passed.
- All three families' agent/resource/config entrypoints are included.
- The Bridge compatibility patch reverse-checks against the original locally
  patched source. No additional Bridge branch or changed gitlink is required.

Reproduce the focused tests with `tools/test_mixed_visual_lightweight.sh` and
Python3.13 plus pytest, PyYAML, Hydra/OmegaConf, Pydantic, and Pillow. Test packages
were installed in a separate environment, not the pinned training runtime.
The RL tests cover manifest validation, provenance, metrics, processor assets,
recipes, launcher/build helpers and concurrent encoder access. They do not
substitute for the retained heavyweight multimodal/Gym/model regression suites.

No GPU jobs, real-model rollouts, distributed updates, checkpoint reload, full
MasterConfig import, or whole-repository pre-commit suite were run for this
packaging task. The historical 220-update mixed run predates these exact sharing
commits. Qualification of a colleague's container/assets remains necessary.

The branches contain source and a build recipe for VisGym, not its candidate
wheel. The existing dependency redistribution caveat is preserved in Gym.

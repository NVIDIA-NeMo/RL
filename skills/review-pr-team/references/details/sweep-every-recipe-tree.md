# sweep-every-recipe-tree

> When a PR says "every X now uses Y", resolve every shipped config, not the tree the PR globbed, and check the PR's own guard test covers them all

**Why:** a guard test that globs one directory is green by construction on the directories it never
sees. The PR then reads as complete while recipes elsewhere still carry the old value, and the old
name lingers in schemas, fixtures, docs and test-suite labels.

**How to apply:**

1. List every place that ships configs: `examples/configs/recipes/llm/`, `examples/configs/recipes/vlm/`,
   `examples/nemo_gym/`, `research/`. Resolve each through its `defaults:` chain and select the key
   the PR migrates; report what still has the old value.
2. Grep the old key or value outside the YAMLs too: schema declarations in `nemo_rl/`, fixtures in
   `tests/`, `docs/`, and the labels in `tests/test_suites/*.txt`.
3. If the PR adds a guard test, check its glob root. An invariant over every config belongs in
   `tests/unit/test_config_validation.py` with `load_config_with_inheritance`: parametrize an old-value
   guard over `full_config_files` (every shipped tree, including `examples/nemo_gym/` and `research/`), and a
   schema check over `config_files` (`examples/configs` only, which includes `recipes/vlm/`).
4. In the comment, list each file left behind with its line, and say which trees you checked.

**Seen:** #4037 a DeepEP to HybridEP migration whose guard test globbed only `recipes/llm`, leaving three AutoModel EP VLM recipes on `dispatcher: deepep`; `enable_deepep` stayed in the schema and a guide and nightly labels still said DeepEP (team)

Related: [diff-removals-are-a-work-list](diff-removals-are-a-work-list.md).

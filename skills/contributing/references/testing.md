# Testing

How to add and change unit, functional and nightly tests.

## Unit tests

- **Add `# pragma: no cover` on the `class` or `def` line of anything under `nemo_rl/` decorated with `@ray.remote`.** These run in separate Ray processes that coverage does not track reliably. Contributing Details: [testing](details/testing.md).
- **Do not test recipe or test-script settings: a unit test that only asserts a key of an existing config, or an argument of an existing functional or nightly test script, has a given value. Do not unit-test functional test scripts at all; running them is their test.** A recipe or script change always has a reason, so such a test only fails CI for the next intentional change. A test that keeps every config away from a known issue is fine, for example `test_all_config_no_tp_size_accuracy_issues`, which stops a new config from hitting the known TP>=4 accuracy issue. (#4037)
- **Do not add a unit test that only checks a config under `examples/configs/` loads or validates.** `test_all_config_files_have_required_keys` in `tests/unit/test_config_validation.py` already loads every one with inheritance and validates it against its algorithm's `MasterConfig`, and `test_single_controller_configs_pass_single_controller_validation` runs `validate_single_controller_config` on every SingleController config. If a new kind of config needs a check, extend those tests.
- **Do not add a unit test that only asserts a config field's default value.** A default change is a behaviour change for every run that omits the key; its reason belongs in the PR, not in a test that fails on it. Test what the code does with the value instead, for example a validator rejecting a combination or a property derived from it.
- **Do not add a unit test that only checks what pydantic already does for a config: omitted fields falling back to their defaults, a passed value reading back unchanged, or a missing required field or a value outside its `Literal` or type being rejected.** That is pydantic's job, and the central config tests already load every shipped config. Test it only when our own code sits between the input and the model, such as a custom merge or a `mode="before"` validator. Next to a validator of ours, a test that an edge value is accepted is fine; constructing it without an error is the check, so do not read the value back.
- **Put a research project's unit tests under `research/<project>/tests/`.** CI syncs the project's runtime dependencies and `test` group before running them. Contributing Details: [testing](details/testing.md).

## Functional tests

- **Put a research project's functional tests under `research/<project>/tests/`.** CI discovers them there, the same way as its unit tests. Contributing Details: [testing](details/testing.md).

## Nightly tests

- **When you add support for a new model, add a nightly test: a recipe YAML under `examples/configs/recipes/{llm,vlm}/`, a driver script with the same base name under `tests/test_suites/{llm,vlm}/`, and the script's path in `tests/test_suites/nightly.txt`.** Contributing Details: [testing](details/testing.md).
- **Name the recipe and its script by the recipe naming rules: `<algo>-<model>-<nodes>n<gpus>g-<strategy-and-params>[-modifiers][-long][.vN]`, with a `vlm_` prefix for VLM.** Contributing Details: [testing](details/testing.md).
- **Put a research project's suite scripts under `research/<project>/tests/test_suites/` and its recipes under `research/<project>/configs/recipes/`, and list each script by its repository-relative path in the root suite lists (`nightly.txt`, `disabled.txt`, etc.).** Contributing Details: [testing](details/testing.md).

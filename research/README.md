# Research and Community Projects

This directory contains research experiments and community-contributed projects built on NeMo RL. Each project is self-contained and demonstrates different techniques and applications.

## Getting Started

To create a new research project, start with the template:

```bash
cp -r research/template_project research/my_new_project
```

The template includes:
- A minimal train-and-generate loop example:
    - Main loop: [single_update.py](research/template_project/single_update.py)
    - Utilities used by the main loop: [template_project/](research/template_project/template_project)
- Configuration examples: [configs/](research/template_project/configs)
    - The subdirectory [configs/recipes/](research/template_project/configs/recipes) is used only for test suites
- Documentation template: [README.md](research/template_project/README.md)
- Complete test suite structure (unit, functional, and test suites): [tests/](research/template_project/tests)
- Dependency specification: [.python-version](research/template_project/.python-version) and [pyproject.toml](research/template_project/pyproject.toml)

## What Needs To Be Provided

A new research project needs to include at least:
- Driver script and main loop
    - You can refer to [run_grpo.py](examples/run_grpo.py) and [grpo.py](nemo_rl/algorithms/grpo.py) in the core repository, and [single_update.py](research/template_project/single_update.py) in the research template for examples.
- Configuration
    - A runnable `config.yaml` that defines the experiment.
    - You can refer to [examples/configs/](examples/configs) in the core repository and [configs/](research/template_project/configs) in the research template for examples.
- Documentation
    - A `README.md` that describes the project, how to run it, and how to reproduce results.
- Functional test
    - An end-to-end test with minimal configuration to ensure that changes elsewhere do not break the research project.

The following are optional:
- Unit tests and test suites (adding these is encouraged).
- Dependency specifications (required if the project’s "driver" dependencies differ from the core `nemo_rl` package).

## Expectations for Research Project Authors

> [!NOTE]
> This section is for research and community project authors contributing to the repository.

### Acceptance Criteria

The acceptance criteria for merging your research project into the main repository are reproduction steps for the results outlined in this README. We want to make sure others can reproduce your great work! Please include sufficient documentation in the README.md that enables users to follow and reproduce your results step-by-step.

> [!NOTE]
> We strongly encourage you to consider contributing universally applicable features directly to the core `nemo_rl` package. Your work can help improve NeMo RL for everyone! However, if your innovation introduces complexity that doesn't align with the core library's design principles, the research folder is exactly the right place for it. This directory exists specifically to showcase novel ideas and experimental approaches that may not fit neatly into "core".

### Code Reviews and Ownership

Code reviews for research projects will always involve the original authors. Please add your name to the `.github/CODEOWNERS` file to be alerted when any changes touch your project. The NeMo RL core team reserves the right to merge PRs that touch your project if the original author does not respond in a timely manner. This allows the core team to move quickly to resolve issues.

### Testing

Authors are encouraged to write tests for their research projects. This template demonstrates three types of tests:
1. **Unit tests** - Fast, isolated component tests
2. **Functional tests** - End-to-end tests with minimal configurations
3. **Test suites** (nightlies) - Longer-running comprehensive validation tests

All of these will be included in our automation. When changes occur in nemo-rl "core", the expectation is that it should not break tests that are written. 

In the event that we cannot resolve test breakage and the authors are unresponsive, we reserve the right to disable the tests to ensure a high fidelity test signal. An example of this would be if we are deprecating a backend and the research project has not migrated to its replacement. 

It should be noted that because we use `uv`, even if we must disable tests because the project will not work top-of-tree anymore, a user can always go back to the last working commit and run the research project with nemo-rl since the `uv.lock` represents the last known working state. Users can also build the Dockerfile at that commit to ensure a fully reproducible environment.

## Projects

- **[template_project](template_project/)** - A starting point for new research projects with example code and test structure

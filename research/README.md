# Research and Community Projects

This directory contains research experiments and community-contributed projects built on NeMo RL. Each project is self-contained and demonstrates different techniques and applications.

## Getting Started

To create a new research project, start with the template:

```bash
cp -r research/template_project research/my_new_project
```

The template includes:
- A minimal train-and-generate loop example
- Complete test suite structure (unit, functional, and test suites)
- Configuration examples
- Documentation template

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

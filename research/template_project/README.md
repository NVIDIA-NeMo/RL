# Template Project: A Starting Point

This is a template project for research experiments with NeMo RL.

> [!IMPORTANT]
> This is a template! To start a new research project, copy this directory to a new location:
> ```bash
> cp -r research/template_project research/my_new_project
> ```
> Then add your code and tests! Note that this project includes `nemo-rl` as a core dependency.

## What This Shows

The `single_update.py` script demonstrates a minimal train-and-generate loop:
1. Sets up a Ray compute cluster
2. Initializes vLLM generation and an LM policy
3. Trains the policy on a small batch using NLL loss
4. Refits the generation engine with the updated policy weights
5. Generates outputs with the new policy
6. Repeats the loop (10 iterations by default)

This shows the basic cycle of training a language model and using it for generation.

## Running the Example

To run the `single_update.py` script:

```bash
uv run single_update.py
```

## Testing

This project includes a comprehensive test suite following NeMo RL's testing patterns.

### Unit Tests

Unit tests validate individual components and functions.

```bash
# Run all unit tests
uv run --group test pytest tests/unit/
```

### Functional Tests

Functional tests run end-to-end scenarios with minimal configurations. These tests require GPU access.

> [!IMPORTANT]
> Functional tests require at least 1 GPU to run.

```bash
# Run the single_update functional test (runs for 1 step)
uv run bash tests/functional/single_update.sh
```

### Test Suites

Test suites are longer-running comprehensive tests designed for validation on multiple steps.

> [!IMPORTANT]
> Test suites require 8 GPUs and may take several minutes to complete.

```bash
# Run the single_update test suite locally (runs for 10 steps on 1 node with 8 GPUs)
bash tests/test_suites/llm/single_update_1n8g.sh

# Launch on SLURM with code snapshots
# For full documentation on tools/launch, see:
# https://github.com/NVIDIA-NeMo/RL/blob/main/tests/test_suites/README.md#launching-with-code-snapshots
bash ../../tools/launch tests/test_suites/llm/single_update_1n8g.sh

# Dry run to estimate GPU hours needed
DRYRUN=1 bash ../../tools/launch tests/test_suites/llm/single_update_1n8g.sh
```

> [!TIP]
> The `tools/launch` script creates code snapshots and launches SLURM jobs for reproducible experiments. It automatically extracts the configuration from your test suite script and submits the appropriate number of jobs.

The test suite structure mirrors nemo-rl's test organization:
- `tests/unit/` - Fast, isolated unit tests
- `tests/functional/` - End-to-end tests with minimal configurations
- `tests/test_suites/llm/` - Comprehensive multi-step validation tests
- `configs/recipes/llm/` - Configuration files for test suites (using defaults to inherit from base configs)

## Updating Dependencies

If you update the dependencies of this research project, run the following command to update the global `uv.lock` file and freeze the working set of dependencies:

```bash
uv lock
```

This command will:
- Resolve all dependencies
- Update `uv.lock` with the latest compatible versions
- Ensure dependency consistency across environments

## Python Version

> [!NOTE]
> This project uses Python 3.12 as specified in `.python-version`.  
> This Python version should always be kept in sync with the `.python-version` file at the root of the `nemo-rl` repository to ensure compatibility.


## Citation

If you use this research project or have questions, please contact:

```
Author: AUTHOR NAMES HERE
Email: AUTHOR EMAILS HERE
Organization: ORGANIZATION HERE (optional)
```

If you use this research project, please cite it using the following BibTeX entry:

```bibtex
@misc{template-project,
title = {Template Project: A Starting Point},
author = {AUTHOR NAMES HERE},
howpublished = {\url{https://github.com/NVIDIA-NeMo/RL/tree/main/research/template_project}},
year = {2025},
note = {Research project based on NeMo RL},
}
```

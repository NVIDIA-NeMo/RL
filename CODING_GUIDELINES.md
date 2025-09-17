
# NeMo-RL Coding Guidelines

Note: This repository is Python-first. Prefer the Python guidelines in this document.

## Style Guides We Follow

- Python: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Shell: [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

## uv Guidelines

### Use uv run instead of python

Use `uv run` to execute scripts, rather than activating a virtual environment and calling `python` directly.

Don't:

```bash
source .venv/bin/activate
python examples/run_grpo_math.py
```

Do:

```bash
uv run examples/run_grpo_math.py
```

Exception: `Dockerfile.ngc_pytorch` is exempt from this rule.

## Python Coding Guidelines
### Python Standard
1. The code developed for NeMo RL should conform to Python 3.12+.

### Indentation
1. Indent code with 4 spaces. Do not use tabs.

### Naming

#### Identifier Format
1. Files
- snake_case: `some_file.py`

2. Classes
- PascalCase: `class SomeClass`

3. Functions and Methods
- snake_case: `def my_awesome_function():`

4. Local Variables
- snake_case: `my_variable = ...`
- prefix `k` for variable names that start with a number: `k_99th_percentile = ...`

5. Global Variables
- upper snake_case and prefix `G`: `G_MY_GLOBAL = ...`

6. Constants
- upper snake_case: `MY_CONSTANT = ...`

#### Identifier Guidelines
1. Avoid shadowing variables declared in an outer scope.
2. Initialize all externally visible memberes of a class in the constructor.

### Comments

1. For interfaces that may be used outside a file, prefer docstrings over comments.
2. Comments should be reserved for code within a function, or interfaces that are local to a file.
3. If a piece of code is commented out, there should be a comment around that piece of code describing it's usage and why it's commented out. Otherwise that is a debug comment and it should be removed before merging

### Docstring Syntax
#### Classes and Functions
Use the [Google style](https://google.github.io/styleguide/pyguide.html), which can be parsed by Sphinx.

### Avoid Reflection
Avoid using reflection when functionality can be easily achieved without reflection.

For example, instead of:

```python
def make_complex(*args):
    x, y = args
    return dict(**locals())
```

Do:

```python
def make_complex(x, y):
    return {'x': x, 'y': y}
```

### Error Handling
1. When using try-except blocks, limit the except to the smallest set of errors possible.

For example, instead of:

```python
try:
    open(path, "r").read()
except:
    print("Failed to open file")
```

Do:

```python
try:
    open(path, "r").read()
except FileNotFoundError:
    print("Failed to open file")
```


2. When using try-except blocks to handle multiple possible variable types (i.e. duck-typing), keep the body of the try as small as possible, using the else block to implement the logic.

For example, instead of:

```python
try:
    f.seek(0)
    f.read()
except AttributeError:
    ... # Not a file-like object, do something else
```

Do:

```python
try:
    f.seek # Do not call to minimize chance of unrelated failure
except AttributeError:
    ... # Not a file-like object, do something else
else:
    f.seek(0)
    f.read()
```

### Configuration Defaults

- **YAML is the single source of truth for defaults.** Do not set non-`None` defaults in the code for configuration values. The loaded YAML (and any user overrides) must supply required values.
- **Access config directly and expect presence.** For required attributes, write code like `policy_cfg["precision"]` and assume it is present. Do not introduce hidden defaults deep in the code (e.g., defaulting `policy.precision` to `"bfloat16"`).
- **Express optionality via `TypedDict`.** Use `typing.NotRequired` to mark optional attributes. Optional attributes may be absent/`None`; code may check for their presence.
- **Where defaults live.** Exemplar configs under `examples/configs/*.yaml` include documented defaults. Recipe YAMLs under `examples/configs/recipes/**/*.yaml` are runnable snapshots and may omit documentation.
- **Additions must be documented.** When adding a new config key to a `TypedDict` subclass, document the key’s purpose, valid values/types, and recommended default (if applicable), and reflect the default in the exemplar YAMLs under `examples/configs/*.yaml`.
- **Rationale.** Centralizing defaults in YAML avoids surprising behavior and makes value provenance clear.

Forbidden patterns:

```python
# Hidden default in code
precision = policy_cfg.get("precision", "bfloat16")

# Function parameter defaulting a config value
def build_policy(policy_cfg, precision: str = "bfloat16"):
    ...
```

Preferred patterns:

```python
# Required attribute: expect it to come from YAML or user override
precision: str = policy_cfg["precision"]

# Optional attribute: check for presence
if "milestones" in scheduler_cfg:
    configure_milestones(scheduler_cfg["milestones"])
```

See also: [TypedDict and Configuration Defaults](docs/design-docs/design-and-philosophy.md#typeddict-and-configuration-defaults).

## Doc Guidelines

### Ensure docs/index.md is up to date

When a new markdown doc is added under `docs/**/*.md` or a markdown file is renamed, ensure that `docs/index.md` is updated and the document appears in the most appropriate section.

## Tests

### Coverage and Ray Actors

- For any source file under `nemo_rl/*.py` that defines a class or function decorated with `@ray.remote`, add a coverage pragma because these run in separate Ray processes and are not reliably tracked by coverage.
- Place `# pragma: no cover` on the `class` or `def` line (and on any remote functions), for example:

```python
import ray

@ray.remote  # pragma: no cover
class RolloutActor:
    def run(self) -> None:
        ...

@ray.remote  # pragma: no cover
def remote_eval(batch):
    ...
```

### Nightly Tests for New Model Support

When adding support for a new model, add a corresponding nightly test consisting of:

1) Recipe YAML under `examples/configs/recipes/`
- Place the YAML in the appropriate domain subdirectory (e.g., `examples/configs/recipes/llm/` or `examples/configs/recipes/vlm/`).
- Name it following our recipe naming rules (see below). The YAML filename should mirror the driver script name but with `.yaml`.

2) Driver script under `tests/test_suites/`
- Create a shell script in the matching domain (e.g., `tests/test_suites/llm/` or `tests/test_suites/vlm/`).
- The script should source any common environment (e.g., `common.env`) and invoke the training entrypoint with `uv run ... --config <path-to-yaml>` as appropriate.
- Match the driver script filename to the YAML base name, with `.sh`.

3) Add to nightly list
- Append the driver script path (relative to `tests/test_suites/`) to `tests/test_suites/nightly.txt`.

### Recipe Naming Rules (YAML and Driver Scripts)

Base pattern (LLM):

```
<algo>-<model>-<nodes>n<gpus>g-<strategy-and-params>[-modifiers][-long][.vN].(yaml|sh)
```

- **algo**: task or algorithm, e.g., `sft`, `dpo`, `grpo`.
- **model**: model identifier, e.g., `llama3.1-8b-instruct`, `qwen2.5-7b-instruct`.
- **nodes/gpus**: cluster allocation, e.g., `1n8g`, `4n8g`, `8n8g`.
- **strategy-and-params**: parallelism or framework detail, e.g., `fsdp2tp1`, `tp4pp2`, `megatron`, `dtensor2tp1`.
- **modifiers** (optional): short flags like `sp` (sequence packing), `actckpt` (activation checkpointing), `fp8`, `noncolocated`, `quick`.
- **-long** (optional): indicates long-running recipe.
- **.vN** (optional): version suffix (e.g., `.v2`, `.v3`) reserved for convergence-impacting changes.

Examples (from current tree):

```
sft-llama3.1-8b-1n8g-fsdp2tp1-long.yaml
dpo-llama3.1-8b-instruct-4n8g-fsdp2tp4.yaml
grpo-llama3.1-8b-instruct-1n8g-megatron-fp8.yaml
grpo-qwen2.5-7b-instruct-4n8g-fsdp2tp4sp.v3.yaml
```

VLM pattern:

```
vlm_<algo>-<model>-<nodes>n<gpus>g-<strategy>[-modifiers][.vN].(yaml|sh)
```

Examples:

```
vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n2g-dtensor2tp1.v1.yaml
vlm_grpo-smolvlm2-2.2b-instruct-clevr-1n2g-dtensor2tp1.v1.sh
```

Known exceptions currently present:
- Deepscaler recipes encode context length in place of the cluster tuple, e.g., `grpo-deepscaler-1.5b-8K.(yaml|sh)`. These are allowed but should document the intended hardware in the script body.
- Some recipes include additional short flags in the strategy token (e.g., `fsdp2tp8sp`). Treat these as modifiers appended to the strategy.

Directory placement:

```
examples/configs/recipes/
  llm/
    <name>.yaml
  vlm/
    <name>.yaml

tests/test_suites/
  llm/
    common.env
    <name>.sh
  vlm/
    common.env
    <name>.sh
  nightly.txt
```

## NVIDIA Copyright

1. In NeMo-RL, add the following NVIDIA copyright header to all Python files and shell scripts and this header should include the current year. Exclude tests (e.g., files under `tests/` or test-only scripts). The header should appear at the top of the file.
```py
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

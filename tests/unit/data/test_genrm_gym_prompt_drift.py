# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""The Gym DWRL judge servers (NeMo Gym's resources_servers/dwrl_genrm_compare_pointwise and _pairwise) build the judge
prompt from their own copies of the prompt template and default principle (server config) and of
`flatten_to_single_turn` (app.py), because Gym cannot import NeMo-RL. These tests fail when a copy drifts from what DWRL
training uses (`nemo_rl.data.genrm_prompts`), so that a judge is never served a prompt it was not trained on.
Likewise for the vanilla GenRM, whose prompt Gym's genrm_model builds with examples/nemotron_super_v3.5_genrm_template.jinja.

Gym is found through NEMO_GYM_DIR (a Gym checkout) or else the installed `nemo_gym` package (3rdparty/Gym-workspace)."""

import ast
import importlib.util
import os
import pathlib
import re
import types

import jinja2.ext
import pytest
import yaml
from jinja2.sandbox import ImmutableSandboxedEnvironment

from nemo_rl.data.genrm_prompts import (
    DWRL_AND_GENRM_PAIRWISE_RUBRIC_TEMPLATE,
    DWRL_POINTWISE_RUBRIC_TEMPLATE,
    GENRM_PAIRWISE_DEFAULT_PRINCIPLE,
    GENRM_POINTWISE_DEFAULT_PRINCIPLE,
    flatten_to_single_turn,
)


def _gym_dir():
    if os.environ.get("NEMO_GYM_DIR"):
        return pathlib.Path(os.environ["NEMO_GYM_DIR"])
    spec = importlib.util.find_spec("nemo_gym")
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).parents[1]  # <gym>/nemo_gym/__init__.py


GYM_DIR = _gym_dir()
SERVERS = {
    "dwrl_genrm_compare_pointwise": (
        DWRL_POINTWISE_RUBRIC_TEMPLATE,
        GENRM_POINTWISE_DEFAULT_PRINCIPLE,
    ),
    "dwrl_genrm_compare_pairwise": (
        DWRL_AND_GENRM_PAIRWISE_RUBRIC_TEMPLATE,
        GENRM_PAIRWISE_DEFAULT_PRINCIPLE,
    ),
}

requires_gym = pytest.mark.skipif(
    GYM_DIR is None
    or not all((GYM_DIR / "resources_servers" / name).is_dir() for name in SERVERS),
    reason=f"the Gym DWRL servers were not found (Gym: {GYM_DIR}); set NEMO_GYM_DIR to a Gym checkout",
)
GENRM_CHAT_TEMPLATE = (
    pathlib.Path(__file__).parents[3]
    / "examples"
    / "nemotron_super_v3.5_genrm_template.jinja"
)

# Text-only conversations (the Gym servers do not handle media yet), covering each role, the "User:" style prefixes
# that flatten_to_single_turn strips, reasoning in assistant turns, and surrounding whitespace.
CONVERSATIONS = [
    [("user", "What is 2+2?")],
    [("system", " Be nice "), ("user", "Hi"), ("assistant", "Hello!"), ("user", "Bye")],
    [
        ("system", "System: You are a tutor."),
        ("user", "User: explain {braces} please"),
        ("user", "  and more  "),
    ],
    [
        ("user", "Hi"),
        ("assistant", "<think>secret plan</think>Hello!"),
        ("assistant", "unterminated opener</think>\n\n Hi again "),
        ("user", "What does <think> mean?"),
    ],
    [
        ("user", "q"),
        ("assistant", "Assistant: prefixed answer"),
        ("user", "ASSISTANT: not a real prefix? \n"),
    ],
    [
        ("system", "line one\nline two"),
        ("user", "Ünïcödé ✓\n\nwith blank lines"),
        ("assistant", ""),
    ],
]


def _server_config(name):
    config = yaml.safe_load(
        (GYM_DIR / "resources_servers" / name / "configs" / f"{name}.yaml").read_text(
            encoding="utf-8"
        )
    )
    for block in config.values():
        if isinstance(block, dict) and name in (block.get("resources_servers") or {}):
            return block["resources_servers"][name]
    raise AssertionError(f"no resources_servers.{name} block in the {name} config")


def _gym_flatten_to_single_turn(name):
    """Gym's flatten_to_single_turn, executed from its source (the Gym server app itself needs Gym's dependencies)."""
    path = GYM_DIR / "resources_servers" / name / "app.py"
    node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "flatten_to_single_turn"
    )
    namespace = {"re": re}
    exec(compile(ast.Module([node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["flatten_to_single_turn"]


@requires_gym
@pytest.mark.parametrize("name", SERVERS)
def test_gym_prompt_template_is_the_training_template(name):
    template, _ = SERVERS[name]
    assert (
        _server_config(name)["dwrl_prompt"].strip() == template
    )  # the server formats dwrl_prompt.strip()


@requires_gym
@pytest.mark.parametrize("name", SERVERS)
def test_gym_default_principle_is_the_training_default(name):
    _, principle = SERVERS[name]
    assert (
        _server_config(name)["default_principle"].strip() == principle
    )  # the server uses default_principle.strip()


@requires_gym
@pytest.mark.parametrize("name", SERVERS)
@pytest.mark.parametrize("conversation", CONVERSATIONS)
def test_gym_conversation_context_is_the_training_context(name, conversation):
    gym_flatten = _gym_flatten_to_single_turn(name)
    gym_messages = [
        types.SimpleNamespace(role=role, content=content)
        for role, content in conversation
    ]
    training_context, _ = flatten_to_single_turn(
        [{"role": role, "content": content} for role, content in conversation]
    )
    assert gym_flatten(gym_messages) == training_context


@pytest.mark.parametrize("conversation", CONVERSATIONS)
@pytest.mark.parametrize(
    "principle",
    [None, "Evaluate both responses using ONLY these criteria:\n1. Be accurate."],
)
def test_vanilla_genrm_chat_template_renders_the_training_prompt(
    conversation, principle
):
    """Gym's genrm_model sends the conversation, then `principle` (when use_principle is on), `response_1` and
    `response_2` messages; the template must render the prompt run_genrm_pairwise_vlm.py trains on, in Super 3.5's
    chat format."""
    history = [{"role": role, "content": content} for role, content in conversation]
    response_1, response_2 = " First response \n", '{"json": true}'
    messages = (
        history
        + ([{"role": "principle", "content": principle}] if principle else [])
        + [
            {"role": "response_1", "content": response_1},
            {"role": "response_2", "content": response_2},
        ]
    )
    # the Jinja environment transformers' apply_chat_template (and so vLLM) renders chat templates with
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    rendered = environment.from_string(
        GENRM_CHAT_TEMPLATE.read_text(encoding="utf-8")
    ).render(messages=messages, add_generation_prompt=True)

    prompt = DWRL_AND_GENRM_PAIRWISE_RUBRIC_TEMPLATE.format(
        context=flatten_to_single_turn(history)[0],
        response_1=response_1,
        response_2=response_2,
        principle=principle or GENRM_PAIRWISE_DEFAULT_PRINCIPLE,
    )
    assert rendered == (
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )


def test_vanilla_genrm_gym_config_sends_the_training_principle():
    """With use_principle on, Gym's genrm_compare sends each row's principle, or default_principle (unstripped) when a
    row has none; its own default is not the training one, so the run config sets it."""
    config_path = (
        pathlib.Path(__file__).parents[3]
        / "examples"
        / "configs"
        / "grpo_super_v3.5_dwrl_pairwise_genrm.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    genrm_compare = config["env"]["nemo_gym"]["genrm_compare_resources_server"][
        "resources_servers"
    ]["genrm_compare"]
    assert genrm_compare["use_principle"] is True
    assert genrm_compare["default_principle"] == GENRM_PAIRWISE_DEFAULT_PRINCIPLE

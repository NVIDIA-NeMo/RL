# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the production key-mapping wrapper across Automodel API generations."""

import ast
import os
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def install_wrapper(monkeypatch):
    monkeypatch.delenv("NRL_DISABLE_QWENVL_VISION_PATCH", raising=False)
    source = (
        Path(__file__).resolve().parents[2] / "nemo_rl/models/automodel/checkpoint.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_qwen_vl_vision_key_mapping"
    )
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    modules = {}
    for name in (
        "nemo_automodel",
        "nemo_automodel.components",
        "nemo_automodel.components.checkpoint",
        "nemo_automodel.components.checkpoint.checkpointing",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    checkpoint = modules["nemo_automodel.components.checkpoint.checkpointing"]

    def install(upstream):
        checkpoint.get_combined_key_mapping = upstream
        namespace[function.name]()
        return checkpoint.get_combined_key_mapping

    return install, namespace[function.name], checkpoint


def test_model_keyword_is_forwarded_without_changing_mapping(install_wrapper):
    install, _, _ = install_wrapper
    calls = []
    expected = {r"^checkpoint\.": "native."}

    def upstream(model_type, model_key_mapping=None, *, model=None):
        calls.append((model_type, model_key_mapping, model))
        return expected

    model, explicit = object(), {r"^custom\.": "special."}
    wrapped = install(upstream)
    assert wrapped("flint", explicit, model=model) is expected
    assert calls == [("flint", explicit, model)]


@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "qwen2_vl"])
def test_scoped_qwen_callable_mapping_is_preserved(install_wrapper, model_type):
    install, _, _ = install_wrapper
    model = object()

    def scoped_mapping(name):
        if name.startswith("visual."):
            return name.replace("visual.", "model.visual.", 1)
        if name.startswith("model.layers."):
            return name.replace("model.layers.", "model.language_model.layers.", 1)
        return name

    def upstream(actual_type, model_key_mapping=None, *, model=None):
        assert actual_type == model_type
        assert model is not None
        return scoped_mapping

    mapping = install(upstream)(model_type, model=model)
    assert mapping is scoped_mapping
    assert (
        mapping("visual.patch_embed.proj.weight")
        == "model.visual.patch_embed.proj.weight"
    )
    assert mapping("model.layers.0.weight") == "model.language_model.layers.0.weight"
    assert (
        mapping("model.language_model.layers.0.weight")
        == "model.language_model.layers.0.weight"
    )


@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "qwen2_vl"])
def test_legacy_two_arg_api_retains_missing_qwen_renames(install_wrapper, model_type):
    install, apply_patch, checkpoint = install_wrapper
    original = {r"^lm_head\.": "lm_head."}

    def legacy(model_type, model_key_mapping=None):
        return original

    wrapped = install(legacy)
    mapping = wrapped(model_type)
    assert mapping[r"^visual\."] == "model.visual."
    language_rule = r"^model(?!\.(language_model|visual))"
    assert (
        re.sub(language_rule, mapping[language_rule], "model.layers.0.weight")
        == "model.language_model.layers.0.weight"
    )
    assert original == {r"^lm_head\.": "lm_head."}
    apply_patch()
    assert checkpoint.get_combined_key_mapping is wrapped
    assert wrapped._nrl_orig is legacy


def test_existing_qwen_dict_renames_are_not_overwritten(install_wrapper):
    install, _, _ = install_wrapper
    existing = {
        r"^visual\.": "model.visual.vision_model.",
        r"^model\.": "model.language_model.",
    }
    wrapped = install(lambda model_type, model_key_mapping=None: existing)
    assert wrapped("qwen2_5_vl") == existing

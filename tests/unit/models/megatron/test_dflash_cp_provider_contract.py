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

from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[4]


def _function(
    tree: ast.AST,
    *,
    class_name: str | None,
    function_name: str,
) -> ast.FunctionDef:
    nodes = tree.body if isinstance(tree, ast.Module) else []
    if class_name is not None:
        cls = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = cls.body
    return next(
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def test_provider_threads_cp_local_layout_and_groups() -> None:
    training = ast.parse(
        (_REPO_ROOT / "nemo_rl/models/megatron/draft/training.py").read_text()
    )
    train = ast.parse((_REPO_ROOT / "nemo_rl/models/megatron/train.py").read_text())

    protocol_forward = _function(
        training,
        class_name="DraftTrainingProvider",
        function_name="forward",
    )
    protocol_parameters = {
        argument.arg for argument in protocol_forward.args.kwonlyargs
    }
    assert {
        "input_ids_cp_local",
        "sequence_layout",
        "context_parallel_group",
        "tensor_parallel_group",
    } <= protocol_parameters

    forward_driver = _function(
        train,
        class_name=None,
        function_name="forward_with_post_processing_fn",
    )
    provider_calls = [
        node
        for node in ast.walk(forward_driver)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "forward"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "draft_provider"
    ]
    assert len(provider_calls) == 1
    keyword_values = {
        keyword.arg: keyword.value for keyword in provider_calls[0].keywords
    }
    assert ast.unparse(keyword_values["input_ids_cp_local"]) == "input_ids_cp_sharded"
    assert (
        ast.unparse(keyword_values["sequence_layout"])
        == "processed_mb.draft_sequence_layout"
    )
    assert ast.unparse(keyword_values["context_parallel_group"]) == (
        "get_context_parallel_group()"
    )
    assert ast.unparse(keyword_values["tensor_parallel_group"]) == (
        "get_tensor_model_parallel_group()"
    )

    normalization_counts = _function(
        training,
        class_name="DraftTrainingProvider",
        function_name="normalization_counts",
    )
    assert "sequence_layout" in {
        argument.arg for argument in normalization_counts.args.kwonlyargs
    }


def test_provider_capabilities_are_explicit_and_method_scoped() -> None:
    training = ast.parse(
        (_REPO_ROOT / "nemo_rl/models/megatron/draft/training.py").read_text()
    )
    capability_names = {
        "supports_context_parallel",
        "supports_sequence_packing",
        "supports_target_sequence_parallel",
        "requires_full_cp_local_capture",
    }

    protocol = next(
        node
        for node in training.body
        if isinstance(node, ast.ClassDef) and node.name == "DraftTrainingProvider"
    )
    protocol_capabilities = {
        target.id
        for node in protocol.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        for target in (node.target,)
    }
    assert capability_names <= protocol_capabilities

    for class_name, expected in (
        ("Eagle3Speculator", False),
        ("DFlashSpeculator", True),
    ):
        provider = next(
            node
            for node in training.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        assignments = {
            target.id: node.value.value
            for node in provider.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
            for target in (node.target,)
            if target.id in capability_names
        }
        assert assignments == dict.fromkeys(capability_names, expected)


def test_dflash_allows_target_sp_but_keeps_draft_sp_disabled() -> None:
    training = ast.parse(
        (_REPO_ROOT / "nemo_rl/models/megatron/draft/training.py").read_text()
    )
    build_model = _function(
        training,
        class_name="DFlashSpeculator",
        function_name="build_model",
    )
    source = ast.unparse(build_model)

    assert "if model_provider.sequence_parallel" not in source
    sequence_parallel_keywords = [
        keyword
        for node in ast.walk(build_model)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "sequence_parallel"
    ]
    assert len(sequence_parallel_keywords) == 1
    assert isinstance(sequence_parallel_keywords[0].value, ast.Constant)
    assert sequence_parallel_keywords[0].value.value is False


def test_dflash_body_receives_cp_layout_and_group() -> None:
    training = ast.parse(
        (_REPO_ROOT / "nemo_rl/models/megatron/draft/training.py").read_text()
    )
    forward = _function(
        training,
        class_name="DFlashSpeculator",
        function_name="forward",
    )
    calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "draft_model"
    ]
    assert len(calls) == 1
    keyword_values = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.unparse(keyword_values["sequence_layout"]) == "sequence_layout"
    assert (
        ast.unparse(keyword_values["context_parallel_group"])
        == "context_parallel_group"
    )
    prepare_batch = _function(
        training,
        class_name="DFlashSpeculator",
        function_name="prepare_batch",
    )
    assert "max_cp_boundary_exclusion_fraction" in ast.unparse(prepare_batch)


def test_draft_loss_wraps_packed_and_unpacked_policy_losses() -> None:
    train = ast.parse((_REPO_ROOT / "nemo_rl/models/megatron/train.py").read_text())
    loss_call = _function(
        train,
        class_name="LossPostProcessor",
        function_name="__call__",
    )
    packing_branch = next(
        node
        for node in loss_call.body
        if isinstance(node, ast.If) and "pack_sequences" in ast.unparse(node.test)
    )
    draft_wrappers = [
        node
        for node in ast.walk(loss_call)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DraftLossWrapper"
    ]

    assert len(draft_wrappers) == 1
    assert draft_wrappers[0].lineno > packing_branch.end_lineno


def test_worker_uses_method_neutral_packed_cp_capability_guard() -> None:
    worker = ast.parse(
        (
            _REPO_ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
        ).read_text()
    )
    validator = _function(
        worker,
        class_name=None,
        function_name="_validate_draft_training_setup",
    )
    source = ast.unparse(validator)

    assert "supports_context_parallel" in source
    assert "supports_sequence_packing" in source
    assert "supports_target_sequence_parallel" in source
    assert "requires sequence_packing.enabled=true" in source
    assert "virtual_pipeline_model_parallel_size must be 1" in source
    assert "generation context_parallel_size must be 1" in source


def test_split_draft_counts_reduce_once_over_dp_cp() -> None:
    worker = ast.parse(
        (
            _REPO_ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
        ).read_text()
    )
    finish = _function(
        worker,
        class_name="MegatronPolicyWorkerImpl",
        function_name="_finish_train_step_body",
    )
    source = ast.unparse(finish)

    assert source.count("with_context_parallel=True") == 1
    assert "draft_counts" in source
    assert "all_reduce(draft_counts" in source

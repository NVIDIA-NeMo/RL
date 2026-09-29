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
import importlib
from pathlib import Path

import pytest

from nemo_rl.environments import gym_checkpoint
from nemo_rl.environments.gym_checkpoint import (
    GymAgentContinuationRoot,
    GymExternalStorageReference,
)


# NeMo-RL intentionally copies this small wire contract so that importing the
# core environment adapter does not require NeMo-Gym. Keep every copied value
# tied to either its Gym definition or an installed Gym route.
GYM_CONSTANT_SOURCES: dict[str, tuple[str, str]] = {
    "GYM_CHECKPOINT_SCHEMA_VERSION": (
        "nemo_gym._checkpoint.control",
        "CONTROL_SCHEMA_VERSION",
    ),
    "GYM_CHECKPOINT_CONTROL_PREFIX": (
        "nemo_gym._checkpoint.control",
        "CONTROL_URL_PREFIX",
    ),
    "GYM_CHECKPOINT_CAPABILITIES_PATH": ("route", "GET"),
    "GYM_MODEL_ADMISSION_PREFIX": (
        "nemo_gym._checkpoint.model_control_contracts",
        "MODEL_ADMISSION_URL_PREFIX",
    ),
    "GYM_MODEL_CHECKPOINT_PREFIX": (
        "nemo_gym._checkpoint.ledger",
        "MODEL_CHECKPOINT_URL_PREFIX",
    ),
    "GYM_AGENT_CHECKPOINT_PREFIX": (
        "nemo_gym._checkpoint.agent",
        "AGENT_CHECKPOINT_URL_PREFIX",
    ),
    "GYM_AGENT_COMPLETION_RECEIPT_PATH": ("route", "GET"),
    "GYM_AGENT_COMPLETION_ACK_PATH": ("route", "POST"),
    "GYM_AGENT_RETIRE_PATH": ("route", "POST"),
    "GYM_AGENT_DISCARD_RESTORED_CONTINUATION_PATH": ("route", "POST"),
    "GYM_RESOURCES_CHECKPOINT_PREFIX": (
        "nemo_gym._checkpoint.resources",
        "RESOURCES_CHECKPOINT_URL_PREFIX",
    ),
    "GYM_AGENT_CONTINUATION_INDEX_FEATURE": (
        "nemo_gym._checkpoint.artifacts",
        "AGENT_CONTINUATION_INDEX_FEATURE",
    ),
    "GYM_AGENT_DISCARD_RESTORED_CONTINUATION_FEATURE": (
        "nemo_gym._checkpoint.agent",
        "DISCARD_RESTORED_CONTINUATION_FEATURE",
    ),
    "GYM_AGENT_RESOURCE_DEPENDENCY_INDEX_FEATURE": (
        "nemo_gym._checkpoint.artifacts",
        "AGENT_RESOURCE_DEPENDENCY_INDEX_FEATURE",
    ),
    "GYM_AGENT_COMPLETED_RESULT_ACKNOWLEDGEMENT_FEATURE": (
        "nemo_gym._checkpoint.agent",
        "COMPLETED_RESULT_ACKNOWLEDGEMENT_FEATURE",
    ),
    "GYM_AGENT_INLINE_COMPLETION_RECEIPT_FEATURE": (
        "nemo_gym._checkpoint.agent",
        "COMPLETION_RECEIPT_IN_RUN_RESPONSE_FEATURE",
    ),
    "GYM_EXTERNAL_STORAGE_REFERENCE_INDEX_FEATURE": (
        "nemo_gym._checkpoint.artifacts",
        "EXTERNAL_STORAGE_REFERENCE_INDEX_FEATURE",
    ),
    "_IDENTITY_PATTERN": (
        "nemo_gym.rollout_correlation",
        "ROLLOUT_ID_PATTERN",
    ),
}


def _module_constants() -> set[str]:
    module_path = Path(gym_checkpoint.__file__ or "")
    assert module_path.is_file()
    module = ast.parse(module_path.read_text())
    return {
        node.targets[0].id
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and (
            node.targets[0].id.startswith("GYM_")
            or node.targets[0].id == "_IDENTITY_PATTERN"
        )
    }


def test_every_copied_gym_constant_declares_its_source() -> None:
    assert set(GYM_CONSTANT_SOURCES) == _module_constants()


@pytest.mark.nemo_gym
def test_copied_gym_constants_match_the_pinned_gym_contract() -> None:
    from fastapi import FastAPI
    from nemo_gym._checkpoint.agent import (
        AgentCheckpointParticipant,
        install_agent_checkpoint,
    )
    from nemo_gym._checkpoint.control import (
        ControlCapabilities,
        ControlFence,
        MultiProcessCapability,
        install_control_plane,
    )

    app = FastAPI()
    fence = ControlFence()
    install_control_plane(
        app,
        capabilities=ControlCapabilities(
            component="responses_api_agents",
            name="contract-test",
            multi_process=MultiProcessCapability(
                mode="single_worker",
                num_workers=1,
            ),
        ),
        fence=fence,
    )
    install_agent_checkpoint(
        app,
        participant=AgentCheckpointParticipant(),
        fence=fence,
        auth_token="test-token",
    )
    routes = {
        (method, route.path)
        for route in app.routes
        for method in (route.methods or set())
    }

    for name, (module_name, source_name) in GYM_CONSTANT_SOURCES.items():
        actual = getattr(gym_checkpoint, name)
        if module_name == "route":
            assert (source_name, actual) in routes
            continue
        expected = getattr(importlib.import_module(module_name), source_name)
        if hasattr(expected, "pattern"):
            expected = expected.pattern
        assert actual == expected


@pytest.mark.nemo_gym
def test_copied_checkpoint_artifact_models_round_trip_through_gym() -> None:
    from nemo_gym._checkpoint.artifacts import (
        AgentContinuationRoot,
        ExternalStorageReference,
    )

    continuation = GymAgentContinuationRoot(
        rollout_id="rollout-1",
        attempt_index=1,
        capture_key="rollout-1-a1",
        last_committed_model_call_id="call-1",
        resource_state_revisions={"sandbox": 2},
    )
    gym_continuation = AgentContinuationRoot.model_validate(
        continuation.model_dump(mode="json")
    )
    assert (
        GymAgentContinuationRoot.model_validate(
            gym_continuation.model_dump(mode="json")
        )
        == continuation
    )

    reference = GymExternalStorageReference(
        capture_key="rollout-1-a1",
        boundary_model_call_id="call-1",
        key="staging-key-1",
    )
    gym_reference = ExternalStorageReference.model_validate(
        reference.model_dump(mode="json")
    )
    assert (
        GymExternalStorageReference.model_validate(
            gym_reference.model_dump(mode="json")
        )
        == reference
    )

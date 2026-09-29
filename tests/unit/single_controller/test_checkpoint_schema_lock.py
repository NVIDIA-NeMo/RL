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

"""Lock durable checkpoint field names to their declared schema versions."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel

from nemo_rl.algorithms.single_controller_utils.rollout_checkpoint import (
    ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
    RolloutSnapshotManifest,
)
from nemo_rl.environments.gym_checkpoint import (
    GYM_CHECKPOINT_SCHEMA_VERSION,
    GymAgentContinuationRoot,
    GymCheckpointCommitResult,
    GymExternalStorageReference,
)
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    RolloutRecoverySidecarState,
)

_SCHEMA_LOCK_PATH = Path(__file__).with_name("checkpoint_schema_lock.json")


def _nested_pydantic_models(annotation: Any) -> set[type[BaseModel]]:
    nested: set[type[BaseModel]] = set()
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        nested.add(annotation)
    for argument in get_args(annotation):
        nested.update(_nested_pydantic_models(argument))
    return nested


def _pydantic_model_fields(*roots: type[BaseModel]) -> dict[str, list[str]]:
    """Return every Pydantic model reachable from durable root models."""
    pending = list(roots)
    models: dict[str, list[str]] = {}
    while pending:
        model = pending.pop()
        if model.__name__ in models:
            continue
        models[model.__name__] = list(model.model_fields)
        for field in model.model_fields.values():
            pending.extend(_nested_pydantic_models(field.annotation))
    return dict(sorted(models.items()))


def _current_checkpoint_schema() -> dict[str, Any]:
    return {
        "gym_checkpoint_commit_result": {
            "schema_version": GYM_CHECKPOINT_SCHEMA_VERSION,
            "models": _pydantic_model_fields(GymCheckpointCommitResult),
        },
        "gym_checkpoint_artifacts": {
            "schema_version": GYM_CHECKPOINT_SCHEMA_VERSION,
            "models": _pydantic_model_fields(
                GymAgentContinuationRoot,
                GymExternalStorageReference,
            ),
        },
        "rollout_recovery": {
            "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "models": _pydantic_model_fields(RolloutRecoverySidecarState),
        },
        "rollout_snapshot_manifest": {
            "schema_version": ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
            "fields": [
                field.name for field in dataclasses.fields(RolloutSnapshotManifest)
            ],
        },
    }


def test_checkpoint_schema_lock() -> None:
    expected = json.loads(_SCHEMA_LOCK_PATH.read_text())
    actual = _current_checkpoint_schema()
    assert actual == expected, (
        "A durable checkpoint schema changed without updating its lock file. "
        "Review compatibility, bump the owning schema version, then run "
        "`python tests/unit/single_controller/test_checkpoint_schema_lock.py "
        "--regen`."
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regen",
        action="store_true",
        help="rewrite the checked-in checkpoint schema lock",
    )
    args = parser.parse_args()
    if not args.regen:
        parser.error("pass --regen to rewrite the checkpoint schema lock")
    current = _current_checkpoint_schema()
    if _SCHEMA_LOCK_PATH.is_file():
        previous = json.loads(_SCHEMA_LOCK_PATH.read_text())
        for owner, current_contract in current.items():
            previous_contract = previous.get(owner)
            if (
                previous_contract is not None
                and previous_contract.get("schema_version")
                == current_contract["schema_version"]
                and previous_contract != current_contract
            ):
                parser.error(
                    f"{owner} fields changed without a schema-version bump; "
                    "bump the owning version before regenerating"
                )
    _SCHEMA_LOCK_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    _main()

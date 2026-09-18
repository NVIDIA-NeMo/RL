#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify reused game services against current source without installing anything.

This checks real reset/render/close for all 30 Gym-V train/held-out games,
reset/render/step for all 12 VisGym games, and the real mixed manifest contract.
It is not model-rollout, policy-update, or end-to-end training qualification.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from tools.check_visual_image_tools_mix import (
    GYM_V_TRAIN,
    GYM_V_VALIDATION,
    VISGYM_TRAIN,
    audit_mix,
    read_rows,
)


def observation_image_count(observations: list) -> int:
    """Count image parts in both raw-dict and parsed Responses observations."""
    return sum(
        (part.get("type") if isinstance(part, dict) else part.type) == "input_image"
        for message in observations
        if isinstance(message.content, list)
        for part in message.content
    )


def check_source() -> dict:
    # These SDKs live in the service interpreter being checked, not the driver.
    import nemo_gym
    import openai

    root = Path(__file__).resolve().parents[1]
    assert (
        Path(nemo_gym.__file__)
        .resolve()
        .is_relative_to(root / "3rdparty/Gym-workspace/Gym")
    ), nemo_gym.__file__
    assert openai.__version__ == "2.6.1", openai.__version__
    return {
        "python": sys.executable,
        "gym_source": nemo_gym.__file__,
        "openai": openai.__version__,
    }


async def gym_v_reset(row: dict) -> dict:
    # Only this service interpreter contains Gym-V's newer Gymnasium dependency.
    import gymnasium
    from fastapi import Request
    from omegaconf import OmegaConf
    from nemo_gym.config_types import BaseServerConfig
    from nemo_gym.server_utils import ServerClient
    from resources_servers.gym_v.app import GymVResourcesServer
    from resources_servers.gym_v.schemas import (
        GymVAgentVerifyRequest,
        GymVCloseRequest,
        GymVNeMoGymResponse,
        GymVResourcesServerConfig,
        GymVSeedSessionRequest,
        GymVTaskRow,
    )

    assert gymnasium.__version__ != "1.1.1", "Gym-V accidentally uses the VisGym fork"
    server = GymVResourcesServer(
        config=GymVResourcesServerConfig(
            name="gym_v_probe",
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            disable_text_feedback=True,
            max_image_wh=512,
        ),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=0),
            global_config_dict=OmegaConf.create({}),
        ),
    )
    request = Request({"type": "http", "headers": []})
    task = GymVTaskRow.model_validate(row)
    seeded = await server.seed_session(request, GymVSeedSessionRequest(task_row=task))
    try:
        images = observation_image_count(seeded.obs)
        assert images > 0, "No rendered image in actual seed-session observation"
    finally:
        await server.close(request, GymVCloseRequest(env_id=seeded.env_id))
    verified = await server.verify(
        request,
        GymVAgentVerifyRequest(
            responses_create_params=task.responses_create_params,
            response=GymVNeMoGymResponse(
                id="probe",
                created_at=0.0,
                model="no-model-called",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
                env_id=seeded.env_id,
            ),
        ),
    )
    assert math.isfinite(verified.reward)
    assert not server.env_id_to_env and not server.env_id_to_operation
    return {
        "env_id": row["env_id"],
        "status": "PASS",
        "scope": "reset-render-close-verify",
        "images": images,
        "gymnasium": gymnasium.__version__,
    }


def run_one(env_id: str) -> None:
    provenance = check_source()
    if env_id in VISGYM_TRAIN:
        # VisGym runs in its isolated fork interpreter and a fresh native process.
        from tools import visgym_multienv_smoke
        import gymnasium

        assert gymnasium.__version__ == "1.1.1", gymnasium.__version__
        sys.argv = [sys.argv[0], env_id, "--result", os.environ["COMPONENT_RESULT"]]
        visgym_multienv_smoke.main()
    else:
        row = next(
            row
            for row in read_rows(
                [
                    Path(os.environ["GAMES_TRAIN_MANIFEST"]),
                    Path(os.environ["GAMES_EVAL_MANIFEST"]),
                ]
            )
            if row["env_id"] == env_id
        )
        record = asyncio.run(gym_v_reset(row))
        Path(os.environ["COMPONENT_RESULT"]).write_text(
            json.dumps({**record, **provenance}) + "\n"
        )
    print(json.dumps(provenance), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-env")
    args = parser.parse_args()
    if args.one_env:
        run_one(args.one_env)
        return
    output = Path(os.environ["RUN_DIR"])
    venv = Path(os.environ["GAMES_VENV_ROOT"])
    certificate = venv / ".visual-games-suite-prefetch-complete"
    before = hashlib.sha256(certificate.read_bytes()).hexdigest()
    audit = audit_mix(
        train=read_rows(
            [
                Path(os.environ["GAMES_TRAIN_MANIFEST"]),
                Path(os.environ["IMAGE_TRAIN_MANIFEST"]),
            ]
        ),
        validation=read_rows([Path(os.environ["GAMES_EVAL_MANIFEST"])]),
        max_output_tokens=512,
    )
    (output / "manifest-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    environments = sorted(GYM_V_TRAIN | GYM_V_VALIDATION | VISGYM_TRAIN)

    def probe(env_id: str) -> dict:
        component = "visgym" if env_id in VISGYM_TRAIN else "gym_v"
        interpreter = venv / "resources_servers" / component / ".venv/bin/python"
        name = env_id.replace("/", "__")
        result_path = output / f"{name}.json"
        with (output / f"{name}.log").open("w") as log:
            try:
                run = subprocess.run(
                    [
                        str(interpreter),
                        "-m",
                        "tools.check_visual_image_tools_components",
                        "--one-env",
                        env_id,
                    ],
                    env={**os.environ, "COMPONENT_RESULT": str(result_path)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=240,
                    check=False,
                )
                passed = run.returncode == 0 and result_path.exists()
                reason = str(run.returncode)
            except subprocess.TimeoutExpired:
                passed, reason = False, "timeout"
        record = {
            "env_id": env_id,
            "passed": passed,
            "exit": reason,
            "log": f"{name}.log",
        }
        print(json.dumps(record), flush=True)
        return record

    # Two native processes at a time; every game is tried even if another fails.
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(probe, environments))
    unchanged = hashlib.sha256(certificate.read_bytes()).hexdigest() == before
    report = {
        "scope": "components-only-no-model",
        "results": results,
        "certificate_sha256": before,
        "certificate_unchanged": unchanged,
        "runtime_verified": False,
    }
    (output / "component-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if not unchanged or not all(row["passed"] for row in results):
        raise SystemExit("Component qualification failed; inspect per-game logs")
    print(
        "ALL_42_GAME_COMPONENTS_PASSED; model rollout qualification still required",
        flush=True,
    )


if __name__ == "__main__":
    main()

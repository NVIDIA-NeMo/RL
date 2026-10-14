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

"""Exercise Harbor/OpenCode credential propagation over real Docker HTTP.

Requires the optional requirements-harbor.txt environment and an image with
OpenCode preinstalled. Uses a deterministic fake model, never a paid provider.
Only installation is replaced; Harbor's job, trial, agent run, and verifier run
normally. This is a correctness smoke test, not a latency benchmark.
"""

import argparse
import asyncio
import json
import re
import secrets
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from aiohttp import ClientSession, web

# Optional isolated dependency; still type-check against the real installed API
# with --python-interpreter-path /tmp/nrl-harbor-key-venv/bin/python.
from harbor.agents.installed.opencode import OpenCode  # pyrefly: ignore[missing-import]
from harbor.environments.base import BaseEnvironment  # pyrefly: ignore[missing-import]
from harbor.job import Job  # pyrefly: ignore[missing-import]
from harbor.models.job.config import JobConfig, RetryConfig  # pyrefly: ignore[missing-import]
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig  # pyrefly: ignore[missing-import]

Mode = Literal["provider-env", "config-env", "config-literal"]
MODES: tuple[Mode, ...] = ("provider-env", "config-env", "config-literal")


class PreinstalledOpenCode(OpenCode):
    """Reuse a local image without modifying Harbor's inference/run code."""

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_agent(environment, command="opencode --version")


def agent_config(*, mode: Mode, key: str, base_url: str) -> AgentConfig:
    """Build the three supported launch configurations with explicit routing."""
    provider = "nrl"
    env = {
        "OPENCODE_DISABLE_AUTOUPDATE": "true",
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
    }
    options = {"baseURL": base_url}
    if mode == "provider-env":
        env["OPENAI_API_KEY"] = key
    elif mode == "config-env":
        env["NRL_ROLLOUT_API_KEY"] = key
        options["apiKey"] = "{env:NRL_ROLLOUT_API_KEY}"
    elif mode == "config-literal":
        options["apiKey"] = key
    else:
        raise ValueError(f"Unknown credential mode: {mode}")
    return AgentConfig(
        import_path="tools.http_profile.harbor_key_smoke:PreinstalledOpenCode",
        model_name=f"{provider}/smoke-model",
        env=env,
        override_timeout_sec=180,
        kwargs={
            "opencode_config": {
                "enabled_providers": [provider],
                "small_model": f"{provider}/smoke-model",
                "agent": {"title": {"disable": True}},
                "provider": {
                    provider: {
                        "npm": "@ai-sdk/openai-compatible",
                        "env": ["OPENAI_API_KEY"] if mode == "provider-env" else [],
                        "options": options,
                        "models": {
                            "smoke-model": {
                                "name": "smoke-model",
                                "limit": {"context": 32768, "output": 1024},
                            }
                        },
                    }
                },
            }
        },
    )


@dataclass(frozen=True)
class Observation:
    rollout: str
    marker_matches: bool
    tool_result: bool
    stream: bool
    custom_identity_fields: list[str]
    custom_identity_headers: list[str]


class FakeModel:
    """Attribute by Authorization before parsing the body for test assertions."""

    def __init__(self) -> None:
        self.bindings: dict[str, str] = {}
        self.observations: list[Observation] = []
        self.rejections = 0
        self.retry_injections: set[str] = set()
        self.first_requests: dict[str, set[str]] = {}
        self.pair_ready: dict[str, asyncio.Event] = {}

    async def complete(self, request: web.Request) -> web.StreamResponse:
        rollout = self.bindings.get(request.headers.get("Authorization", ""))
        if rollout is None:
            self.rejections += 1
            return web.json_response({"error": "unknown rollout key"}, status=401)
        body = await request.json()
        messages = body["messages"]
        marker = f"SMOKE_{rollout}"
        marker_matches = marker in json.dumps(messages)
        tool_result = any(
            message["role"] == "tool" and marker in json.dumps(message)
            for message in messages
        )
        identity_names = {"conv_id", "conversation_id", "rollout_id", "ng_capture"}
        self.observations.append(
            Observation(
                rollout,
                marker_matches,
                tool_result,
                body.get("stream", False),
                sorted(identity_names.intersection(body)),
                sorted(
                    name
                    for name in request.headers
                    if re.search("conv.*id|rollout.*id|ng.capture", name, re.I)
                ),
            )
        )
        if not marker_matches:
            return web.json_response({"error": "wrong rollout key"}, status=409)
        if rollout not in self.retry_injections:
            self.retry_injections.add(rollout)
            group = rollout.rsplit("_", 1)[0]
            arrived = self.first_requests.setdefault(group, set())
            ready = self.pair_ready.setdefault(group, asyncio.Event())
            arrived.add(rollout)
            if len(arrived) == 2:
                ready.set()
            # Require real HTTP overlap, then make the real SDK retry with its key.
            await asyncio.wait_for(ready.wait(), timeout=45)
            return web.json_response(
                {
                    "error": {
                        "message": "intentional smoke retry",
                        "type": "server_error",
                    }
                },
                status=503,
                headers={"Retry-After": "0"},
            )
        tools = body.get("tools", [])
        has_bash = any(tool.get("function", {}).get("name") == "bash" for tool in tools)
        if has_bash and not tool_result:
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call_{rollout}",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps(
                                {
                                    "command": f"printf '{marker}\\n' | tee /app/nrl-key-smoke.txt",
                                    "description": "Write the smoke-test marker",
                                }
                            ),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            delta = {"role": "assistant", "content": f"Done: {marker}"}
            finish = "stop"
        envelope = {
            "id": f"chatcmpl-{secrets.token_hex(8)}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "smoke-model",
        }
        usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        if body.get("stream"):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for chunk_delta, reason in ((delta, None), ({}, finish)):
                payload: dict[str, object] = {
                    **envelope,
                    "choices": [
                        {"index": 0, "delta": chunk_delta, "finish_reason": reason}
                    ],
                }
                if reason:
                    payload["usage"] = usage
                await response.write(f"data: {json.dumps(payload)}\n\n".encode())
                await asyncio.sleep(0.01)
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        return web.json_response(
            {
                **envelope,
                "object": "chat.completion",
                "choices": [{"index": 0, "message": delta, "finish_reason": finish}],
                "usage": usage,
            }
        )


def write_task(path: Path, *, image: str, rollout: str) -> None:
    """Create a tiny real Harbor task with an observable tool/verifier result."""
    path.mkdir(parents=True)
    (path / "tests").mkdir()
    (path / "instruction.md").write_text(
        f"Write the marker SMOKE_{rollout} to /app/nrl-key-smoke.txt using bash, then finish.\n"
    )
    (path / "task.toml").write_text(
        f'version = "1.0"\n[environment]\ndocker_image = {json.dumps(image)}\n'
        "cpus = 1\nmemory_mb = 2048\n[agent]\ntimeout_sec = 180\n"
        "[verifier]\ntimeout_sec = 30\n"
    )
    (path / "tests" / "test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\n"
        f"if grep -Fxq SMOKE_{rollout} /app/nrl-key-smoke.txt; then\n"
        "  echo 1 > /logs/verifier/reward.txt\nelse\n"
        "  echo 0 > /logs/verifier/reward.txt\nfi\n"
    )


async def smoke(args: argparse.Namespace) -> None:
    """Run paired concurrent Harbor jobs and persist only assertion evidence."""
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    model = FakeModel()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", model.complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 0)
    await site.start()
    port = runner.addresses[0][1]
    base_url = f"http://host.docker.internal:{port}/v1"
    print(f"Fake model listening on port {port}", flush=True)
    try:
        async with ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={"Authorization": "Bearer unregistered"},
                data=b"deliberately not JSON",
            ) as response:
                assert response.status == 401  # reject before deserializing
        for mode in args.modes:
            pair = []
            for index in range(2):
                rollout = f"{mode.replace('-', '_')}_{index}"
                key = f"smoke-only-{secrets.token_hex(16)}"
                model.bindings[f"Bearer {key}"] = rollout
                task = root / "tasks" / rollout
                write_task(task, image=args.image, rollout=rollout)
                config = JobConfig(
                    job_name=rollout,
                    jobs_dir=root / "jobs",
                    tasks=[TaskConfig(path=task)],
                    agents=[agent_config(mode=mode, key=key, base_url=base_url)],
                    environment=EnvironmentConfig(type="docker"),
                    n_attempts=1,
                    n_concurrent_trials=1,
                    retry=RetryConfig(max_retries=0),
                    quiet=True,
                )
                # This is Gym's JSON-compatible Ray handoff, with Harbor's
                # explicit internal-transport setting. Persisted logs still redact.
                wire = config.model_dump(
                    mode="json", context={"redact_sensitive_env": False}
                )
                received = JobConfig.model_validate(json.loads(json.dumps(wire)))
                assert received.agents[0].env == config.agents[0].env
                pair.append(await Job.create(received))
            print(f"Running concurrent jobs: {mode}", flush=True)
            await asyncio.gather(*(job.run() for job in pair))
        observations = [asdict(item) for item in model.observations]
        counts = Counter(item.rollout for item in model.observations)
        failures = []
        for rollout in model.bindings.values():
            seen = [item for item in model.observations if item.rollout == rollout]
            if len(seen) < 3 or not any(item.tool_result for item in seen):
                failures.append(f"{rollout}: missing tool round trip")
        if not all(
            item.marker_matches
            and not item.custom_identity_fields
            and not item.custom_identity_headers
            for item in model.observations
        ):
            failures.append("credential leakage or custom identity metadata")
        trial_results = []
        for path in sorted((root / "jobs").glob("*/*/result.json")):
            result = json.loads(path.read_text())
            trial_results.append(result)
            verifier = result["verifier_result"]
            if (
                result["exception_info"]
                or not verifier
                or verifier["rewards"]["reward"] != 1
            ):
                failures.append(f"failed trial: {path}")
        if len(trial_results) != 2 * len(args.modes):
            failures.append("missing trial results")
        summary = {
            "harbor_version": version("harbor"),
            "image": args.image,
            "modes": args.modes,
            "requests_per_rollout": dict(counts),
            "observations": observations,
            "unknown_key_rejections": model.rejections,
            "retry_injections": len(model.retry_injections),
            "concurrent_pairs": sum(
                event.is_set() for event in model.pair_ready.values()
            ),
            "trials": len(trial_results),
            "failures": failures,
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        assert not failures, failures
    finally:
        await runner.cleanup()


def main() -> None:
    """CLI entrypoint; all Docker images and output paths are explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    asyncio.run(smoke(parser.parse_args()))


if __name__ == "__main__":
    main()

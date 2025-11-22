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

import asyncio
import re
import sys
import types

import pytest

# Stub out external modules that are not available in the test environment
google_module = sys.modules.get("google", types.ModuleType("google"))
google_module.adk = types.ModuleType("google.adk")
google_module.adk.agents = types.ModuleType("google.adk.agents")
google_module.adk.agents.Agent = object
google_module.adk.Agent = object
google_module.adk.runners = types.ModuleType("google.adk.runners")
google_module.adk.runners.Runner = object
google_module.adk.sessions = types.ModuleType("google.adk.sessions")
google_module.adk.sessions.InMemorySessionService = object
google_module.genai = types.ModuleType("google.genai")


class _Part:
    def __init__(self, text: str):
        self.text = text

    @classmethod
    def from_text(cls, text: str):
        return cls(text)


class _Content:
    def __init__(self, role: str | None = None, parts: list[_Part] | None = None):
        self.role = role
        self.parts = parts or []


google_module.genai.types = types.SimpleNamespace(
    GenerateContentConfig=object,
    SafetySetting=object,
    HarmCategory=object,
    HarmBlockThreshold=object,
    Content=_Content,
    Part=_Part,
)
google_module.genai.errors = types.SimpleNamespace(ServerError=Exception)
sys.modules.setdefault("google.adk", google_module.adk)
sys.modules.setdefault("google.adk.agents", google_module.adk.agents)
sys.modules.setdefault("google.adk.runners", google_module.adk.runners)
sys.modules.setdefault("google.adk.sessions", google_module.adk.sessions)
sys.modules.setdefault("google.genai", google_module.genai)
sys.modules.setdefault("google.genai.types", google_module.genai.types)
sys.modules.setdefault("google.genai.errors", google_module.genai.errors)

ray_module = types.ModuleType("ray")
ray_module.remote = lambda cls=None, **_: cls
sys.modules.setdefault("ray", ray_module)

torch_module = types.ModuleType("torch")
torch_module.tensor = lambda data, **_: data
torch_module.float32 = "float32"
torch_module.bool = "bool"
torch_module.Tensor = object
torch_module.distributed = types.SimpleNamespace(ProcessGroup=object)
torch_module.device = object
sys.modules.setdefault("torch", torch_module)

transformers_module = types.ModuleType("transformers")
transformers_module.PreTrainedTokenizerBase = object
sys.modules.setdefault("transformers", transformers_module)

from nemo_rl.environments.simulated_user import adk_utils, unique_numbers


# Dummy runner object used for mocking run_prompt_async behaviour
class DummyRunner:
    def __init__(self, numbers):
        self.numbers = numbers


def _make_metadata(numbers):
    return {
        "numbers": numbers,
        "unique_count": len(set(numbers)),
        "turn": 0,
        "max_turns": 5,
        "simulated_user_runner": DummyRunner(numbers),
        "grader_runner": DummyRunner(numbers),
    }


@pytest.fixture
def patch_unique_numbers(monkeypatch):
    async def fake_run_prompt_async(runner, user_id, msg, silence=True):
        match = re.search(r"what is number (\d+)", msg, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(runner.numbers):
                return str(runner.numbers[idx])
        return "<invalid>"

    monkeypatch.setattr(adk_utils, "run_prompt_async", fake_run_prompt_async)
    monkeypatch.setattr(unique_numbers, "run_prompt_async", fake_run_prompt_async)
    monkeypatch.setattr(
        unique_numbers._UniqueNumbersRunner,
        "_maybe_dump_adk_messages_to_file",
        lambda *a, **k: None,
    )


@pytest.fixture
def patch_adk_utils(monkeypatch):
    class FakeAgent:
        def __init__(self, instruction: str | None = None, **_):
            self.instruction = instruction

    class FakeSession:
        def __init__(self, user_id: str, session_id: str = "s1"):
            self.user_id = user_id
            self.id = session_id
            self.events = []

    class FakeSessionService:
        def __init__(self):
            self.sessions = {}

        async def create_session(self, app_name: str, user_id: str):
            sess = FakeSession(user_id)
            self.sessions.setdefault(app_name, {}).setdefault(user_id, {})[sess.id] = (
                sess
            )
            return sess

    class FakeRunner:
        def __init__(
            self, agent=None, app_name="app", session_service=None, responses=None
        ):
            self.agent = agent
            self.app_name = app_name
            self.session_service = session_service or FakeSessionService()
            self.responses = list(responses or [])

        async def run_async(self, user_id: str, session_id: str, new_message):
            session = self.session_service.sessions[self.app_name][user_id][session_id]
            session.events.append(
                types.SimpleNamespace(author="user", content=new_message)
            )
            text = self.responses.pop(0) if self.responses else "ack"
            event = types.SimpleNamespace(
                author="assistant",
                content=adk_utils.types.Content(
                    parts=[adk_utils.types.Part.from_text(text)]
                ),
            )
            session.events.append(event)
            yield event

    class FakeServerError(Exception):
        pass

    monkeypatch.setattr(adk_utils, "Agent", FakeAgent)
    monkeypatch.setattr(adk_utils, "Runner", FakeRunner)
    monkeypatch.setattr(adk_utils, "InMemorySessionService", FakeSessionService)
    monkeypatch.setattr(adk_utils, "ServerError", FakeServerError)
    return FakeRunner


def test_process_turn_query(patch_unique_numbers):
    runner = unique_numbers._UniqueNumbersRunner()
    numbers = [1, 2, 3]
    metadata = _make_metadata(numbers)

    msg_log = [{"role": "assistant", "content": "What is number 2?"}]
    obs, reward, terminated, _, next_meta = runner.process_turn(msg_log, metadata)

    assert obs == {"role": "user", "content": "2"}
    assert reward == unique_numbers.PENALTY_FOR_EVERY_ASK
    assert not terminated
    assert next_meta is not None and next_meta["turn"] == 1


def test_process_turn_correct_guess(patch_unique_numbers):
    runner = unique_numbers._UniqueNumbersRunner()
    numbers = [1, 2, 1]
    metadata = _make_metadata(numbers)

    msg_log = [{"role": "assistant", "content": "There are 2 unique numbers"}]
    obs, reward, terminated, _, next_meta = runner.process_turn(msg_log, metadata)

    assert obs == {"role": "user", "content": "<done>"}
    assert reward == 1.0
    assert terminated
    assert next_meta is None


def test_process_turn_incorrect_guess(patch_unique_numbers):
    runner = unique_numbers._UniqueNumbersRunner()
    numbers = [1, 2, 1]
    metadata = _make_metadata(numbers)

    msg_log = [{"role": "assistant", "content": "There are 3 unique numbers"}]
    obs, reward, terminated, _, _ = runner.process_turn(msg_log, metadata)

    assert obs["content"] == "<done>"
    assert reward == unique_numbers.PENALTY_FOR_INCORRECT_GUESS
    assert terminated


def test_process_turn_no_message(patch_unique_numbers):
    runner = unique_numbers._UniqueNumbersRunner()
    numbers = [1, 2, 1]
    metadata = _make_metadata(numbers)

    msg_log = []
    obs, reward, terminated, _, _ = runner.process_turn(msg_log, metadata)

    assert obs["content"] == "<done>"
    assert reward == unique_numbers.PENALTY_FOR_NO_GUESS
    assert terminated


def test_run_prompt_async_basic(patch_adk_utils):
    agent = adk_utils.Agent(instruction="hi")
    runner = adk_utils.Runner(agent=agent, responses=["7"])
    asyncio.run(runner.session_service.create_session("app", "u1"))
    out = asyncio.run(adk_utils.run_prompt_async(runner, "u1", "hello", silence=True))
    assert out == "7"


def test_run_prompt_async_retry(monkeypatch, patch_adk_utils):
    class ErrorRunner(adk_utils.Runner):
        def __init__(self):
            super().__init__(agent=adk_utils.Agent())
            self.call = 0

        async def run_async(self, user_id: str, session_id: str, new_message):
            self.call += 1
            if self.call == 1:
                raise adk_utils.ServerError()
            async for e in super().run_async(user_id, session_id, new_message):
                yield e

    runner = ErrorRunner()
    asyncio.run(runner.session_service.create_session("app", "u2"))
    res = asyncio.run(
        adk_utils.run_prompt_async(
            runner, "u2", "hi", silence=True, max_retries=2, initial_delay=0
        )
    )
    assert res == "ack"
    assert runner.call == 2


def test_extract_conversation_history(patch_adk_utils):
    agent = adk_utils.Agent(instruction="inst")
    runner = adk_utils.Runner(agent=agent, responses=["42"])
    asyncio.run(runner.session_service.create_session("app", "u3"))
    asyncio.run(adk_utils.run_prompt_async(runner, "u3", "question", silence=True))
    session_id, convo = adk_utils.extract_conversation_history(runner, "u3")
    assert session_id == "s1"
    assert convo[0] == {"role": "instruction", "content": "inst"}
    assert convo[1]["role"] == "user"
    assert convo[2]["content"] == "42"

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

from contextlib import ExitStack
from tempfile import TemporaryDirectory

import pytest
import ray
from transformers import AutoTokenizer

from nemo_rl.environments.code_environment import (
    CodeEnvConfig,
    CodeEnvironment,
    CodeEnvMetadata,
    _resolve_execution_limits,
    _supports_memory_limit,
)

MODEL_NAME = "meta-llama/Llama-3.2-1B"

cfg: CodeEnvConfig = {
    "num_workers": 2,
    "terminate_on_evaluation": True,
}


def _make_metadata(
    working_dir: str,
    *,
    timeout_seconds: float | None = None,
    memory_limit_bytes: int | None = None,
) -> CodeEnvMetadata:
    metadata: CodeEnvMetadata = {
        "context": {},
        "working_dir": working_dir,
    }
    if timeout_seconds is not None:
        metadata["timeout_seconds"] = timeout_seconds
    if memory_limit_bytes is not None:
        metadata["memory_limit_bytes"] = memory_limit_bytes
    return metadata


def _step_code(
    env_actor,
    code: str,
    metadata: CodeEnvMetadata,
):
    return ray.get(
        env_actor.step.remote(
            [[{"role": "user", "content": f"<code>{code}</code>"}]],
            [metadata],
        )
    )


@pytest.fixture(scope="function")
def code_env():
    """Create a code environment for testing."""
    env_actor = None
    try:
        env_actor = CodeEnvironment.remote(cfg)
        yield env_actor
    finally:
        if env_actor:
            ray.kill(env_actor)


@pytest.fixture(scope="function")
def tokenizer():
    """Loads the tokenizer for the tests."""
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(
        "Tokenizer loaded. "
        f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id}), "
        f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})"
    )
    return tokenizer


@pytest.fixture(scope="function")
def cluster():
    """Create a virtual cluster for testing."""
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster

    cluster_instance = None
    cluster_name = f"test-code-cluster-{id(cluster_instance)}"
    print(f"\nCreating virtual cluster '{cluster_name}'...")
    try:
        cluster_instance = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            num_gpus_per_node=1,
            max_colocated_worker_groups=2,
        )
        yield cluster_instance
    finally:
        print(f"\nCleaning up cluster '{cluster_name}'...")
        if cluster_instance:
            cluster_instance.shutdown()


def test_resolve_execution_limits_uses_metadata_overrides():
    metadata = CodeEnvMetadata(
        context={},
        working_dir="/tmp/code-env",
        timeout_seconds=2.0,
        memory_limit_bytes=2048,
    )

    limits = _resolve_execution_limits(
        metadata,
        default_timeout_seconds=1.0,
        default_memory_limit_bytes=1024,
    )

    assert limits.timeout_seconds == 2.0
    assert limits.memory_limit_bytes == 2048


def test_resolve_execution_limits_uses_defaults_when_metadata_missing():
    metadata = CodeEnvMetadata(
        context={},
        working_dir="/tmp/code-env",
    )

    limits = _resolve_execution_limits(
        metadata,
        default_timeout_seconds=1.5,
        default_memory_limit_bytes=4096,
    )

    assert limits.timeout_seconds == 1.5
    assert limits.memory_limit_bytes == 4096


def test_untrusted_code(code_env):
    """Test whether the code environment can block untrusted code."""
    codes = [
        "with open('allowed_file.txt', 'w') as fout:\n"
        "    fout.write('some content')\n"
        "with open('allowed_file.txt') as fin:\n"
        "    content = fin.read()\n"
        "content",
        "with open('/etc/passwd', 'r') as fin:\n    fin.read()",
        "import math\nround(math.sqrt(8))",
        "import os",
    ]
    results = [
        "\n\n<result>\n'some content'\n</result>",
        (
            "\n\n<result>\n"
            "PermissionError("
            "'Access beyond the temporary working directory is blocked'"
            ")\n"
            "</result>"
        ),
        "\n\n<result>\n3\n</result>",
        (
            "<result>"
            "PermissionError('Importing system and network modules is blocked')"
            "</result>"
        ),
    ]

    with ExitStack() as stack:
        temp_dirs = [stack.enter_context(TemporaryDirectory()) for _ in codes]
        metadata_batch = [
            _make_metadata(temp_dir)
            for temp_dir in temp_dirs
        ]

        output = ray.get(
            code_env.step.remote(
                [
                    [{"role": "user", "content": f"<code>{code}</code>"}]
                    for code in codes
                ],
                metadata_batch,
            )
        )
        responses = [obs["content"] for obs in output.observations]

    assert responses == results, f"Got wrong output {responses}"


def test_syntax_error_returns_observation_and_actor_stays_healthy(code_env):
    with TemporaryDirectory() as temp_dir:
        metadata = _make_metadata(temp_dir)
        output = _step_code(code_env, "def broken(:\n    pass", metadata)

        assert "SyntaxError" in output.observations[0]["content"]
        follow_up = _step_code(code_env, "1 + 1", output.metadata[0])
        assert follow_up.observations[0]["content"] == "<result>2</result>"


def test_timeout_returns_observation_and_actor_stays_healthy(code_env):
    with TemporaryDirectory() as temp_dir:
        metadata = _make_metadata(temp_dir, timeout_seconds=0.5)
        output = _step_code(code_env, "while True:\n    pass", metadata)

        assert "TimeoutError" in output.observations[0]["content"]
        follow_up = _step_code(code_env, "40 + 2", output.metadata[0])
        assert follow_up.observations[0]["content"] == "<result>42</result>"


@pytest.mark.skipif(
    not _supports_memory_limit(),
    reason="Memory limits require resource.RLIMIT_AS support",
)
def test_memory_limit_returns_observation_and_actor_stays_healthy(code_env):
    with TemporaryDirectory() as temp_dir:
        metadata = _make_metadata(
            temp_dir,
            memory_limit_bytes=256 * 1024 * 1024,
        )
        output = _step_code(
            code_env,
            "blob = bytearray(1024 * 1024 * 1024)\nlen(blob)",
            metadata,
        )

        assert "MemoryError" in output.observations[0]["content"]
        follow_up = _step_code(code_env, "6 * 7", output.metadata[0])
        assert follow_up.observations[0]["content"] == "<result>42</result>"


def test_default_timeout_applies_when_metadata_omits_limit():
    env_actor = None
    try:
        env_actor = CodeEnvironment.remote(
            CodeEnvConfig(
                num_workers=1,
                terminate_on_evaluation=True,
                default_timeout_seconds=0.5,
            )
        )
        with TemporaryDirectory() as temp_dir:
            metadata = _make_metadata(temp_dir)
            output = _step_code(env_actor, "while True:\n    pass", metadata)

        assert "TimeoutError" in output.observations[0]["content"]
    finally:
        if env_actor:
            ray.kill(env_actor)


def test_multiturn_context_survives_subprocess_boundary(code_env):
    with TemporaryDirectory() as temp_dir:
        metadata = _make_metadata(temp_dir)
        first_output = _step_code(
            code_env,
            "def square(x):\n    return x * x\nvalue = 7",
            metadata,
        )
        second_output = _step_code(
            code_env,
            "square(value)",
            first_output.metadata[0],
        )

        assert second_output.observations[0]["content"] == "<result>49</result>"


@pytest.mark.hf_gated
def test_vllm_execute_code(cluster, tokenizer, code_env):
    """Test that vLLM can call the code executor."""
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.experience.rollouts import run_multi_turn_rollout
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.models.generation.vllm import VllmGeneration

    basic_vllm_test_config = {
        "backend": "vllm",
        "model_name": MODEL_NAME,
        "tokenizer_name": None,
        "dtype": "bfloat16",
        "max_new_tokens": 100,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "async_engine": False,
            "precision": "bfloat16",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "max_model_len": 1024,
            "disable_log_stats": True,
            "disable_log_requests": True,
            "gpu_memory_utilization": 0.6,
            "enforce_eager": "False",
        },
        "colocated": {
            "enabled": True,
            "resources": {
                "gpus_per_node": None,
                "num_nodes": None,
            },
        },
    }

    codes = [
        "<code>x = 3; y = 4</code>\nThis is some regular text.\n<code>x + y</code>\n",
        "<code>\ndef f(x):\n    return x * x\n\nf(2)\n</code>\n",
    ]
    results = ["<result>7</result>", "\n<result>\n4\n</result>"]

    message_logs = []
    metadata_batch = []
    temp_dirs = []
    for code in codes:
        prompt = code * 4
        token_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0]
        temp_dir = TemporaryDirectory()
        message_logs.append(
            [{"role": "user", "content": prompt, "token_ids": token_ids}]
        )
        metadata_batch.append(_make_metadata(temp_dir.name))
        temp_dirs.append(temp_dir)

    initial_batch = BatchedDataDict(
        {
            "message_log": message_logs,
            "extra_env_info": metadata_batch,
            "task_name": ["code_execution"] * len(codes),
            "stop_strings": [["</code>"]] * len(codes),
        }
    )

    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)
    vllm_generation = VllmGeneration(cluster, vllm_config)

    task_to_env = {"code_execution": code_env}

    try:
        vllm_generation.prepare_for_generation()
        final_batch, _ = run_multi_turn_rollout(
            policy_generation=vllm_generation,
            input_batch=initial_batch,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            max_seq_len=256,
            max_rollout_turns=2,
            greedy=True,
        )
    finally:
        vllm_generation.finish_generation()
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    for i, msg_log in enumerate(final_batch["message_log"]):
        last_msg = msg_log[-1]
        assert last_msg["role"] == "environment"
        assert last_msg["content"] == results[i], (
            f"Expected {results[i]}, got {last_msg['content']}"
        )

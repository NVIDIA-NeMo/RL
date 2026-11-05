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

"""CPU contract tests for the external generation adapter and exporter.

The cookbook transport is exercised with a small NeMo-RL interface stand-in,
so these tests do not need a local inference engine or GPU dependencies.
"""

import abc
import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cwtrain")
ROOT = Path(__file__).resolve().parents[4]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter(monkeypatch):
    """Load real transport code without importing local generation engines."""

    class GenerationInterface(abc.ABC):
        pass

    class WeightSynchronizer(abc.ABC):
        pass

    packages = {
        "nemo_rl": ROOT / "nemo_rl",
        "nemo_rl.models": ROOT / "nemo_rl/models",
        "nemo_rl.models.generation": ROOT / "nemo_rl/models/generation",
        "nemo_rl.models.generation.coreweave": ROOT
        / "nemo_rl/models/generation/coreweave",
        "nemo_rl.models.policy": ROOT / "nemo_rl/models/policy",
        "nemo_rl.distributed": ROOT / "nemo_rl/distributed",
        "nemo_rl.weight_sync": ROOT / "nemo_rl/weight_sync",
    }
    names = [
        *packages,
        "nemo_rl.models.generation.interfaces",
        "nemo_rl.distributed.batched_data_dict",
        "nemo_rl.weight_sync.interfaces",
    ]
    saved = dict(sys.modules)
    for name in names:
        module = types.ModuleType(name)
        if name in packages:
            module.__path__ = [str(packages[name])]
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules[
        "nemo_rl.models.generation.interfaces"
    ].GenerationInterface = GenerationInterface
    sys.modules["nemo_rl.distributed.batched_data_dict"].BatchedDataDict = dict
    sys.modules[
        "nemo_rl.weight_sync.interfaces"
    ].WeightSynchronizer = WeightSynchronizer
    for name in list(sys.modules):
        if name.startswith("cwtrain.integrations.nemo_rl"):
            monkeypatch.delitem(sys.modules, name)
    module = load_file(
        "nemo_rl.models.generation.coreweave.coreweave_generation",
        "nemo_rl/models/generation/coreweave/coreweave_generation.py",
    )
    yield module
    for name in list(sys.modules):
        if name not in saved and (
            name.startswith("nemo_rl.")
            or name.startswith("cwtrain.integrations.nemo_rl")
        ):
            del sys.modules[name]


def config(**overrides):
    settings = dict(
        base_url="https://gateway.example/v1",
        model="policy",
        deployment_id="deployment",
        run_id="test",
        state_dir="/tmp/coreweave-test",
        max_model_len=16,
    )
    settings.update(overrides)
    return dict(
        backend="coreweave",
        max_new_tokens=8,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        stop_token_ids=[2],
        stop_strings=None,
        _pad_token_id=0,
        colocated={
            "enabled": False,
            "resources": {"num_nodes": None, "gpus_per_node": None},
        },
        coreweave_cfg=settings,
    )


def generation(adapter, **overrides):
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    return adapter.CoreWeaveGeneration(config(**overrides), tokenizer)


@pytest.mark.parametrize(
    "changes",
    [
        dict(wait=False),
        dict(verify=False),
        dict(require_logprobs=False),
        dict(prompt_cache_policy="preserve"),
    ],
)
def test_reject_unsafe_update_settings(adapter, changes):
    with pytest.raises(ValueError):
        generation(adapter, **changes)


@pytest.mark.parametrize(
    "ids, logprobs", [(None, (-1.0,)), ((3, 2), (-1.0,)), ((3,), (float("nan"),))]
)
def test_reject_missing_or_bad_sampled_data(adapter, ids, logprobs):
    from cwtrain.integrations.rollout import Completion, RolloutError

    result = Completion(
        index=0, text="x", finish_reason="stop", token_ids=ids, token_logprobs=logprobs
    )
    with pytest.raises(RolloutError):
        generation(adapter)._decode(result)


def test_request_preserves_tokens_and_batch_order(adapter, monkeypatch):
    import httpx
    from cwtrain.integrations.rollout import CompletionSampler

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        choices = [
            dict(
                index=i,
                text="ignored",
                finish_reason="stop",
                logprobs=dict(
                    tokens=["token_id:7", "token_id:2"], token_logprobs=[-0.3, -0.7]
                ),
            )
            for i in range(body["n"])
        ]
        return httpx.Response(200, json={"choices": choices})

    original = CompletionSampler.__init__
    monkeypatch.setattr(
        CompletionSampler,
        "__init__",
        lambda self, cfg: original(self, cfg, transport=httpx.MockTransport(respond)),
    )
    gen = generation(adapter)
    gen.weight_synchronizer = SimpleNamespace(
        served_identity="test-v000001",
        is_stale=False,
        config=SimpleNamespace(wait=True),
    )
    out = gen.generate(
        dict(
            input_ids=torch.tensor([[5, 6, 0], [8, 0, 0], [5, 6, 0]]),
            input_lengths=torch.tensor([2, 1, 2]),
        ),
        greedy=False,
    )
    assert len(requests) == 2
    assert {tuple(request["prompt"]) for request in requests} == {(5, 6), (8,)}
    assert all(request["nvext"]["cache_salt"] == "test-v000001" for request in requests)
    assert out["output_ids"].tolist() == [
        [5, 6, 7, 2, 0],
        [8, 7, 2, 0, 0],
        [5, 6, 7, 2, 0],
    ]
    assert out["generation_lengths"].tolist() == [2, 2, 2]
    assert out["unpadded_sequence_lengths"].tolist() == [4, 3, 4]
    assert out["logprobs"][1, 1:3].tolist() == pytest.approx([-0.3, -0.7])


def test_context_overflow_never_sends_request(adapter):
    from cwtrain.integrations.rollout import RolloutError

    with pytest.raises(RolloutError, match="context limit"):
        asyncio.run(
            generation(adapter)._sample_group(None, list(range(16)), 1, None, False)
        )


def test_export_preserves_layout_and_source_precision(adapter, tmp_path):
    from safetensors import safe_open

    exporter = load_file(
        "nemo_rl.models.policy.checkpoint_export",
        "nemo_rl/models/policy/checkpoint_export.py",
    )
    output = tmp_path / "export"
    model_config = tmp_path / "config"
    model_config.mkdir()
    (model_config / "config.json").write_text("{}")
    tensors = [
        ("weight", torch.ones(2, dtype=torch.bfloat16)),
        ("bias", torch.tensor([1.00001], dtype=torch.float32)),
    ]
    result = exporter.export_hf_checkpoint(
        iter(tensors),
        str(output),
        ["weight", "bias"],
        str(model_config),
        100,
        {"weight": "BF16", "bias": "F32"},
    )
    assert result["tensors"] == 2
    index = json.loads((output / "model.safetensors.index.json").read_text())
    with safe_open(str(output / index["weight_map"]["bias"]), framework="pt") as shard:
        assert torch.equal(shard.get_tensor("bias"), tensors[1][1])
    assert (output / "config.json").is_file()
    with pytest.raises(RuntimeError, match="restore"):
        exporter.export_hf_checkpoint(
            iter(tensors),
            str(output),
            ["weight", "bias"],
            str(model_config),
            100,
            {"weight": "F32", "bias": "F32"},
        )


def test_write_failure_does_not_stop_other_ranks_gathers(
    adapter, tmp_path, monkeypatch
):
    exporter = load_file(
        "nemo_rl.models.policy.checkpoint_export",
        "nemo_rl/models/policy/checkpoint_export.py",
    )

    def fail_write(*args):
        raise OSError("disk full")

    monkeypatch.setattr(exporter._ShardWriter, "add", fail_write)
    gathered = []

    def params():
        for name in ("a", "b", "c"):
            gathered.append(name)
            yield name, torch.ones(1)

    with pytest.raises(RuntimeError, match="disk full"):
        exporter.export_hf_checkpoint(
            params(), str(tmp_path / "out"), ["a", "b", "c"], None, 100, None
        )
    assert gathered == ["a", "b", "c"]


@pytest.mark.parametrize("fail_activation", [False, True])
def test_hotload_only_commits_verified_weights(
    adapter, tmp_path, monkeypatch, fail_activation
):
    from cwtrain.integrations.nemo_rl import weight_sync

    exporter = load_file(
        "nemo_rl.models.policy.checkpoint_export",
        "nemo_rl/models/policy/checkpoint_export.py",
    )
    checkpoint = tmp_path / "base"
    base_config = tmp_path / "base-config"
    base_config.mkdir()
    (base_config / "config.json").write_text("{}")
    exporter.export_hf_checkpoint(
        iter([("weight", torch.ones(2))]),
        str(checkpoint),
        ["weight"],
        str(base_config),
        100,
        {"weight": "F32"},
    )
    events = []
    failure = [fail_activation]
    weight = [2.0]
    operation = SimpleNamespace(
        id="operation", completed=True, state=SimpleNamespace(name="COMPLETED")
    )

    class Loader:
        def completed_hot_load(self, identity):
            return None

        def read_manifest(self, identity):
            return None

        def upload(self, *args, **kwargs):
            events.append("upload")

        def wait_for_deployment_ready(self, **kwargs):
            pass

        def trigger(self, *args, **kwargs):
            events.append("trigger")
            return operation

        def wait(self, *args, **kwargs):
            events.append("wait")
            if failure[0]:
                raise TimeoutError("partial replica update")
            return operation

        def verify_served(self, operation_id, identity):
            events.append("verify")
            return [SimpleNamespace()]

        def upload_manifest(self, *args):
            events.append("manifest")
            return "s3://bucket/manifest"

    class Policy:
        cfg = {"model_name": str(checkpoint)}

        def run_all_workers_single_data(self, method, **kwargs):
            assert method == "export_hf_checkpoint"
            return [
                exporter.export_hf_checkpoint(
                    iter([("weight", torch.full((2,), weight[0]))]),
                    kwargs["path"],
                    kwargs["allowed_names"],
                    kwargs["config_dir"],
                    100,
                    kwargs["dtypes"],
                )
            ]

    monkeypatch.setenv("COREWEAVE_API_TOKEN", "test-token")
    monkeypatch.setattr(weight_sync, "HotLoader", lambda *args, **kwargs: Loader())
    module = load_file(
        "nemo_rl.weight_sync.coreweave_weight_synchronizer",
        "nemo_rl/weight_sync/coreweave_weight_synchronizer.py",
    )
    gen = generation(adapter, state_dir=str(tmp_path / "state"), full_every=20)
    sync = module.CoreWeaveHotLoadWeightSynchronizer(Policy(), gen)
    gen.weight_synchronizer = sync
    assert sync.is_stale and sync.served_identity is None
    if fail_activation:
        with pytest.raises(TimeoutError):
            sync.sync_weights()
        assert sync.is_stale and sync.served_identity is None
        assert "manifest" not in events
    else:
        sync.sync_weights()
        assert not sync.is_stale
        assert sync.served_identity == "test-v000000"
        assert events == ["upload", "trigger", "wait", "verify", "manifest"]
        unchanged = sync.sync_weights()
        assert unchanged["hotload/skipped_empty"] == 1.0
        assert not sync.is_stale
        assert (
            tmp_path / "state" / "served_identity"
        ).read_text().strip() == sync.served_identity
        weight[0] = 3.0
        failure[0] = True
        with pytest.raises(TimeoutError):
            sync.sync_weights()
        assert sync.is_stale
        assert not (tmp_path / "state" / "served_identity").exists()
        with pytest.raises(adapter.RolloutError, match="verified checkpoint"):
            gen._cache_salt()


def test_no_sampling_before_verified_checkpoint(adapter):
    from cwtrain.integrations.rollout import RolloutError

    with pytest.raises(RolloutError, match="verified checkpoint"):
        generation(adapter)._cache_salt()


def gym_master_config(tmp_path):
    """A synchronous policy and Gym endpoint sharing the checkpoint record."""
    generation_config = config(state_dir=str(tmp_path))
    return SimpleNamespace(
        policy={
            "generation": generation_config,
            "automodel_cfg": {"enabled": True},
            "megatron_cfg": {"enabled": False},
            "tokenizer": {"name": "/shared/model"},
            "offload_policy_before_refit": False,
        },
        grpo=SimpleNamespace(async_grpo=SimpleNamespace(enabled=False)),
        data_plane=None,
        env={
            "should_use_nemo_gym": True,
            "nemo_gym": {
                "config_paths": [
                    "responses_api_models/external_model/configs/external_model_for_training.yaml"
                ],
                "policy_tokenizer_path": "/shared/model",
                "policy_max_model_len": 16,
                "policy_served_identity_file": str(tmp_path / "served_identity"),
                "policy_api_key_file": None,
            },
        },
    )


def test_gym_accepts_matching_external_endpoint(adapter, tmp_path):
    master = gym_master_config(tmp_path)
    adapter.CoreWeaveGeneration.validate_settings(master)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("validated", [False, True])
def test_gym_handles_draft_config_models(adapter, tmp_path, enabled, validated):
    # Import inside the fixture's package tree to avoid optional policy dependencies.
    from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

    master = gym_master_config(tmp_path)
    master.policy["draft"] = (
        Eagle3DraftConfig(enabled=enabled) if validated else {"enabled": enabled}
    )
    if enabled:
        with pytest.raises(ValueError, match="draft training"):
            adapter.CoreWeaveGeneration.validate_settings(master)
    else:
        adapter.CoreWeaveGeneration.validate_settings(master)


@pytest.mark.parametrize(
    "key,value",
    [
        ("policy_tokenizer_path", "/different/model"),
        ("policy_max_model_len", 17),
        ("policy_served_identity_file", "/different/served_identity"),
        ("policy_api_key_file", "/different/key"),
    ],
)
def test_gym_rejects_inconsistent_endpoint_metadata(adapter, tmp_path, key, value):
    master = gym_master_config(tmp_path)
    master.env["nemo_gym"][key] = value
    with pytest.raises(ValueError, match=key):
        adapter.CoreWeaveGeneration.validate_settings(master)


def test_gym_requires_exact_token_adapter(adapter, tmp_path):
    master = gym_master_config(tmp_path)
    master.env["nemo_gym"]["config_paths"] = [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
    ]
    with pytest.raises(ValueError, match="external_model"):
        adapter.CoreWeaveGeneration.validate_settings(master)

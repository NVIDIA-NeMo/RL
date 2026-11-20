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
"""Co-resident generation owner (nemo_rl/models/megatron/sm_partition.py): config, MPS env, worker check (CPU), and
the same-GPU CUDA IPC refit between two processes (GPU)."""

import os
import subprocess
import tempfile
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron import sm_partition as smp

MPS = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps"}


@pytest.mark.parametrize("contiguous", [True, False])
def test_bind_shared_expert_stacks_uses_native_megatron_buffers(contiguous):
    # Use the installed class so the test cannot invent an unavailable Core API.
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    experts = InferenceGroupedMLP.__new__(InferenceGroupedMLP)
    torch.nn.Module.__init__(experts)
    experts.num_local_experts = 2
    experts._concatenated_weights_built = False
    sources = {}
    for linear_name, shape in [("linear_fc1", (4, 3)), ("linear_fc2", (3, 4))]:
        linear = torch.nn.Module()
        packed = torch.arange(2 * shape[0] * shape[1], dtype=torch.float32).reshape(
            2, *shape
        )
        for index in range(2):
            source = packed[index] if contiguous else packed[index].clone()
            linear.register_parameter(f"weight{index}", torch.nn.Parameter(source))
            sources[f"{linear_name}.weight{index}"] = source
        experts.add_module(linear_name, linear)

    stats = smp.bind_shared_expert_stacks(experts, sources)
    assert experts._concatenated_weights_built is contiguous
    assert stats["shared_expert_layers"] == int(contiguous)
    if contiguous:
        assert stats["shared_expert_bytes"] == sum(
            value.numel() * value.element_size() for value in sources.values()
        )
        sources["linear_fc1.weight0"].add_(7)
        torch.testing.assert_close(
            experts._fc1_weight[0], sources["linear_fc1.weight0"]
        )
        assert (
            experts._fc1_weight.data_ptr() == sources["linear_fc1.weight0"].data_ptr()
        )
        assert (
            experts._fc2_weight.data_ptr() == sources["linear_fc2.weight0"].data_ptr()
        )
    else:
        assert stats["shared_expert_bytes"] == 0
        assert "_fc1_weight" not in experts._buffers
        assert "_fc2_weight" not in experts._buffers


def _cfg(**over):
    c = {
        "megatron_cfg": {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
        },
        "generation": {
            "backend": "megatron",
            "refit_transport": "ipc",
            "colocated": {
                "enabled": False,
                "engine_sm_fraction": 0.42,
                "sm_partition_backend": "mps_percentage",
            },
            "mcore_generation_config": {
                "tensor_model_parallel_size": 1,
                "expert_model_parallel_size": 4,
            },
        },
    }
    for path, v in over.items():
        d = c
        keys = path.split(".")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = v
    return c


def test_off_is_inert():
    c = _cfg(**{"generation.colocated.engine_sm_fraction": None})
    assert smp.engine_sm_fraction(c) is None
    smp.validate(c, env={})
    assert smp.actor_env(c, "engine", env=MPS) == {}
    assert smp.check_worker(c, "engine") is None


def test_validate_accepts_the_supported_config():
    smp.validate(_cfg(), env=MPS)
    smp.validate(
        _cfg(**{"generation.mcore_generation_config.expert_model_parallel_size": 8}),
        env=MPS,
    )
    smp.validate(
        _cfg(
            **{
                "megatron_cfg.tensor_model_parallel_size": 2,
                "generation.mcore_generation_config.tensor_model_parallel_size": 2,
            }
        ),
        env=MPS,
    )


@pytest.mark.parametrize(
    "over, env, match",
    [
        ({"generation.colocated.engine_sm_fraction": -0.01}, MPS, r"\[0, 1\)"),
        ({"generation.colocated.engine_sm_fraction": 1.0}, MPS, r"\[0, 1\)"),
        ({"generation.backend": "vllm"}, MPS, "Megatron generation backend"),
        ({"generation.colocated.enabled": True}, MPS, "colocated.enabled=false"),
        ({"generation.refit_transport": None}, MPS, "refit_transport=ipc"),
        ({}, {}, "MPS control daemon"),
        (
            {
                "megatron_cfg.pipeline_model_parallel_size": 2,
                "generation.mcore_generation_config.pipeline_model_parallel_size": 2,
            },
            MPS,
            "pipeline parallelism",
        ),
    ],
)
def test_validate_rejects(over, env, match):
    with pytest.raises(ValueError, match=match):
        smp.validate(_cfg(**over), env=env)


@pytest.mark.parametrize("engine_tp", [2, 4])
def test_validate_accepts_local_tp_refinement(engine_tp: int) -> None:
    smp.validate(
        _cfg(
            **{
                "generation.mcore_generation_config.tensor_model_parallel_size": engine_tp
            }
        ),
        env=MPS,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "megatron_cfg.tensor_model_parallel_size": 4,
            "generation.mcore_generation_config.tensor_model_parallel_size": 2,
        },
        {
            "megatron_cfg.tensor_model_parallel_size": 2,
            "generation.mcore_generation_config.tensor_model_parallel_size": 4,
        },
        {"generation.mcore_generation_config.expert_model_parallel_size": 8},
        {"generation.mcore_generation_config.expert_model_parallel_size": 2},
        {
            "megatron_cfg.expert_tensor_parallel_size": 2,
            "generation.mcore_generation_config.expert_tensor_parallel_size": 1,
        },
        {"megatron_cfg.context_parallel_size": 2},
        {
            "megatron_cfg.pipeline_model_parallel_size": 2,
            "generation.mcore_generation_config.pipeline_model_parallel_size": 1,
        },
    ],
)
def test_validate_allows_layouts_requiring_native_residual_transfers(
    overrides: dict,
) -> None:
    smp.validate(_cfg(**overrides), env=MPS)


def test_validate_keeps_inference_context_parallel_limit() -> None:
    with pytest.raises(ValueError, match="does not support context parallelism"):
        smp.validate(
            _cfg(**{"generation.mcore_generation_config.context_parallel_size": 2}),
            env=MPS,
        )


def test_validate_rejects_unsupported_features():
    with pytest.raises(ValueError, match="cluster.segment_size is not built yet"):
        smp.validate(
            _cfg(), env=MPS, unsupported={"cluster.segment_size": True, "PPO": False}
        )
    smp.validate(_cfg(), env=MPS, unsupported={"cluster.segment_size": False})


@pytest.mark.parametrize("frac", [0.01, 0.26, 0.42, 0.5, 0.64, 0.995, 0.999])
def test_mps_percentage_caps_the_learner_only(frac):
    assert smp.mps_percentage(frac, "engine") is None
    assert 1 <= smp.mps_percentage(frac, "learner") <= 99


def test_actor_env():
    c = _cfg()
    assert smp.actor_env(c, "engine", env=MPS) == {**MPS}  # uncapped
    assert smp.actor_env(c, "learner", env=MPS) == {smp.MPS_PCT_ENV: "58", **MPS}
    assert smp.actor_env(c, "learner", env={}) == {smp.MPS_PCT_ENV: "58"}


@pytest.mark.parametrize(
    "role,pct,seen,error",
    [
        ("engine", None, 152, None),
        ("engine", "100", 152, None),
        ("engine", "42", 152, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
        ("engine", "58", 152, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
        ("learner", "58", 88, None),
        ("learner", "58", 152, "not running under its MPS limit"),
        ("learner", "42", 88, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
        ("learner", "100", 88, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
        ("learner", None, 88, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
    ],
)
def test_check_worker_verifies_role_percentage(
    monkeypatch: pytest.MonkeyPatch,
    role: smp.Role,
    pct: str | None,
    seen: int,
    error: str | None,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=seen),
    )
    monkeypatch.setattr(smp, "_device_sm_count", lambda: 152)
    if pct is None:
        monkeypatch.delenv(smp.MPS_PCT_ENV, raising=False)
    else:
        monkeypatch.setenv(smp.MPS_PCT_ENV, pct)
    if error is None:
        info = smp.check_worker(_cfg(), role)
        assert info["mps_pct"] == pct
        assert info["sms_seen"] == seen
    else:
        with pytest.raises(RuntimeError, match=error):
            smp.check_worker(_cfg(), role)


# ---- same-GPU CUDA IPC refit, two processes (GPU) ----


def _learner(q_to_engine, q_from_engine):
    torch.cuda.set_device(0)
    g = torch.Generator().manual_seed(0)
    buf = torch.randn(1 << 20, generator=g).to(
        "cuda", torch.bfloat16
    )  # one contiguous buffer, as DDP's
    named = [
        ("a.weight", buf[: 512 * 1024].view(512, 1024)),
        ("b.bias", buf[512 * 1024 : 512 * 1024 + 1024]),
        ("e.weight0", buf[600000 : 600000 + 64 * 64].view(64, 64)),
    ]
    q_to_engine.put(smp.export_refit_handles(named))
    while True:
        msg = q_from_engine.get(timeout=120)
        if msg == "quit":
            return
        assert msg == "update", msg
        buf.add_(1.0)  # the optimizer step, in place
        torch.cuda.synchronize()
        q_to_engine.put({n: t.float().cpu() for n, t in named})


def _engine(q_from_learner, q_to_learner, out):
    try:
        torch.cuda.set_device(0)
        src = smp.open_refit_handles(q_from_learner.get())
        dst = {
            "a.weight": torch.zeros(512, 1024, device="cuda", dtype=torch.bfloat16),
            "b.bias": torch.zeros(1024, device="cuda", dtype=torch.bfloat16),
            "e.weight0": torch.zeros(64, 64, device="cuda", dtype=torch.float16),
        }  # engine-held dtype: copy_ casts
        res = {}
        q_to_learner.put("update")
        expect = q_from_learner.get()
        res["bytes"] = smp.copy_refit(src, dst.items())["bytes"]
        res["bitwise"] = all(
            torch.equal(dst[n].float().cpu(), expect[n].to(dst[n].dtype).float())
            for n in dst
        )
        q_to_learner.put("update")
        expect = q_from_learner.get()  # negative control: no copy after this update
        res["stale_detected"] = not all(
            torch.equal(dst[n].float().cpu(), expect[n].to(dst[n].dtype).float())
            for n in dst
        )
        try:
            smp.copy_refit(src, [("a.weight", dst["a.weight"])])
        except ValueError as ex:
            res["name_mismatch"] = "differ by name" in str(ex)
        try:
            smp.copy_refit(
                src,
                [("a.weight", dst["a.weight"].view(1024, 512))]
                + [(n, t) for n, t in dst.items() if n != "a.weight"],
            )
        except ValueError as ex:
            res["shape_mismatch"] = "has shape" in str(ex)
        out.put(res)
    except BaseException as ex:  # surfaced by the test
        out.put({"error": repr(ex)})
    finally:
        q_to_learner.put("quit")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_ipc_refit_two_processes():
    ctx = torch.multiprocessing.get_context("spawn")
    l2e, e2l, out = ctx.Queue(), ctx.Queue(), ctx.Queue()
    pl = ctx.Process(target=_learner, args=(l2e, e2l))
    pl.start()
    pe = ctx.Process(target=_engine, args=(l2e, e2l, out))
    pe.start()
    try:
        res = out.get(timeout=300)
        pe.join(30)
        pl.join(30)
        assert pe.exitcode == 0 and pl.exitcode == 0, (pe.exitcode, pl.exitcode)
        assert "error" not in res, res
        assert (
            res["bitwise"]
            and res["stale_detected"]
            and res["name_mismatch"]
            and res["shape_mismatch"]
        ), res
        assert res["bytes"] == 512 * 1024 * 2 + 1024 * 2 + 64 * 64 * 2
    finally:
        for proc in (pe, pl):
            if proc.is_alive():
                proc.terminate()
            proc.join(10)


def test_refit_fails_fast() -> None:
    ray = pytest.importorskip("ray")
    from nemo_rl.weight_sync.megatron_weight_synchronizer import CoresidentRefit

    with tempfile.TemporaryDirectory(prefix="nrl-smp-") as ray_dir:
        try:
            ray.init(num_cpus=1, num_gpus=0, include_dashboard=False, _temp_dir=ray_dir)

            @ray.remote(num_gpus=1)
            def pending() -> None:
                pass  # Cannot start on this CPU-only cluster.

            @ray.remote
            def boom() -> None:
                raise KeyError("engine failed mid-refit")

            group = _Group(
                [], "engine", {"coresident_pause": [pending.remote(), boom.remote()]}
            )
            refit = CoresidentRefit(
                SimpleNamespace(cfg=_cfg()),
                SimpleNamespace(worker_group=group),
                refit_timeout_s=60.0,
            )
            with pytest.raises(
                ray.exceptions.RayTaskError, match="engine failed mid-refit"
            ):
                refit.sync_weights()
        finally:
            ray.shutdown()


@pytest.mark.parametrize("count", [1, 37])
def test_copy_refit_matches_materialized_sources(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None)
    )
    src = {
        str(i): torch.arange(24, dtype=torch.float32).reshape(4, 6) + i
        for i in range(count)
    }
    dst = {n: torch.empty_like(t, dtype=torch.bfloat16) for n, t in src.items()}
    src["strided"] = torch.arange(24, dtype=torch.float32).reshape(6, 4).t()
    dst["strided"] = torch.empty(4, 6, dtype=torch.float16)
    src["large"] = torch.arange(256 * 1024, dtype=torch.float32).reshape(512, 512)
    dst["large"] = torch.empty_like(src["large"], dtype=torch.bfloat16)
    src["float_buffer"] = torch.arange(7, dtype=torch.float32)
    dst["float_buffer"] = torch.empty_like(src["float_buffer"])
    src["shared"] = dst["shared"] = torch.ones(3)
    stats = smp.copy_refit(src, dst.items())
    for name, tensor in dst.items():
        assert torch.equal(tensor, src[name].to(tensor.dtype)), name
    assert stats == {
        "bytes": float((count + 1) * 24 * 4 + 256 * 1024 * 4 + 7 * 4),
        "bytes_shared": 12.0,
    }


def test_copy_refit_rejects_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None)
    )
    src = {"a": torch.ones(4), "b": torch.ones(4)}
    dst = {"a": torch.zeros(4), "b": torch.zeros(5)}
    with pytest.raises(ValueError, match="has shape"):
        smp.copy_refit(src, dst.items())


def test_storage_fingerprint_moves_with_storage():
    a, b = torch.zeros(4), torch.zeros(8)
    fp = smp.storage_fingerprint([("a", a), ("b", b)])
    assert fp == smp.storage_fingerprint([("a", a), ("b", b)])
    a.add_(1.0)  # in-place update: same storage
    assert fp == smp.storage_fingerprint([("a", a), ("b", b)])
    b.data = torch.zeros(8)  # rebound storage (offload / reload): must change
    assert fp != smp.storage_fingerprint([("a", a), ("b", b)])


def test_partition_backend_swap_point(tmp_path, monkeypatch):
    assert smp.partition_backend(_cfg()) == "mps_percentage"
    with pytest.raises(ValueError, match="must be one of"):
        smp.validate(
            _cfg(**{"generation.colocated.sm_partition_backend": "green"}), env=MPS
        )
    static = _cfg(**{"generation.colocated.sm_partition_backend": "mps_static"})
    with pytest.raises(ValueError, match="NRL_MPS_SM_PARTITION_MAP"):
        smp.validate(static, env=MPS)
    m = tmp_path / "map.json"
    m.write_text(
        '{"0": {"engine": 7, "learner": 8, "engine_sms": 64, "learner_sms": 88}}'
    )
    env = {**MPS, smp.MPS_PARTITION_MAP_ENV: str(m)}
    smp.validate(static, env=env)
    assert smp.actor_env(static, "engine", env=env) == {
        **env
    }  # no percentage under the static backend
    monkeypatch.setenv(smp.MPS_PARTITION_MAP_ENV, str(m))
    monkeypatch.setenv("CUDA_MPS_SM_PARTITION", "existing")
    monkeypatch.setenv("NRL_MPS_SM_PARTITION_SMS", "152")
    smp.select_static_partition(_cfg(), "learner", 0)
    assert os.environ["CUDA_MPS_SM_PARTITION"] == "existing"
    smp.select_static_partition(static, "learner", 0)
    assert os.environ["CUDA_MPS_SM_PARTITION"] == "8"
    assert os.environ["NRL_MPS_SM_PARTITION_SMS"] == "88"


def test_bind_shared_then_copy_whatever_is_not_aliased(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU stand-in for the IPC-mapped source. Bound tensors alias the source and are skipped by the per-step copy;
    tensors that were never bound (dtype, derived view, strided source) or that the engine re-homed after binding
    (MCore concatenates expert weights into its own buffer on its first inference forward) are copied."""
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None)
    )
    src = {"w": torch.randn(4, 4), "e0": torch.randn(4), "h": torch.randn(4)}
    owner = torch.nn.Module()
    owner.w = torch.nn.Parameter(torch.zeros(4, 4))
    owner.register_buffer("h", torch.zeros(4))
    grouped = torch.zeros(8)
    e0 = grouped[:4]  # a view of a grouped parameter: never bound
    named = [("w", owner.w), ("e0", e0), ("h", owner.h)]
    stats = smp.bind_shared(src, named, {id(owner.w), id(owner.h), id(grouped)})
    assert stats == {"tensors_bound": 1, "tensors": 3}
    assert owner.w.data_ptr() == src["w"].data_ptr()
    buffer_before_backward = owner.h.clone()
    src["h"].add_(1.0)
    assert torch.equal(owner.h, buffer_before_backward)
    src["w"].add_(
        1.0
    )  # the learner's in-place optimizer write is what the engine reads
    assert torch.equal(owner.w.detach(), src["w"])
    out = smp.copy_refit(src, named)
    assert out["bytes_shared"] == 16 * 4 and out["bytes"] == 2 * 4 * 4
    assert torch.equal(e0, src["e0"]) and torch.equal(owner.h, src["h"])
    # The engine re-homes a bound tensor (its own copy): the next refit must copy it, not trust the binding.
    owner.w.data = owner.w.detach().clone()
    src["w"].add_(1.0)
    out = smp.copy_refit(src, named)
    assert out["bytes_shared"] == 0 and torch.equal(owner.w.detach(), src["w"])
    with pytest.raises(ValueError, match="differ by name"):
        smp.bind_shared(src, named[:1], set())
    strided = torch.nn.Parameter(torch.zeros(4, 4))
    assert (
        smp.bind_shared({"w": torch.randn(4, 4).t()}, [("w", strided)], {id(strided)})[
            "tensors_bound"
        ]
        == 0
    )


@pytest.mark.parametrize(
    "world, l_ep, e_ep", [(16, 4, 8), (16, 8, 8), (8, 2, 8), (4, 2, 4), (64, 8, 32)]
)
def test_engine_rank_gpus_put_experts_on_their_learner(world, l_ep, e_ep):
    gpus = smp.engine_rank_gpus(
        world,
        learner=smp.ParallelismLayout(1, l_ep, 1, 1, 1),
        engine=smp.ParallelismLayout(1, e_ep, 1, 1, 1),
    )
    assert sorted(gpus) == list(range(world))
    experts = 128
    for r, g in enumerate(gpus):
        e_block = range((r % e_ep) * experts // e_ep, (r % e_ep + 1) * experts // e_ep)
        l_block = range((g % l_ep) * experts // l_ep, (g % l_ep + 1) * experts // l_ep)
        assert set(e_block) <= set(l_block), (r, g)
    if l_ep == e_ep:
        assert gpus == list(range(world))


class _Group:
    """Record worker calls and return one reply per rank in worker order."""

    def __init__(self, log: list, role: str, replies: dict) -> None:
        self.log, self.role, self.replies = log, role, replies
        self.calls = []

    def run_all_workers_single_data(self, method: str, **kwargs) -> list:
        self.log.append(f"{self.role}.{method}")
        self.calls.append((method, kwargs))
        return self.replies.get(method, [])

    run_all_workers_multiple_data = run_all_workers_single_data


@pytest.fixture
def local_refit_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_rl.weight_sync import megatron_weight_synchronizer as mws

    def get(refs: list, *, timeout: float) -> list:
        assert timeout == 360.0
        for ref in refs:
            if isinstance(ref, Exception):
                raise ref
        return refs

    monkeypatch.setattr(mws.ray, "get", get)
    monkeypatch.setattr(mws.ray, "put", lambda value: value)


@pytest.fixture
def single_rank_workers(local_refit_rpc):
    from nemo_rl.models.generation.megatron import MegatronGeneration

    log = []
    policy = SimpleNamespace(
        cfg=_cfg(),
        worker_group=_Group(
            log,
            "learner",
            {
                "export_coresident_refit_handles": [
                    smp.RefitExport(smp.RefitPlacement("gpu0", 0), [], {}, "fp")
                ],
                "coresident_refit_source_ready": ["fp"],
            },
        ),
    )
    generation = object.__new__(MegatronGeneration)
    generation._owns_policy = False
    generation.cfg = {"mcore_generation_config": {"expose_http_server": True}}
    generation.dp_openai_server_base_urls = []
    generation._policy = SimpleNamespace(
        worker_group=_Group(
            log,
            "engine",
            {
                "coresident_refit_placement": [smp.RefitPlacement("gpu0", 0)],
                "open_coresident_refit_handles": [smp.RefitBinding([], {})],
                "coresident_refit_copy": [
                    {"seconds": 0, "bytes": 1, "bytes_shared": 0}
                ],
                "report_dp_openai_server_base_url": ["http://node:8000/v1"],
            },
        ),
    )
    return policy, generation, log


@pytest.mark.parametrize("weights", ["copy", "shared"])
def test_coresident_refit_publishes_only_in_sync_weights(single_rank_workers, weights):
    """Shared weights: the engine pauses before the optimizer write and stays paused until sync_weights (the loop's
    refit, after its parameter all-gather) copies the rest, bumps the epoch and resumes it. Copy: the write needs no
    pause. Either way the new version is published in exactly one place."""
    from nemo_rl.weight_sync import megatron_weight_synchronizer as mws

    policy, generation, log = single_rank_workers
    policy.cfg["generation"]["colocated"]["engine_weights"] = weights
    refit = mws.CoresidentRefit(policy=policy, generation=generation)
    refit.init_communicator()
    refit.before_learner_write()  # before the engine runs: nothing to pause
    refit.sync_weights()  # first refit starts the engine
    assert log[-2:] == [
        "engine.coresident_resume",
        "engine.report_dp_openai_server_base_url",
    ]
    assert generation.dp_openai_server_base_urls == ["http://node:8000/v1"]
    log.clear()
    refit.before_learner_write()
    assert log == (["engine.coresident_pause"] if weights == "shared" else [])
    refit.sync_weights()
    assert log == [
        "engine.coresident_pause",
        "learner.coresident_refit_source_ready",
        "engine.coresident_refit_copy",
        "engine.coresident_resume",
    ]
    policy.worker_group.replies["coresident_refit_source_ready"] = ["moved"]
    with pytest.raises(RuntimeError, match="moved"):
        refit.sync_weights()


def test_coresident_wrapper_rejects_unsupported_kv_scales() -> None:
    from nemo_rl.weight_sync.megatron_weight_synchronizer import (
        MegatronWeightSynchronizer,
    )

    policy = SimpleNamespace(cfg=_cfg())
    generation = SimpleNamespace(
        cfg={"refit_transport": "ipc"}, uses_native_refit=False
    )
    sync = MegatronWeightSynchronizer(
        policy,
        generation,
        colocated=False,
        train_cluster=object(),
        inference_cluster=object(),
    )
    with pytest.raises(NotImplementedError, match="FP8 KV scales"):
        sync.sync_weights(kv_scales={"layer.0": 0.5})
    assert sync.is_stale


def test_coresident_failed_pause_does_not_mark_engine_paused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.weight_sync import megatron_weight_synchronizer as mws

    def timed_out(refs: list, *, timeout: float) -> list:
        assert timeout == 0.25
        raise mws.ray.exceptions.GetTimeoutError("engine pause did not finish")

    monkeypatch.setattr(mws.ray, "get", timed_out)
    log = []
    refit = mws.CoresidentRefit(
        policy=SimpleNamespace(cfg=_cfg()),
        generation=SimpleNamespace(worker_group=_Group(log, "engine", {})),
        refit_timeout_s=0.25,
    )
    with pytest.raises(TimeoutError, match="engine pause"):
        refit.sync_weights()
    assert refit._paused_at is None
    assert log == ["engine.coresident_pause"]


@pytest.mark.parametrize(
    "method", ["coresident_resume", "report_dp_openai_server_base_url"]
)
def test_first_refit_failure_does_not_publish_engine(single_rank_workers, method):
    from nemo_rl.weight_sync.megatron_weight_synchronizer import CoresidentRefit

    policy, generation, log = single_rank_workers
    refit = CoresidentRefit(policy, generation)
    refit.init_communicator()
    generation.worker_group.replies[method] = [TimeoutError("engine timed out")]
    with pytest.raises(TimeoutError, match="engine timed out"):
        refit.sync_weights()
    assert log[-1] == f"engine.{method}"
    assert generation.dp_openai_server_base_urls == []
    assert not refit._engine_running


def test_coresident_copy_failure_keeps_engine_paused(single_rank_workers) -> None:
    from nemo_rl.weight_sync import megatron_weight_synchronizer as mws

    policy, generation, log = single_rank_workers
    refit = mws.CoresidentRefit(policy, generation)
    refit.init_communicator()
    refit.sync_weights()
    generation.worker_group.replies["coresident_refit_copy"] = [
        ValueError("invalid weights")
    ]
    log.clear()
    with pytest.raises(ValueError, match="invalid weights"):
        refit.sync_weights()
    assert log == [
        "engine.coresident_pause",
        "learner.coresident_refit_source_ready",
        "engine.coresident_refit_copy",
    ]
    assert refit._paused_at is not None


@pytest.mark.parametrize(
    "layout", ["packed", "reversed", "padded", "separate", "strided"]
)
def test_contiguous_expert_stack_preserves_storage_or_falls_back(layout: str) -> None:
    backing = torch.arange(3 * 8 * 4, dtype=torch.float32).reshape(3, 8, 4)
    weights = list(backing.unbind())
    if layout == "reversed":
        weights.reverse()
    elif layout == "padded":
        weights = [weight[:4] for weight in weights]
    elif layout == "separate":
        weights = [weight.clone() for weight in weights]
    elif layout == "strided":
        weights = [weight.t() for weight in weights]
    stack = smp._contiguous_expert_stack(weights)
    if layout != "packed":
        assert stack is None
        return
    assert stack is not None
    assert stack.is_contiguous()
    assert stack.data_ptr() == backing.data_ptr()
    backing.add_(3)
    assert torch.equal(stack, backing)


def test_contiguous_expert_stack_accepts_offset_subsets_and_rejects_mixed_shapes() -> (
    None
):
    backing = torch.randn(6, 8, 4)
    weights = list(backing[2:5].unbind())
    stack = smp._contiguous_expert_stack(weights)
    assert stack is not None
    assert torch.equal(stack, backing[2:5])
    assert stack.data_ptr() == weights[0].data_ptr()
    assert smp._contiguous_expert_stack([weights[0], weights[1][:4]]) is None


def test_expert_parameter_order_preserves_names_objects_and_optimizer_updates() -> None:
    def make_linear() -> torch.nn.Module:
        linear = torch.nn.Module()
        for index in range(3):
            linear.register_parameter(
                f"weight{index}", torch.nn.Parameter(torch.full((16, 16), index + 0.25))
            )
            linear.register_parameter(
                f"bias{index}", torch.nn.Parameter(torch.zeros(16))
            )
        return linear

    original, packed = make_linear(), make_linear()
    identities = {name: id(parameter) for name, parameter in packed.named_parameters()}
    state = {name: tensor.clone() for name, tensor in original.state_dict().items()}
    assert smp._reverse_expert_parameter_order(packed, num_experts=3)
    assert {
        name: id(parameter) for name, parameter in packed.named_parameters()
    } == identities
    assert list(packed._parameters)[-3:] == ["weight2", "weight1", "weight0"]
    packed.load_state_dict(state, strict=True)
    optimizers = [
        torch.optim.AdamW(module.parameters(), lr=0.01, foreach=False)
        for module in (original, packed)
    ]
    for _ in range(2):
        for module, optimizer in zip((original, packed), optimizers):
            for name, parameter in module.named_parameters():
                parameter.grad = torch.full_like(parameter, int(name[-1]) + 0.5)
            optimizer.step()
        for name, tensor in original.state_dict().items():
            assert torch.equal(tensor, packed.state_dict()[name])


def test_expert_parameter_order_leaves_incompatible_parameters_untouched() -> None:
    linear = torch.nn.Module()
    linear.register_parameter("weight0", torch.nn.Parameter(torch.ones(16, 16)))
    linear.register_parameter("weight1", torch.nn.Parameter(torch.ones(16, 16).t()))
    before = list(linear._parameters.items())
    assert not smp._reverse_expert_parameter_order(linear, num_experts=2)
    assert list(linear._parameters.items()) == before


@pytest.mark.parametrize(
    "failure", [None, "export_coresident_refit_destinations", "coresident_reshard"]
)
def test_residual_transfer_precedes_publication_and_failures_keep_engine_paused(
    local_refit_rpc, failure: str | None
) -> None:
    from nemo_rl.weight_sync import megatron_weight_synchronizer as mws

    log = []

    exports = [
        smp.RefitExport(
            smp.RefitPlacement(f"gpu{rank}", rank),
            [{"rank": rank}],
            {"w": (rank,)},
            f"source{rank}",
        )
        for rank in range(4)
    ]
    residual = [
        {"owner_rank": rank, "name": "weight", "shape": (8, 4), "element_size": 4}
        for rank in (3, 2)
    ]
    learner = _Group(
        log,
        "learner",
        {
            "export_coresident_refit_handles": exports,
            "coresident_refit_source_ready": [item.fingerprint for item in exports],
        },
    )
    engine = _Group(
        log,
        "engine",
        {
            "coresident_refit_placement": [
                smp.RefitPlacement("gpu3", 1),
                smp.RefitPlacement("gpu2", 0),
            ],
            "open_coresident_refit_handles": [
                smp.RefitBinding([item], {"tensors_bound": 0}) for item in residual
            ],
            "export_coresident_refit_destinations": [
                smp.RefitDestination(f"gpu{rank}", f"receive{rank}", {})
                for rank in (3, 2)
            ],
            "coresident_refit_copy": [
                {"seconds": 0.0, "bytes": 0.0, "bytes_shared": 16.0}
            ]
            * 2,
        },
    )
    generation = SimpleNamespace(
        worker_group=engine, refresh_dp_openai_server_base_urls=lambda **kwargs: None
    )
    refit = mws.CoresidentRefit(
        SimpleNamespace(cfg=_cfg(), worker_group=learner), generation
    )
    refit.init_communicator()
    opened = next(
        kwargs
        for method, kwargs in engine.calls
        if method == "open_coresident_refit_handles"
    )
    assert opened == {
        "local_placement": [exports[3].placement, exports[2].placement],
        "local_handles": [exports[3].handles, exports[2].handles],
        "common_kwargs": {
            "source_metadata": [item for export in exports for item in export.metadata],
            "engine_to_learner": {1: 3, 0: 2},
            "learner_world_size": 4,
            "share": False,
        },
    }
    assert "learner.init_coresident_reshard" in log
    log.clear()
    if failure is not None:
        (engine if failure.startswith("export_") else learner).replies[failure] = [
            ValueError("transfer failed")
        ]
        with pytest.raises(ValueError, match="transfer failed"):
            refit.sync_weights()
        assert "engine.coresident_refit_copy" not in log
        assert "engine.coresident_resume" not in log
        assert refit._paused_at is not None
    else:
        stats = refit.sync_weights()
        assert log == [
            "engine.coresident_pause",
            "learner.coresident_refit_source_ready",
            "engine.export_coresident_refit_destinations",
            "learner.coresident_reshard",
            "engine.coresident_refit_copy",
            "engine.coresident_resume",
        ]
        assert stats["coresident_copy_bytes_total"] == 256
        assert stats["coresident_shared_bytes_total"] == 32
        replacements = [
            smp.RefitDestination(f"gpu{rank}", f"moved{rank}", {"weight": ("new", ())})
            for rank in (3, 2)
        ]
        engine.replies["export_coresident_refit_destinations"] = replacements
        refit.sync_weights()
        forwarded = [
            kwargs["destinations"]
            for method, kwargs in learner.calls
            if method == "coresident_reshard"
        ]
        assert forwarded[-1] == {item.device_id: item for item in replacements}
        refit.shutdown()
        assert log[-1] == "learner.release_coresident_receives"
        refit.shutdown()
        assert log.count("learner.release_coresident_receives") == 1
        with pytest.raises(RuntimeError, match="closed"):
            refit.sync_weights()


@pytest.mark.parametrize(
    "world,learner,engine,expected",
    [
        (4, (2, 1, 1, 1, 1), (4, 1, 1, 1, 1), [0, 2, 1, 3]),
        (8, (2, 4, 1, 1, 1), (4, 8, 1, 1, 1), [0, 4, 1, 5, 2, 6, 3, 7]),
        (8, (2, 4, 1, 1, 1), (4, 4, 1, 1, 1), list(range(8))),
        (8, (4, 1, 1, 1, 1), (2, 1, 1, 1, 1), list(range(8))),
        (8, (2, 1, 1, 2, 1), (2, 1, 1, 1, 1), list(range(8))),
    ],
)
def test_joint_rank_placement_and_residual_fallback(
    world, learner, engine, expected
) -> None:
    assert (
        smp.engine_rank_gpus(
            world,
            learner=smp.ParallelismLayout(*learner),
            engine=smp.ParallelismLayout(*engine),
        )
        == expected
    )


@pytest.mark.parametrize("cp", [1, 2, 4])
def test_joint_placement_matches_native_dense_and_expert_group_coordinates(
    cp: int,
) -> None:
    from megatron.core.parallel_state import RankGenerator

    world = 32
    learner = smp.ParallelismLayout(2, 4, 1, 1, cp)
    engine = smp.ParallelismLayout(4, 8, 1, 1, 1)
    placement = smp.engine_rank_gpus(world, learner=learner, engine=engine)
    assert sorted(placement) == list(range(world))

    def groups(layout):
        dense = RankGenerator(
            tp=layout.tp,
            ep=1,
            dp=world // (layout.tp * layout.cp),
            pp=1,
            cp=layout.cp,
            order="tp-cp-ep-dp-pp",
        )
        expert = RankGenerator(
            tp=layout.etp,
            ep=layout.ep,
            dp=world // (layout.etp * layout.ep),
            pp=1,
            cp=1,
            order="tp-cp-ep-dp-pp",
        )
        return dense.get_ranks("tp"), expert.get_ranks("tp"), expert.get_ranks("ep")

    source_groups, destination_groups = groups(learner), groups(engine)
    ratios = (
        engine.tp // learner.tp,
        engine.etp // learner.etp,
        engine.ep // learner.ep,
    )
    for rank, gpu in enumerate(placement):
        for source, destination, ratio in zip(
            source_groups, destination_groups, ratios
        ):
            source_index = next(group.index(gpu) for group in source if gpu in group)
            destination_index = next(
                group.index(rank) for group in destination if rank in group
            )
            assert source_index == destination_index // ratio


def test_uncapped_mps_configuration() -> None:
    config = _cfg(**{"generation.colocated.engine_sm_fraction": 0.0})
    smp.validate(config, env=MPS)
    assert smp.actor_env(config, "learner", env=MPS) == {smp.MPS_PCT_ENV: "100", **MPS}
    assert smp.actor_env(config, "engine", env=MPS) == MPS
    config["generation"]["colocated"]["sm_partition_backend"] = "mps_static"
    with pytest.raises(ValueError, match="positive engine_sm_fraction"):
        smp.validate(config, env=MPS)


@pytest.mark.parametrize("registered", [False, True])
def test_uncapped_client_requires_daemon_registration(
    monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None)
    )

    def control(command, **kwargs):
        request = kwargs["input"].strip()
        answer = (
            "123\n"
            if request == "get_server_list"
            else f"{os.getpid() if registered else os.getpid() + 1}\n"
        )
        return SimpleNamespace(stdout=answer)

    monkeypatch.setattr(subprocess, "run", control)
    if registered:
        assert smp._check_uncapped_client("learner", "100", seen=152, full=152, rank=0)[
            "mps_client_registered"
        ]
    else:
        with pytest.raises(RuntimeError, match="not registered"):
            smp._check_uncapped_client("learner", "100", seen=152, full=152, rank=0)


@pytest.mark.parametrize(
    "fraction,target", [(0.0, 152), (0.21, 120), (0.42, 88), (0.63, 56), (0.999, 1)]
)
@pytest.mark.parametrize("backend", [None, "mps_gemm"])
def test_gemm_budget_preserves_full_context(
    monkeypatch: pytest.MonkeyPatch, fraction: float, target: int, backend: str | None
) -> None:
    config = _cfg(
        **{
            "generation.colocated.sm_partition_backend": "mps_gemm",
            "generation.colocated.engine_sm_fraction": fraction,
        }
    )
    if backend is None:
        del config["generation"]["colocated"]["sm_partition_backend"]
    smp.validate(config, env=MPS)
    assert smp.actor_env(config, "learner", env=MPS) == {**MPS, smp.MPS_PCT_ENV: "100"}
    assert smp.actor_env(config, "engine", env=MPS) == MPS
    monkeypatch.setenv(smp.MPS_PCT_ENV, "100")
    monkeypatch.setenv("NVTE_EXT_MARGIN_SM", "sentinel")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=152),
    )
    monkeypatch.setattr(smp, "_device_sm_count", lambda: 152)
    registrations = []

    def registered(*args: object, **kwargs: object) -> dict:
        registrations.append(args)
        return {"mps_client_registered": True}

    monkeypatch.setattr(smp, "_check_uncapped_client", registered)
    info = smp.check_worker(config, "learner")
    assert registrations
    assert info["te_grouped_gemm_target_sms"] == target
    assert os.environ["NVTE_EXT_MARGIN_SM"] == str(152 - target)
    monkeypatch.delenv("NVTE_EXT_MARGIN_SM")
    smp.check_worker(config, "engine")
    assert "NVTE_EXT_MARGIN_SM" not in os.environ


@pytest.mark.parametrize("key", ["NVTE_EXT_MARGIN_SM", "CUDA_MPS_SM_PARTITION"])
@pytest.mark.parametrize("source", ["process", "worker"])
def test_gemm_budget_rejects_conflicting_environment(key: str, source: str) -> None:
    config = _cfg(**{"generation.colocated.sm_partition_backend": "mps_gemm"})
    env = dict(MPS)
    if source == "process":
        env[key] = "0"
    else:
        config["megatron_cfg"]["env_vars"] = {key: "0"}
    with pytest.raises(ValueError, match=key):
        smp.validate(config, env=env)


@pytest.mark.parametrize(
    "learner_indices, engine_indices, expected",
    [
        (None, [1, 3], [False, True, False, True]),
        ([6, 2, 5], [2, 4], [False, True, False]),
        ([3, 1], [1, 3], [True, True]),
        ([6, 2], [0, 1], [False, False]),
        ([7, 3], None, [True, True]),
        (None, None, [True, True, True, True]),
    ],
)
def test_gemm_hint_follows_partial_gpu_overlap(
    monkeypatch: pytest.MonkeyPatch,
    learner_indices: list[int] | None,
    engine_indices: list[int] | None,
    expected: list[bool],
) -> None:
    config = _cfg(
        **{
            "generation.colocated.sm_partition_backend": "mps_gemm",
            "generation.colocated.learner_gpu_indices": learner_indices,
            "generation.colocated.engine_gpu_indices": engine_indices,
        }
    )
    monkeypatch.setenv(smp.MPS_PCT_ENV, "100")
    monkeypatch.setenv("NVTE_EXT_MARGIN_SM", "sentinel")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=152),
    )
    monkeypatch.setattr(smp, "_device_sm_count", lambda: 152)
    monkeypatch.setattr(smp, "_check_uncapped_client", lambda *a, **k: {})
    for rank, shared in enumerate(expected):
        # All workers may have the same node-local CUDA index. Logical ranks,
        # rather than that index or raw placement-bundle IDs, decide overlap.
        info = smp.check_worker(config, "learner", rank=rank)
        assert info["shares_engine_gpu"] is shared
        assert info["te_grouped_gemm_target_sms"] == (88 if shared else 152)
        assert os.environ["NVTE_EXT_MARGIN_SM"] == ("64" if shared else "0")
        assert os.environ[smp.MPS_PCT_ENV] == "100"


@pytest.mark.parametrize("rank", [-1, 2])
@pytest.mark.parametrize("engine_indices", [None, [2, 4]])
def test_gemm_hint_rejects_rank_outside_learner_view(
    rank: int, engine_indices: list[int] | None
) -> None:
    config = _cfg(
        **{
            "generation.colocated.learner_gpu_indices": [6, 2],
            "generation.colocated.engine_gpu_indices": engine_indices,
        }
    )
    with pytest.raises(ValueError, match="outside its GPU subset"):
        smp._learner_shares_engine_gpu(config, rank=rank)


def test_uncapped_client_retries_transient_control_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None)
    )
    monkeypatch.setattr(smp.time, "sleep", lambda _: None)
    calls = []

    def control(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs["input"])
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(
            stdout="123\n"
            if kwargs["input"] == "get_server_list\n"
            else f"{os.getpid()}\n"
        )

    monkeypatch.setattr(subprocess, "run", control)
    assert smp._check_uncapped_client("learner", "100", seen=152, full=152, rank=0)[
        "mps_client_registered"
    ]
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("learner", "engine"),
    [
        (None, list(range(8, 16))),
        (list(reversed(range(16))), list(range(8))),
        (list(range(8, 16)), list(range(8, 12))),
    ],
)
def test_validate_role_gpu_subsets(learner, engine):
    cfg = _cfg(
        **{
            "megatron_cfg.context_parallel_size": 1,
            "generation.colocated.learner_gpu_indices": learner,
            "generation.colocated.engine_gpu_indices": engine,
        }
    )
    smp.validate_gpu_subsets(cfg, 16)


@pytest.mark.parametrize(
    ("learner", "engine", "match"),
    [
        (None, [], "nonempty"),
        (None, [0, 0], "unique"),
        (None, [16], "in range"),
        ([0, 1, 2, 3], [4, 5, 6, 7], "also be a learner"),
        (None, [0, 1], "expert"),
        ([0, 1, 2, 3], None, "also be a learner"),
    ],
)
def test_validate_role_gpu_subsets_fails_before_placement(learner, engine, match):
    cfg = _cfg(
        **{
            "megatron_cfg.context_parallel_size": 1,
            "generation.colocated.learner_gpu_indices": learner,
            "generation.colocated.engine_gpu_indices": engine,
        }
    )
    with pytest.raises(ValueError, match=match):
        smp.validate_gpu_subsets(cfg, 16)

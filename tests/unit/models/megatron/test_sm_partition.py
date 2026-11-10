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

import time
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron import sm_partition as smp

MPS = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps"}


def _cfg(**over):
    c = {
        "megatron_cfg": {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
                         "expert_model_parallel_size": 4, "expert_tensor_parallel_size": 1},
        "generation": {"backend": "megatron", "refit_transport": "ipc",
                       "colocated": {"enabled": False, "engine_sm_fraction": 0.42},
                       "mcore_generation_config": {"tensor_model_parallel_size": 1, "expert_model_parallel_size": 4}},
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
    smp.validate(_cfg(**{"generation.mcore_generation_config.expert_model_parallel_size": 8}), env=MPS)
    smp.validate(_cfg(**{"megatron_cfg.tensor_model_parallel_size": 2,
                         "generation.mcore_generation_config.tensor_model_parallel_size": 2}), env=MPS)


@pytest.mark.parametrize(
    "over, env, match",
    [
        ({"generation.colocated.engine_sm_fraction": 0.0}, MPS, r"\(0, 1\)"),
        ({"generation.colocated.engine_sm_fraction": 1.0}, MPS, r"\(0, 1\)"),
        ({"generation.backend": "vllm"}, MPS, "Megatron generation backend"),
        ({"generation.colocated.enabled": True}, MPS, "colocated.enabled=false"),
        ({"generation.refit_transport": None}, MPS, "refit_transport=ipc"),
        ({}, {}, "MPS control daemon"),
        ({"generation.mcore_generation_config.expert_model_parallel_size": 6}, MPS, "not built yet"),
        ({"generation.mcore_generation_config.tensor_model_parallel_size": 2}, MPS, "not built yet"),
        ({"megatron_cfg.tensor_model_parallel_size": 2,  # an EP multiple is placed for TP = 1 only
          "generation.mcore_generation_config.tensor_model_parallel_size": 2,
          "generation.mcore_generation_config.expert_model_parallel_size": 8}, MPS, "TP = ETP = 1"),
        ({"megatron_cfg.pipeline_model_parallel_size": 2,
          "generation.mcore_generation_config.pipeline_model_parallel_size": 2}, MPS, "pipeline parallelism"),
    ],
)
def test_validate_rejects(over, env, match):
    with pytest.raises(ValueError, match=match):
        smp.validate(_cfg(**over), env=env)


def test_validate_rejects_unsupported_features():
    with pytest.raises(ValueError, match="cluster.segment_size is not built yet"):
        smp.validate(_cfg(), env=MPS, unsupported={"cluster.segment_size": True, "PPO": False})
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


def test_check_limit():
    assert smp._check_limit("engine", "42", 62, 152, 0)["sms_seen"] == 62
    with pytest.raises(RuntimeError, match="not running under its MPS limit"):  # no daemon: the full GPU is seen
        smp._check_limit("engine", "42", 152, 152, 0)
    with pytest.raises(RuntimeError, match="not running under its MPS limit"):  # launched without its percentage
        smp._check_limit("learner", None, 88, 152, 0)


@pytest.mark.parametrize(
    "role,pct,accepted",
    [
        ("engine", None, True),
        ("engine", "100", True),
        ("engine", "42", False),
        ("engine", "58", False),
        ("learner", "58", True),
        ("learner", "42", False),
        ("learner", "100", False),
        ("learner", None, False),
    ],
)
def test_check_worker_verifies_role_percentage(
    monkeypatch: pytest.MonkeyPatch, role: smp.Role, pct: str | None, accepted: bool
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=152 if role == "engine" else 88),
    )
    monkeypatch.setattr(smp, "_device_sm_count", lambda: 152)
    if pct is None:
        monkeypatch.delenv(smp.MPS_PCT_ENV, raising=False)
    else:
        monkeypatch.setenv(smp.MPS_PCT_ENV, pct)
    if accepted:
        info = smp.check_worker(_cfg(), role)
        assert info["mps_pct"] == pct
        assert info["sms_seen"] == (152 if role == "engine" else 88)
    else:
        with pytest.raises(RuntimeError, match="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
            smp.check_worker(_cfg(), role)


# ---- same-GPU CUDA IPC refit, two processes (GPU) ----


def _learner(q_to_engine, q_from_engine):
    torch.cuda.set_device(0)
    g = torch.Generator().manual_seed(0)
    buf = torch.randn(1 << 20, generator=g).to("cuda", torch.bfloat16)  # one contiguous buffer, as DDP's
    named = [("a.weight", buf[: 512 * 1024].view(512, 1024)), ("b.bias", buf[512 * 1024 : 512 * 1024 + 1024]),
             ("e.weight0", buf[600000 : 600000 + 64 * 64].view(64, 64))]
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
        dst = {"a.weight": torch.zeros(512, 1024, device="cuda", dtype=torch.bfloat16),
               "b.bias": torch.zeros(1024, device="cuda", dtype=torch.bfloat16),
               "e.weight0": torch.zeros(64, 64, device="cuda", dtype=torch.float16)}  # engine-held dtype: copy_ casts
        res = {}
        q_to_learner.put("update"); expect = q_from_learner.get()
        res["bytes"] = smp.copy_refit(src, dst.items())["bytes"]
        res["bitwise"] = all(torch.equal(dst[n].float().cpu(), expect[n].to(dst[n].dtype).float()) for n in dst)
        q_to_learner.put("update"); expect = q_from_learner.get()  # negative control: no copy after this update
        res["stale_detected"] = not all(torch.equal(dst[n].float().cpu(), expect[n].to(dst[n].dtype).float())
                                        for n in dst)
        try:
            smp.copy_refit(src, [("a.weight", dst["a.weight"])])
        except ValueError as ex:
            res["name_mismatch"] = "differ by name" in str(ex)
        try:
            smp.copy_refit(src, [("a.weight", dst["a.weight"].view(1024, 512))] +
                           [(n, t) for n, t in dst.items() if n != "a.weight"])
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
    pl = ctx.Process(target=_learner, args=(l2e, e2l)); pl.start()
    pe = ctx.Process(target=_engine, args=(l2e, e2l, out)); pe.start()
    try:
        res = out.get(timeout=300)
        pe.join(30); pl.join(30)
        assert pe.exitcode == 0 and pl.exitcode == 0, (pe.exitcode, pl.exitcode)
        assert "error" not in res, res
        assert res["bitwise"] and res["stale_detected"] and res["name_mismatch"] and res["shape_mismatch"], res
        assert res["bytes"] == 512 * 1024 * 2 + 1024 * 2 + 64 * 64 * 2
    finally:
        for proc in (pe, pl):
            if proc.is_alive():
                proc.terminate()
            proc.join(10)


def test_refit_fails_fast():
    ray = pytest.importorskip("ray")
    from nemo_rl.weight_sync.megatron_weight_synchronizer import _get_fail_fast

    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, _temp_dir="/tmp/ray_smp_test")
    try:
        @ray.remote
        def slow():
            time.sleep(60)

        @ray.remote
        def boom():
            raise KeyError("engine failed mid-refit")

        t0 = time.perf_counter()
        with pytest.raises(Exception, match="engine failed mid-refit"):
            _get_fail_fast([slow.remote(), boom.remote()])
        assert time.perf_counter() - t0 < 5
    finally:
        ray.shutdown()


@pytest.mark.parametrize("count", [1, 37])
def test_copy_refit_matches_materialized_sources(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    src = {str(i): torch.arange(24, dtype=torch.float32).reshape(4, 6) + i for i in range(count)}
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
    assert stats == {"bytes": float((count + 1) * 24 * 4 + 256 * 1024 * 4 + 7 * 4), "bytes_shared": 12.0}


def test_copy_refit_rejects_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
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


def test_partition_backend_swap_point(tmp_path):
    assert smp.partition_backend(_cfg()) == "mps_percentage"
    with pytest.raises(ValueError, match="must be one of"):
        smp.validate(_cfg(**{"generation.colocated.sm_partition_backend": "green"}), env=MPS)
    static = _cfg(**{"generation.colocated.sm_partition_backend": "mps_static"})
    with pytest.raises(ValueError, match="NRL_MPS_SM_PARTITION_MAP"):
        smp.validate(static, env=MPS)
    m = tmp_path / "map.json"
    m.write_text('{"0": {"engine": 7, "learner": 8, "engine_sms": 64, "learner_sms": 88}}')
    env = {**MPS, smp.MPS_PARTITION_MAP_ENV: str(m)}
    smp.validate(static, env=env)
    assert smp.actor_env(static, "engine", env=env) == {**env}  # no percentage under the static backend
    import os

    os.environ[smp.MPS_PARTITION_MAP_ENV] = str(m)
    try:
        assert smp.select_static_partition(static, "learner", 0) == "8"
        assert os.environ["NRL_MPS_SM_PARTITION_SMS"] == "88"
        assert smp.select_static_partition(_cfg(), "learner", 0) is None
    finally:
        for k in (smp.MPS_PARTITION_MAP_ENV, "CUDA_MPS_SM_PARTITION", "NRL_MPS_SM_PARTITION_SMS"):
            os.environ.pop(k, None)


def test_bind_shared_then_copy_whatever_is_not_aliased(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU stand-in for the IPC-mapped source. Bound tensors alias the source and are skipped by the per-step copy;
    tensors that were never bound (dtype, derived view, strided source) or that the engine re-homed after binding
    (MCore concatenates expert weights into its own buffer on its first inference forward) are copied."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
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
    src["w"].add_(1.0)  # the learner's in-place optimizer write is what the engine reads
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
    assert smp.bind_shared({"w": torch.randn(4, 4).t()}, [("w", strided)], {id(strided)})["tensors_bound"] == 0


@pytest.mark.parametrize("world, l_ep, e_ep", [(16, 4, 8), (16, 8, 8), (8, 2, 8), (4, 2, 4), (64, 8, 32)])
def test_engine_rank_gpus_put_experts_on_their_learner(world, l_ep, e_ep):
    gpus = smp.engine_rank_gpus(world, l_ep, e_ep)
    assert sorted(gpus) == list(range(world))
    experts = 128
    for r, g in enumerate(gpus):
        e_block = range((r % e_ep) * experts // e_ep, (r % e_ep + 1) * experts // e_ep)
        l_block = range((g % l_ep) * experts // l_ep, (g % l_ep + 1) * experts // l_ep)
        assert set(e_block) <= set(l_block), (r, g)
    if l_ep == e_ep:
        assert gpus == list(range(world))


def test_map_sources_by_resolved_name_and_rank_coordinates():
    """Engine EP rank 1 of 8 (global experts 16..31) next to learner EP rank 0 of 4 (global experts 0..31): each engine
    tensor maps to the learner tensor with the same Megatron resolved name (global expert index)."""
    src = {f"decoder.layers.1.mlp.experts.linear_fc1.weight{i}": torch.full((1,), float(i)) for i in range(32)}
    src["decoder.layers.0.self_attention.linear_qkv.weight"] = torch.zeros(1)
    dst = {f"decoder.layers.1.mlp.experts.linear_fc1.weight{i}": f"decoder.layers.1.mlp.experts.linear_fc1.weight{16 + i}"
           for i in range(16)}
    dst["decoder.layers.0.self_attention.linear_qkv.weight"] = "decoder.layers.0.self_attention.linear_qkv.weight"
    coords = {"tp": (0, 1), "pp": (0, 1)}
    got = smp.map_sources(src, coords, dst, coords)
    assert [int(got[f"decoder.layers.1.mlp.experts.linear_fc1.weight{i}"].item()) for i in range(16)] == list(
        range(16, 32)
    )
    # Negative: the engine rank sits next to another TP rank's learner (same names and shapes, wrong shard).
    with pytest.raises(ValueError, match="engine ranks are misplaced"):
        smp.map_sources(src, {"tp": (1, 2), "pp": (0, 1)}, dst, {"tp": (0, 2), "pp": (0, 1)})
    # Negative: an engine expert that this GPU's learner rank does not hold.
    far = {**dst, "decoder.layers.1.mlp.experts.linear_fc1.weight0": "decoder.layers.1.mlp.experts.linear_fc1.weight40"}
    with pytest.raises(ValueError, match="not on this GPU's learner rank"):
        smp.map_sources(src, coords, far, coords)


class _Group:
    """A worker group stand-in that records the calls the synchronizer makes, in order."""

    def __init__(self, log, role, replies):
        self.log, self.role, self.replies = log, role, replies

    def run_all_workers_single_data(self, method, **kwargs):
        self.log.append(f"{self.role}.{method}")
        return [self.replies.get(method)]


@pytest.mark.parametrize("weights", ["copy", "shared"])
def test_coresident_refit_publishes_only_in_sync_weights(monkeypatch, weights):
    """Shared weights: the engine pauses before the optimizer write and stays paused until sync_weights (the loop's
    refit, after its parameter all-gather) copies the rest, bumps the epoch and resumes it. Copy: the write needs no
    pause. Either way the new version is published in exactly one place."""
    from types import SimpleNamespace

    from nemo_rl.weight_sync import megatron_weight_synchronizer as mws

    monkeypatch.setattr(mws, "_get_fail_fast", lambda refs, **kwargs: list(refs))
    log = []
    policy = SimpleNamespace(
        cfg=_cfg(**{"generation.colocated.engine_weights": weights}),
        worker_group=_Group(log, "learner", {"export_coresident_refit_handles": ("gpu0", {}, "fp", {}),
                                             "coresident_refit_source_ready": "fp"}),
    )
    generation = SimpleNamespace(worker_group=_Group(log, "engine", {
        "open_coresident_refit_handles": {}, "coresident_refit_copy": {"seconds": 0.0, "bytes": 1.0, "bytes_shared": 0.0}}))
    refit = mws.CoresidentRefit(policy=policy, generation=generation)
    refit.init_communicator()
    refit.before_learner_write()  # before the engine runs: nothing to pause
    refit.sync_weights()  # first refit starts the engine
    log.clear()
    refit.before_learner_write()
    assert log == (["engine.coresident_pause"] if weights == "shared" else [])
    refit.sync_weights()
    assert log == ["engine.coresident_pause", "learner.coresident_refit_source_ready", "engine.coresident_refit_copy",
                   "engine.coresident_resume"]
    policy.worker_group.replies["coresident_refit_source_ready"] = "moved"
    with pytest.raises(RuntimeError, match="moved"):
        refit.sync_weights()


def test_coresident_wrapper_rejects_unsupported_kv_scales() -> None:
    from nemo_rl.weight_sync.megatron_weight_synchronizer import MegatronWeightSynchronizer

    policy = SimpleNamespace(cfg=_cfg())
    generation = SimpleNamespace(cfg={"refit_transport": "ipc"}, uses_native_refit=False)
    sync = MegatronWeightSynchronizer(
        policy, generation, colocated=False, train_cluster=object(), inference_cluster=object()
    )
    with pytest.raises(NotImplementedError, match="FP8 KV scales"):
        sync.sync_weights(kv_scales={"layer.0": 0.5})
    assert sync.is_stale


def test_coresident_worker_wait_has_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_rl.weight_sync import megatron_weight_synchronizer as mws

    ticks = iter([0.0, 0.1, 1.1])
    waits = []

    def wait(refs: list, *, num_returns: int, timeout: float) -> tuple[list, list]:
        waits.append(timeout)
        return [], refs

    monkeypatch.setattr(mws, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    monkeypatch.setattr(mws, "ray", SimpleNamespace(wait=wait, get=lambda refs: []))
    with pytest.raises(TimeoutError, match="1 worker call"):
        mws._get_fail_fast(["stalled"], timeout_s=1.0)
    assert waits == [pytest.approx(0.9)]


def test_coresident_failed_pause_does_not_mark_engine_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_rl.weight_sync import megatron_weight_synchronizer as mws

    def timed_out(refs: list, *, timeout_s: float) -> list:
        assert timeout_s == 0.25
        raise TimeoutError("engine pause did not finish")

    monkeypatch.setattr(mws, "_get_fail_fast", timed_out)
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


def test_coresident_copy_failure_keeps_engine_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_rl.weight_sync import megatron_weight_synchronizer as mws

    def get(refs: list, **kwargs) -> list:
        for ref in refs:
            if isinstance(ref, Exception):
                raise ref
        return list(refs)

    monkeypatch.setattr(mws, "_get_fail_fast", get)
    log = []
    policy = SimpleNamespace(
        cfg=_cfg(), worker_group=_Group(log, "learner", {
            "export_coresident_refit_handles": ("gpu0", {}, "fp", {}),
            "coresident_refit_source_ready": "fp",
        })
    )
    generation = SimpleNamespace(worker_group=_Group(log, "engine", {
        "open_coresident_refit_handles": {},
        "coresident_refit_copy": {"seconds": 0.0, "bytes": 1.0, "bytes_shared": 0.0},
    }))
    refit = mws.CoresidentRefit(policy, generation)
    refit.init_communicator()
    refit.sync_weights()
    generation.worker_group.replies["coresident_refit_copy"] = ValueError("invalid weights")
    log.clear()
    with pytest.raises(ValueError, match="invalid weights"):
        refit.sync_weights()
    assert log == ["engine.coresident_pause", "learner.coresident_refit_source_ready", "engine.coresident_refit_copy"]
    assert refit._paused_at is not None


@pytest.mark.parametrize("layout", ["packed", "reversed", "padded", "separate", "strided"])
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


def test_contiguous_expert_stack_accepts_offset_subsets_and_rejects_mixed_shapes() -> None:
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
            linear.register_parameter(f"weight{index}", torch.nn.Parameter(torch.full((16, 16), index + 0.25)))
            linear.register_parameter(f"bias{index}", torch.nn.Parameter(torch.zeros(16)))
        return linear

    original, packed = make_linear(), make_linear()
    identities = {name: id(parameter) for name, parameter in packed.named_parameters()}
    state = {name: tensor.clone() for name, tensor in original.state_dict().items()}
    assert smp._reverse_expert_parameter_order(packed, num_experts=3)
    assert {name: id(parameter) for name, parameter in packed.named_parameters()} == identities
    assert list(packed._parameters)[-3:] == ["weight2", "weight1", "weight0"]
    packed.load_state_dict(state, strict=True)
    optimizers = [torch.optim.AdamW(module.parameters(), lr=0.01, foreach=False) for module in (original, packed)]
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

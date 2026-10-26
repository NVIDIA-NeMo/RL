"""VllmGeneration under prefill/decode disaggregation: layout, dispatch, refit, worker lookup."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import nemo_rl.models.generation.vllm.vllm_generation as vg
from nemo_rl.models.generation.vllm.config import (
    resolve_vllm_disagg_frontend_config,
    resolve_vllm_disagg_layout,
    resolve_vllm_role_overrides,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker import _resolve_disagg_engine
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    _gym_prefix_from_messages,
    _is_prefill_leg,
    _supplied_prompt_token_ids,
)
from nemo_rl.models.generation.openai_server_utils import encode_token_ids_b64


def _config(disaggregation=None, *, colocated=False, **vllm_overrides):
    vllm_cfg = {
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.6,
        "max_model_len": 4096,
        "precision": "bfloat16",
        "async_engine": True,
        "expose_http_server": True,
        "skip_tokenizer_init": False,
        "kv_cache_dtype": "auto",
    }
    vllm_cfg.update(vllm_overrides)
    cfg = {
        "backend": "vllm",
        "model_name": "test/model",
        "max_new_tokens": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "val_temperature": 1.0,
        "val_top_p": 1.0,
        "val_top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "_pad_token_id": 0,
        "colocated": {
            "enabled": colocated,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "vllm_cfg": vllm_cfg,
        "vllm_kwargs": {},
    }
    if disaggregation is not None:
        cfg["disaggregation"] = disaggregation
    return resolve_vllm_role_overrides(cfg)


def _bare_generation(cfg, *, world_size=8):
    """A VllmGeneration with the layout derived but no Ray behind it."""
    gen = VllmGeneration.__new__(VllmGeneration)
    gen.cfg = cfg
    gen.tp_size = cfg["vllm_cfg"]["tensor_parallel_size"]
    gen.pp_size = cfg["vllm_cfg"]["pipeline_parallel_size"]
    gen.model_parallel_size = gen.tp_size * gen.pp_size
    gen._disagg_cfg = resolve_vllm_disagg_layout(cfg)
    gen._disagg_frontend_cfg = (
        resolve_vllm_disagg_frontend_config(cfg) if gen._disagg_cfg.enabled else None
    )
    gen._disagg_actors = []
    gen._disagg_frontend_urls = []
    gen._stamped_engine_roles = {}
    gen._engine_env_vars = {}
    gen._engine_roles, gen._engine_widths, gen.num_replicas = vg.plan_engines(
        world_size,
        layout=gen._disagg_cfg,
        aggregated_width=gen.model_parallel_size,
        prefill_width=gen._role_width("prefill"),
        decode_width=gen._role_width("decode"),
        backend_cfg_key="vllm_cfg",
    )
    gen.num_engines = len(gen._engine_widths)
    gen.dp_size = gen.num_replicas
    gen._engine_owner_indices = vg.engine_owner_indices(gen._engine_widths)
    gen._replica_owner_indices = gen._engine_owner_indices[
        :: gen.num_engines // gen.num_replicas
    ]
    gen.widest_engine_gpus = max(gen._engine_widths)
    gen.worker_group = MagicMock()
    gen.worker_group.dp_size = gen.dp_size
    gen.worker_group.workers = [object() for _ in range(world_size)]
    gen.worker_group.run_single_worker_single_data = MagicMock(
        side_effect=lambda **kw: kw
    )
    gen.worker_group.run_all_workers_single_data = MagicMock(return_value=["grid"])
    gen.fleet_monitor = None
    gen.fleet_selector = None
    gen._refit_membership = None
    return gen


_DISAGG = {"enabled": True, "num_prefill_engines": 1, "num_decode_engines": 2}


def test_layout_asymmetric_tp_derives_replicas_and_owners():
    gen = _bare_generation(
        _config(
            _DISAGG,
            prefill_engine={"tensor_parallel_size": 4},
            decode_engine={"tensor_parallel_size": 2},
        )
    )
    assert gen._engine_roles == ["prefill", "decode", "decode"]
    assert gen._engine_widths == [4, 2, 2]
    assert gen.num_replicas == 1 and gen.dp_size == 1
    assert gen._engine_owner_indices == [0, 4, 6]
    assert gen._replica_owner_indices == [0]
    assert gen.widest_engine_gpus == 4


def test_fan_out_addresses_every_engine_owner_under_disaggregation():
    gen = _bare_generation(_config(_DISAGG), world_size=6)
    futures = gen._fan_out("reset_prefix_cache_async")
    assert [f["worker_idx"] for f in futures] == [0, 2, 4]
    assert all(f["method_name"] == "reset_prefix_cache_async" for f in futures)
    gen.worker_group.run_all_workers_single_data.assert_not_called()


def test_fan_out_keeps_the_grid_dispatch_without_disaggregation():
    gen = _bare_generation(_config(), world_size=8)
    assert gen._fan_out("reset_prefix_cache_async") == ["grid"]
    gen.worker_group.run_all_workers_single_data.assert_called_once()


def test_run_on_engines_spreads_per_engine_kwargs_in_engine_order():
    gen = _bare_generation(_config(_DISAGG), world_size=6)
    futures = gen._run_on_engines(
        "init_collective_async", per_engine={"rank_prefix": [0, 2, 4]}, ip="h"
    )
    assert [(f["worker_idx"], f["rank_prefix"], f["ip"]) for f in futures] == [
        (0, 0, "h"),
        (2, 2, "h"),
        (4, 4, "h"),
    ]


def test_direct_generate_paths_are_rejected_under_disaggregation():
    gen = _bare_generation(_config(_DISAGG), world_size=6)
    with pytest.raises(RuntimeError, match="only wired for the HTTP/NeMo-Gym"):
        gen.generate(MagicMock())
    with pytest.raises(RuntimeError, match="only wired for the HTTP/NeMo-Gym"):
        gen.generate_text(MagicMock())


def test_refit_leaders_are_every_engine_owner_under_disaggregation():
    gen = _bare_generation(_config(_DISAGG), world_size=6)
    leaders = gen._refit_leaders_by_shard()
    assert [(e, o) for e, o, _ in leaders] == [(0, 0), (1, 2), (2, 4)]
    assert [w for _, _, w in leaders] == [
        gen.worker_group.workers[i] for i in (0, 2, 4)
    ]


def test_shard_restart_machinery_is_refused_under_disaggregation():
    gen = _bare_generation(_config(_DISAGG), world_size=6)
    gen._defer_model_load = False
    with pytest.raises(NotImplementedError, match="not supported under PD"):
        gen.restart_shard(0)
    with pytest.raises(NotImplementedError, match="not supported under PD"):
        gen.set_refit_membership(MagicMock())
    with pytest.raises(NotImplementedError, match="not supported under PD"):
        gen.attach_fleet_health(MagicMock(shard_count=1), MagicMock())


def test_config_with_engine_roles_stamps_identity_and_node_local_ports(monkeypatch):
    cfg = _config(
        _DISAGG,
        prefill_engine={"tensor_parallel_size": 4},
        decode_engine={"tensor_parallel_size": 2},
    )
    gen = _bare_generation(cfg, world_size=16)
    assert gen.num_replicas == 2
    # Two 8-GPU nodes, each holding one whole replica.
    bundles = [
        (0, [0, 1, 2, 3]),
        (0, [4, 5]),
        (0, [6, 7]),
        (1, [0, 1, 2, 3]),
        (1, [4, 5]),
        (1, [6, 7]),
    ]
    cluster = MagicMock()
    cluster.get_placement_groups.return_value = [MagicMock(), MagicMock()]
    monkeypatch.setattr(
        vg, "_node_of_placement_groups", lambda pgs: {0: "node-a", 1: "node-b"}
    )

    stamped = gen._config_with_engine_roles(cfg, bundles, cluster)
    roles = stamped["vllm_cfg"]["_engine_roles"]
    assert set(roles) == {"0:0,1,2,3", "0:4,5", "0:6,7", "1:0,1,2,3", "1:4,5", "1:6,7"}
    p0, d0, d1 = roles["0:0,1,2,3"], roles["0:4,5"], roles["0:6,7"]
    assert (p0["role"], p0["ordinal"], p0["replica_idx"]) == ("prefill", 0, 0)
    assert (d0["role"], d0["ordinal"]) == ("decode", 0)
    assert (d1["role"], d1["ordinal"]) == ("decode", 1)
    assert (
        p0["engine_id"] == "replica0-prefill0" and d1["engine_id"] == "replica0-decode1"
    )
    # Node-local slots restart per node and give distinct ports on a node.
    assert [roles[k]["env"]["VLLM_PORT"] for k in ("0:0,1,2,3", "0:4,5", "0:6,7")] == [
        7000,
        7100,
        7200,
    ]
    assert roles["1:0,1,2,3"]["env"]["VLLM_PORT"] == 7000
    assert [
        roles[k]["env"]["VLLM_NIXL_SIDE_CHANNEL_PORT"]
        for k in ("0:0,1,2,3", "0:4,5", "0:6,7")
    ] == [8556, 8557, 8558]
    assert p0["side_channel_host_env"] == "VLLM_NIXL_SIDE_CHANNEL_HOST"
    assert "kv_events_port" not in p0
    # Seeds are unique across the fleet.
    assert len({e["seed"] for e in roles.values()}) == 6
    assert gen._engine_roles_map()[0] is p0
    # The caller's config is not mutated.
    assert "_engine_roles" not in cfg["vllm_cfg"]


def test_kv_cache_aware_routing_gives_prefill_engines_an_events_port(monkeypatch):
    cfg = _config(
        {"enabled": True, "frontend_tokenize": True},
        disagg_frontend={"prefill_router": "kv_cache_aware"},
    )
    gen = _bare_generation(cfg, world_size=4)
    monkeypatch.setattr(vg, "_node_of_placement_groups", lambda pgs: {0: "n"})
    stamped = gen._config_with_engine_roles(
        cfg, [(0, [0, 1]), (0, [2, 3])], MagicMock()
    )
    roles = stamped["vllm_cfg"]["_engine_roles"]
    assert roles["0:0,1"]["kv_events_port"] == 8572
    assert roles["0:0,1"]["kv_events_topic"] == vg._KV_EVENTS_TOPIC
    assert "kv_events_port" not in roles["0:2,3"]
    assert gen._kv_cache_aware_routing()


def test_placement_groups_are_sized_by_the_widest_engine_and_refuse_colocation():
    cluster = MagicMock()
    cluster.num_gpus_per_node = 4
    cfg = _config(
        _DISAGG,
        prefill_engine={"tensor_parallel_size": 8},
        decode_engine={"tensor_parallel_size": 2},
    )
    VllmGeneration.init_cluster_placement_groups(cluster, cfg)
    cluster._init_placement_groups.assert_called_once_with(
        strategy="PACK", use_unified_pg=True
    )
    with pytest.raises(ValueError, match="non-colocated"):
        VllmGeneration.init_cluster_placement_groups(
            cluster, _config(_DISAGG, colocated=True)
        )


def test_resume_after_refit_resets_every_engine_and_the_frontends_first(monkeypatch):
    gen = _bare_generation(_config(_DISAGG), world_size=6)
    order = []
    gen.worker_group.run_single_worker_single_data = MagicMock(
        side_effect=lambda **kw: order.append(
            ("engine", kw["method_name"], kw["worker_idx"])
        )
        or f"ref-{kw['worker_idx']}"
    )
    gen.worker_group.workers = [
        SimpleNamespace(
            resume_generation_async=SimpleNamespace(
                remote=lambda i=i: order.append(("resume", i)) or True
            )
        )
        for i in range(6)
    ]
    gen._disagg_actors = [
        SimpleNamespace(
            invalidate_conversation_cache=SimpleNamespace(
                remote=lambda: order.append(("frontend", "clear")) or 0
            )
        )
    ]
    monkeypatch.setattr(
        vg.ray, "get", lambda refs, **kw: refs if isinstance(refs, list) else refs
    )
    monkeypatch.setattr(vg.ray, "wait", lambda refs, num_returns, timeout: (refs, []))

    assert gen.resume_generation_after_refit(timeout_s=1.0) is True
    resets = [o for o in order if o[0] == "engine"]
    assert resets == [("engine", "reset_prefix_cache_async", i) for i in (0, 2, 4)]
    assert ("frontend", "clear") in order
    resumes = [o for o in order if o[0] == "resume"]
    # Decode engines (owners 2, 4) resume before the prefill engine (owner 0).
    assert [i for _, i in resumes] == [2, 4, 0]
    assert order.index(("frontend", "clear")) < order.index(resumes[0])


def test_pause_for_refit_pauses_prefill_engines_before_decode_engines(monkeypatch):
    gen = _bare_generation(_config(_DISAGG), world_size=6)
    order = []
    gen.worker_group.workers = [
        SimpleNamespace(
            pause_generation_async=SimpleNamespace(
                remote=lambda clear_cache, i=i: order.append(("pause", i, clear_cache))
                or f"ref-{i}"
            )
        )
        for i in range(6)
    ]
    gets = []

    def fake_get(refs, **kw):
        gets.append(list(refs))
        return [True for _ in refs]

    monkeypatch.setattr(vg.ray, "get", fake_get)

    assert gen.pause_generation_for_refit(clear_cache=False, timeout_s=5.0) is True
    assert [(i, c) for _, i, c in order] == [(0, False), (2, False), (4, False)]
    # Two phases: the prefill owner's pause is awaited before any decode pause
    # is even issued, so the decode engines still drain the handed-off reads.
    assert gets == [["ref-0"], ["ref-2", "ref-4"]]


def test_pause_for_refit_times_out_per_phase(monkeypatch):
    gen = _bare_generation(_config(_DISAGG), world_size=6)
    gen.worker_group.workers = [
        SimpleNamespace(
            pause_generation_async=SimpleNamespace(
                remote=lambda clear_cache, i=i: f"ref-{i}"
            )
        )
        for i in range(6)
    ]

    def fake_get(refs, **kw):
        raise vg.ray.exceptions.GetTimeoutError("slow")

    monkeypatch.setattr(vg.ray, "get", fake_get)
    with pytest.raises(vg.RefitAborted, match="prefill engines"):
        gen.pause_generation_for_refit(clear_cache=True, timeout_s=0.1)


def test_invalidate_kv_cache_also_clears_the_frontends(monkeypatch):
    gen = _bare_generation(_config(_DISAGG), world_size=6)
    cleared = []
    gen._disagg_actors = [
        SimpleNamespace(
            invalidate_conversation_cache=SimpleNamespace(
                remote=lambda: cleared.append(1) or 0
            )
        )
    ]
    monkeypatch.setattr(
        vg.ray,
        "get",
        lambda refs, **kw: [True] * len(refs) if isinstance(refs, list) else refs,
    )
    assert gen.invalidate_kv_cache() is True
    assert cleared == [1]


# -------------------------------------------------------------------------- #
#  Worker-side lookups
# -------------------------------------------------------------------------- #


def test_worker_role_lookup_matches_the_driver_key():
    roles = {"3:4,5": {"role": "decode", "ordinal": 1}}
    cfg = {"vllm_cfg": {"_engine_roles": roles}}
    assert _resolve_disagg_engine(cfg, 3, [4, 5]) is roles["3:4,5"]
    # Non-owner workers and uniform runs carry no role.
    assert _resolve_disagg_engine(cfg, 3, None) == {}
    assert _resolve_disagg_engine({"vllm_cfg": {}}, 3, [4, 5]) == {}
    with pytest.raises(RuntimeError, match="No engine role entry"):
        _resolve_disagg_engine(cfg, 3, [0, 1])
    with pytest.raises(RuntimeError, match="bundle_pg_idx"):
        _resolve_disagg_engine(cfg, None, [4, 5])


def test_server_side_leg_helpers():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "a",
            "prompt_token_ids": [1, 2],
            "generation_token_ids": [3],
        },
        {"role": "user", "content": "q2"},
    ]
    assert _gym_prefix_from_messages(messages) == [1, 2, 3]
    assert _gym_prefix_from_messages([{"role": "user", "content": "q"}]) is None

    assert _supplied_prompt_token_ids(
        SimpleNamespace(prompt_token_ids_b64=encode_token_ids_b64([7, 8]))
    ) == [7, 8]
    assert _supplied_prompt_token_ids(
        SimpleNamespace(prompt_token_ids=[1], prompt_token_ids_b64=None)
    ) == [1]
    assert _supplied_prompt_token_ids(
        SimpleNamespace(kv_transfer_params={"prompt_token_ids": [2]})
    ) == [2]
    assert (
        _supplied_prompt_token_ids(
            SimpleNamespace(kv_transfer_params={"do_remote_prefill": True})
        )
        is None
    )

    assert _is_prefill_leg(
        SimpleNamespace(kv_transfer_params={"do_remote_decode": True})
    )
    assert not _is_prefill_leg(
        SimpleNamespace(kv_transfer_params={"do_remote_prefill": True})
    )
    assert not _is_prefill_leg(SimpleNamespace(kv_transfer_params=None))


def test_disagg_engine_env_defaults_keep_nccl_plugin_out_of_nixl_engines():
    assert vg.disagg_engine_env_defaults(_config(_DISAGG)) == {
        "NCCL_NET_PLUGIN": "none"
    }
    assert vg.disagg_engine_env_defaults(_config()) == {}
    assert (
        vg.disagg_engine_env_defaults(
            _config(
                _DISAGG,
                kv_transfer={
                    "kv_connector": "NixlConnector",
                    "kv_buffer_device": "cpu",
                },
            )
        )
        == {}
    )
    assert (
        vg.disagg_engine_env_defaults(
            _config(_DISAGG, kv_transfer={"kv_connector": "MooncakeConnector"})
        )
        == {}
    )

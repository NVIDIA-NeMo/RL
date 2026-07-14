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

from types import SimpleNamespace

import torch

from nemo_rl.models.policy.workers import megatron_remote_sparse_refit

MegatronRemoteSparseRefit = megatron_remote_sparse_refit.MegatronRemoteSparseRefit


_DELTA_CONFIG = {"encoding": "overwrite", "sparse_bucket_size_bytes": 1024}
_XOR_CONFIG = {**_DELTA_CONFIG, "encoding": "xor"}


class _AutoMapping:
    is_expert = False
    is_adapter = False
    is_grouped_export = False
    ep_size = 1
    ep_rank = 0
    tp_size = 1
    tp_rank = 0

    def __init__(self, hf_param, *, parallelism="column", permute_dims=None):
        self.hf_param = hf_param
        self.parallelism = parallelism
        self.permute_dims = permute_dims

    def _detect_parallelism_type(self, _module):
        return self.parallelism


class _ColumnMapping(_AutoMapping):
    pass


class _DirectMapping(_AutoMapping):
    pass


class _GatedMapping(_AutoMapping):
    def __init__(self, *, gate, up):
        super().__init__({"gate": gate, "up": up})


class _RowMapping(_AutoMapping):
    pass


class _ReplicatedMapping(_AutoMapping):
    pass


def _install_mapping_types(monkeypatch, remote_refit_type):
    monkeypatch.setattr(
        remote_refit_type,
        "_bridge_mapping_types",
        staticmethod(
            lambda: (
                _AutoMapping,
                {
                    _ColumnMapping: megatron_remote_sparse_refit._COLUMN,
                    _DirectMapping: megatron_remote_sparse_refit._DIRECT,
                    _GatedMapping: megatron_remote_sparse_refit._GATED,
                    _ReplicatedMapping: megatron_remote_sparse_refit._REPLICATED,
                    _RowMapping: megatron_remote_sparse_refit._ROW,
                },
            )
        ),
    )
    monkeypatch.setattr(
        remote_refit_type,
        "_bridge_exports_are_identity",
        lambda _self: True,
    )


def _worker(tasks=(), *, fp8_cfg=None, export=None):
    if export is None:

        def export(*, conversion_tasks=None):
            return iter(())

    return SimpleNamespace(
        cfg={},
        fp8_cfg=fp8_cfg,
        model=object(),
        megatron_bridge=SimpleNamespace(
            get_conversion_tasks=lambda _models: list(tasks)
        ),
        _iter_params_with_optional_kv_scales=export,
    )


def test_remote_sparse_stream_drains_cuda_before_return(monkeypatch):
    def export(*, conversion_tasks=None):
        assert conversion_tasks == []
        return iter(())

    worker = _worker(export=export)
    remote_refit = MegatronRemoteSparseRefit(worker, _DELTA_CONFIG)
    result = {"payloads": 1, "changed_elements": 2, "total_elements": 3}
    events = []

    def stream(*_args, **_kwargs):
        events.append("stream")
        return result

    monkeypatch.setattr(
        "nemo_rl.models.policy.workers.megatron_remote_sparse_refit."
        "stream_sparse_delta_payloads_via_zmq",
        stream,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    actual = remote_refit.stream(
        "zmq",
        ["tcp://receiver:5555"],
        transfer_id="transfer",
        api_key_env_var=None,
        timeout_s=1.0,
        shard_rank=0,
        shard_count=1,
    )

    assert actual is result
    assert events == ["stream", "sync"]


def test_remote_sparse_stream_combines_local_and_misc_paths(monkeypatch):
    remote_refit = MegatronRemoteSparseRefit(_worker(), _DELTA_CONFIG)
    remote_refit._policy_tracker = object()
    remote_refit._local_tensors = [("local", torch.ones(1))]
    remote_refit._misc_conversion_tasks = []
    monkeypatch.setattr(remote_refit, "_changed_misc_tasks", lambda: ([], 5, 6))
    calls = []

    def stream(*_args, **kwargs):
        calls.append(kwargs)
        if kwargs["partition"] == "names":
            return {"payloads": 4, "changed_elements": 5, "total_elements": 6}
        return {"payloads": 1, "changed_elements": 2, "total_elements": 3}

    monkeypatch.setattr(
        "nemo_rl.models.policy.workers.megatron_remote_sparse_refit."
        "stream_sparse_delta_payloads_via_zmq",
        stream,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = remote_refit.stream(
        "zmq",
        ["tcp://receiver:5555"],
        transfer_id="transfer",
        api_key_env_var=None,
        timeout_s=1.0,
        shard_rank=2,
        shard_count=4,
    )

    assert result == {"payloads": 5, "changed_elements": 7, "total_elements": 9}
    assert {call["transfer_id"] for call in calls} == {
        "transfer-local",
        "transfer-misc",
    }
    local = next(call for call in calls if call["transfer_id"].endswith("-local"))
    assert local["partition"] == "none"


def test_remote_sparse_globalizes_expert_name():
    task = SimpleNamespace(
        mapping=SimpleNamespace(is_expert=True, ep_size=4, ep_rank=2),
        megatron_module=SimpleNamespace(config=SimpleNamespace(num_moe_experts=16)),
    )

    assert (
        MegatronRemoteSparseRefit._canonical_hf_name(
            task, "model.layers.0.mlp.experts.1.up_proj.weight"
        )
        == "model.layers.0.mlp.experts.9.up_proj.weight"
    )
    assert (
        MegatronRemoteSparseRefit._canonical_hf_name(
            task, "model.layers.0.mlp.experts.9.up_proj.weight"
        )
        == "model.layers.0.mlp.experts.9.up_proj.weight"
    )


def test_remote_sparse_projects_bridge_affine_mappings(monkeypatch):
    _install_mapping_types(monkeypatch, MegatronRemoteSparseRefit)

    def task(mapping, module_name, tensor, global_name):
        module = type(module_name, (torch.nn.Module,), {})()
        return SimpleNamespace(
            mapping=mapping,
            megatron_module=module,
            param_weight=tensor,
            global_param_name=global_name,
        )

    column = task(
        _AutoMapping("backbone.layers.0.mixer.D"),
        "TEColumnParallelLinear",
        torch.arange(8).view(4, 2),
        "decoder.layers.0.mixer.D",
    )
    column.mapping.tp_size = 2
    column.mapping.tp_rank = 1
    row = task(
        _AutoMapping("backbone.layers.0.mixer.o_proj.weight", parallelism="row"),
        "TERowParallelLinear",
        torch.arange(8).view(2, 4),
        "decoder.layers.0.self_attention.linear_proj.weight",
    )
    row.mapping.tp_size = 2
    row.mapping.tp_rank = 1
    replicated = task(
        _AutoMapping("backbone.layers.0.norm.weight", parallelism="replicated"),
        "TENorm",
        torch.arange(4),
        "decoder.layers.0.input_layernorm.weight",
    )
    gated = task(
        _GatedMapping(
            gate="model.mlp.gate_proj.weight",
            up="model.mlp.up_proj.weight",
        ),
        "TEColumnParallelLinear",
        torch.arange(16).view(8, 2),
        "mlp.linear_fc1.weight",
    )

    column_projection = MegatronRemoteSparseRefit._task_local_tensors(
        column, megatron_remote_sparse_refit._COLUMN
    )[0][2]
    row_projection = MegatronRemoteSparseRefit._task_local_tensors(
        row, megatron_remote_sparse_refit._ROW
    )[0][2]
    replicated_projection = MegatronRemoteSparseRefit._task_local_tensors(
        replicated, megatron_remote_sparse_refit._REPLICATED
    )[0][2]
    gated_projections = MegatronRemoteSparseRefit._task_local_tensors(
        gated, megatron_remote_sparse_refit._GATED
    )

    assert column_projection.name == "backbone.layers.0.mixer.D"
    assert column_projection.global_shape == (8, 2)
    assert column_projection.offsets == (4, 0)
    assert row_projection.name == "backbone.layers.0.mixer.o_proj.weight"
    assert row_projection.global_shape == (2, 8)
    assert row_projection.offsets == (0, 4)
    assert replicated_projection.name == "backbone.layers.0.norm.weight"
    assert replicated_projection.global_shape == (4,)
    assert replicated_projection.offsets == (0,)
    assert [projection.name for _, _, projection in gated_projections] == [
        "model.mlp.gate_proj.weight",
        "model.mlp.up_proj.weight",
    ]
    assert [tuple(tensor.shape) for _, tensor, _ in gated_projections] == [
        (4, 2),
        (4, 2),
    ]


def test_remote_sparse_uses_local_baseline_to_gate_transformed_tasks(
    monkeypatch,
):
    class _TransformedMapping(_AutoMapping):
        pass

    tensor = torch.tensor([1.0, 2.0, 3.0])
    task = SimpleNamespace(
        mapping=_TransformedMapping("model.q_proj.weight"),
        megatron_module=torch.nn.Linear(3, 1),
        param_weight=tensor,
        global_param_name="decoder.linear_qkv.weight",
    )

    _install_mapping_types(monkeypatch, MegatronRemoteSparseRefit)
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    remote_refit = MegatronRemoteSparseRefit(_worker([task]), _DELTA_CONFIG)
    remote_refit._prepare_paths()

    assert remote_refit._misc_conversion_tasks == [task]
    assert remote_refit._misc_local_tensors == [("0:decoder.linear_qkv.weight", tensor)]
    assert remote_refit._policy_tracker is not None
    remote_refit._policy_tracker.snapshot_baseline(remote_refit._misc_local_tensors)

    tensor[1] = 5
    changed_tasks, changed, total = remote_refit._changed_misc_tasks()
    assert changed_tasks == [task]
    assert (changed, total) == (1, 3)

    remote_refit._policy_tracker.on_sync_succeeded()
    assert remote_refit._changed_misc_tasks() == ([], 0, 3)


def test_remote_sparse_uses_xor_only_for_direct_bitwise_path(monkeypatch):
    class _TransformedMapping(_AutoMapping):
        pass

    direct = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    transformed = torch.tensor([1.0, 2.0])
    tasks = [
        SimpleNamespace(
            mapping=_AutoMapping("model.q_proj.weight"),
            megatron_module=torch.nn.Linear(2, 2),
            param_weight=direct,
            global_param_name="decoder.linear_qkv.weight",
        ),
        SimpleNamespace(
            mapping=_TransformedMapping("backbone.layers.0.mixer.A_log"),
            megatron_module=torch.nn.Linear(2, 1),
            param_weight=transformed,
            global_param_name="decoder.mixer.A_log",
        ),
    ]

    _install_mapping_types(monkeypatch, MegatronRemoteSparseRefit)
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    remote_refit = MegatronRemoteSparseRefit(_worker(tasks), _XOR_CONFIG)
    remote_refit._prepare_paths()

    assert remote_refit._policy_tracker is not None
    assert remote_refit._policy_tracker.encoding == "xor"
    assert remote_refit._tracker.encoding == "overwrite"

    remote_refit._policy_tracker.snapshot_baseline(remote_refit._local_tensors)
    direct[0, 0] = 5
    direct_metadata = remote_refit._policy_tracker.prepare_sparse_delta_payload(
        remote_refit._local_tensors
    )[0][2]
    assert [item["operation"] for item in direct_metadata] == ["xor"]

    residual = [("backbone.layers.0.mixer.A_log", transformed)]
    remote_refit._tracker.snapshot_baseline(residual)
    transformed[0] = 3
    residual_metadata = remote_refit._tracker.prepare_sparse_delta_payload(residual)[0][
        2
    ]
    assert [item["operation"] for item in residual_metadata] == ["overwrite"]
    assert remote_refit.refit_info() == {
        "backbone.layers.0.mixer.A_log": ((2,), torch.float32),
        "model.q_proj.weight": ((2, 2), torch.float32),
    }


def test_remote_sparse_preserves_bridge_task_dependencies(monkeypatch):
    grouped = SimpleNamespace(is_grouped_export=True, group_key="experts")
    tasks = [
        SimpleNamespace(mapping=grouped),
        SimpleNamespace(mapping=grouped),
        SimpleNamespace(
            mapping=SimpleNamespace(is_grouped_export=False, group_key="other")
        ),
    ]
    remote_refit = MegatronRemoteSparseRefit(object(), _DELTA_CONFIG)
    remote_refit._misc_conversion_tasks = tasks
    remote_refit._filter_misc_tasks = True
    monkeypatch.setattr(remote_refit, "_all_reduce_max", lambda _flags: [1, 0, 0])

    assert remote_refit._changed_misc_tasks()[0] == tasks[:2]

    remote_refit._filter_misc_tasks = False
    assert remote_refit._changed_misc_tasks()[0] == tasks


def test_remote_sparse_balances_tasks_across_equivalent_replicas(monkeypatch):
    from megatron.core import parallel_state

    from nemo_rl.utils.weight_transfer_remote_sparse import sparse_name_shard

    ranks = {"dp": 0, "expert_dp": 0}
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_rank",
        lambda *, with_context_parallel: ranks["dp"],
    )
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_world_size",
        lambda *, with_context_parallel: 2,
    )
    monkeypatch.setattr(
        parallel_state,
        "get_expert_data_parallel_rank",
        lambda: ranks["expert_dp"],
    )
    monkeypatch.setattr(
        parallel_state,
        "get_expert_data_parallel_world_size",
        lambda: 4,
    )

    dense = SimpleNamespace(
        global_param_name="decoder.layers.0.input_layernorm.weight",
        mapping=SimpleNamespace(is_expert=False, tp_rank=0, tp_size=2),
    )
    dense_owners = []
    for dp_rank in range(2):
        ranks["dp"] = dp_rank
        for tp_rank in range(2):
            dense.mapping.tp_rank = tp_rank
            if MegatronRemoteSparseRefit._owns_policy_local_task(
                dense, replicated=True
            ):
                dense_owners.append(dp_rank * 2 + tp_rank)
    assert dense_owners == [sparse_name_shard(dense.global_param_name, 4)]

    owner_counts = [0] * 4
    for expert_id in range(64):
        name = f"decoder.layers.0.mlp.experts.local_experts.{expert_id}.weight"
        expert = SimpleNamespace(
            global_param_name=name,
            mapping=SimpleNamespace(is_expert=True),
        )
        owners = []
        for expert_dp_rank in range(4):
            ranks["expert_dp"] = expert_dp_rank
            if MegatronRemoteSparseRefit._owns_policy_local_task(expert):
                owners.append(expert_dp_rank)
                owner_counts[expert_dp_rank] += 1
        assert owners == [sparse_name_shard(name, 4)]
    assert min(owner_counts) > 0


def test_remote_sparse_fp8_policy_keeps_full_export_path(monkeypatch):
    task = object()

    def export(*, conversion_tasks):
        assert conversion_tasks == [task]
        return iter(())

    remote_refit = MegatronRemoteSparseRefit(
        _worker([task], fp8_cfg={"fp8_param": True}, export=export), _DELTA_CONFIG
    )
    snapshots = []
    monkeypatch.setattr(
        "nemo_rl.models.policy.workers.megatron_remote_sparse_refit."
        "init_sparse_delta_baseline_from_iterator",
        lambda iterator, **_kwargs: snapshots.append(list(iterator)),
    )

    remote_refit.initialize_baseline(shard_rank=0, shard_count=1, transport="zmq")

    assert snapshots == [[]]
    assert remote_refit._misc_conversion_tasks == [task]
    assert remote_refit._policy_tracker is None

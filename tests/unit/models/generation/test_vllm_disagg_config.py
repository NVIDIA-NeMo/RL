"""The vLLM backend's prefill/decode disaggregation config: schemas, resolvers, guards."""

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.interfaces import DisaggConfig
from nemo_rl.models.generation.vllm.config import (
    VLLM_ENGINE_ARG_KEYS,
    VllmDisaggFrontendConfig,
    VllmKvTransferConfig,
    hybrid_ssm_requires_homogeneous_tp,
    resolve_vllm_connector_profile,
    resolve_vllm_disagg_frontend_config,
    resolve_vllm_kv_transfer_config,
    resolve_vllm_role_overrides,
    validate_vllm_disaggregation,
    vllm_kv_transfer_config_for_role,
    vllm_replica_width,
    vllm_role_engine_config,
)
from nemo_rl.models.generation.vllm.kv_connector_profiles import (
    CONNECTOR_PROFILES,
    connector_profile,
)


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
        "colocated": {
            "enabled": colocated,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "vllm_cfg": vllm_cfg,
    }
    if disaggregation is not None:
        cfg["disaggregation"] = disaggregation
    return cfg


def _disagg(**vllm_overrides):
    return resolve_vllm_role_overrides(
        _config(disaggregation={"enabled": True}, **vllm_overrides)
    )


# -------------------------------------------------------------------------- #
#  Schemas
# -------------------------------------------------------------------------- #


def test_the_engine_arg_whitelist_is_exactly_the_schema():
    assert VLLM_ENGINE_ARG_KEYS == {
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "vllm_kwargs",
    }


def test_frontend_defaults_come_from_the_schema_not_the_call_sites():
    frontend = resolve_vllm_disagg_frontend_config(_disagg())
    assert frontend == VllmDisaggFrontendConfig()
    assert frontend.prefill_router == "conversation"
    assert frontend.decode_router == "load_balancing"
    assert frontend.bidirectional_kv_xfer is False
    assert frontend.decode_tokids_b64 is False
    assert frontend.decode_strip_message_history is False


def test_a_layout_key_under_disagg_frontend_is_rejected():
    with pytest.raises(ValueError, match="belongs in generation.disaggregation"):
        resolve_vllm_disagg_frontend_config(
            _disagg(disagg_frontend={"num_frontend_workers": 2})
        )


def test_an_unknown_frontend_key_is_rejected():
    with pytest.raises(ValueError):
        resolve_vllm_disagg_frontend_config(_disagg(disagg_frontend={"max_retries": 3}))


def test_a_plausible_but_wrong_router_is_rejected():
    with pytest.raises(ValueError):
        resolve_vllm_disagg_frontend_config(
            _disagg(disagg_frontend={"prefill_router": "sticky"})
        )


def test_kv_transfer_defaults_to_nixl_and_forwards_only_set_extras():
    kv = resolve_vllm_kv_transfer_config(_disagg())
    assert kv == VllmKvTransferConfig()
    assert kv.kv_connector == "NixlConnector"
    assert kv.kv_connector_extra_config.model_dump(exclude_none=True) == {}


def test_bidirectional_cannot_be_set_on_the_connector_block():
    with pytest.raises(ValueError, match="written by NeMo RL"):
        resolve_vllm_kv_transfer_config(
            _disagg(
                kv_transfer={
                    "kv_connector_extra_config": {"bidirectional_kv_xfer": True}
                }
            )
        )


def test_connector_profiles_are_looked_up_by_connector_or_explicit_name():
    assert (
        connector_profile("NixlConnector", None) is CONNECTOR_PROFILES["NixlConnector"]
    )
    assert (
        connector_profile("my.module.Connector", "NixlConnector").name
        == "NixlConnector"
    )
    with pytest.raises(ValueError, match="No connector profile"):
        connector_profile("my.module.Connector", None)
    assert resolve_vllm_connector_profile(_disagg()).supports_bidirectional


# -------------------------------------------------------------------------- #
#  Role resolution
# -------------------------------------------------------------------------- #


def test_role_overrides_resolve_per_role_tp_and_kwargs():
    cfg = _disagg(
        prefill_engine={"tensor_parallel_size": 4, "vllm_kwargs": {"max_num_seqs": 8}}
    )
    assert vllm_role_engine_config(cfg, "prefill")["tensor_parallel_size"] == 4
    assert vllm_role_engine_config(cfg, "decode")["tensor_parallel_size"] == 2
    assert vllm_role_engine_config(cfg, "prefill")["vllm_kwargs"] == {"max_num_seqs": 8}
    assert vllm_role_engine_config(cfg, "decode")["vllm_kwargs"] == {}
    assert vllm_replica_width(cfg) == 6


def test_role_resolution_is_a_no_op_without_disaggregation():
    cfg = _config(prefill_engine={"precision": "fp8"})
    assert resolve_vllm_role_overrides(cfg) is cfg
    assert vllm_replica_width(cfg) == 2


# -------------------------------------------------------------------------- #
#  Per-role KVTransferConfig
# -------------------------------------------------------------------------- #


def test_kv_transfer_config_follows_the_role_and_the_frontend_flag():
    cfg = _disagg(
        disagg_frontend={"bidirectional_kv_xfer": True},
        kv_transfer={
            "kv_connector_extra_config": {"backends": ["UCX"], "kv_lease_duration": 600}
        },
    )
    prefill = vllm_kv_transfer_config_for_role(cfg, "prefill", engine_id="r0-p0")
    decode = vllm_kv_transfer_config_for_role(cfg, "decode", engine_id="r0-d0")
    assert prefill["kv_role"] == "kv_producer"
    assert decode["kv_role"] == "kv_consumer"
    assert prefill["engine_id"] == "r0-p0"
    assert prefill["kv_connector_extra_config"] == {
        "backends": ["UCX"],
        "kv_lease_duration": 600,
        "bidirectional_kv_xfer": True,
    }
    assert "kv_connector_module_path" not in prefill


def test_bidirectional_is_absent_from_the_connector_config_when_off():
    extra = vllm_kv_transfer_config_for_role(_disagg(), "decode", engine_id="x")[
        "kv_connector_extra_config"
    ]
    assert "bidirectional_kv_xfer" not in extra


# -------------------------------------------------------------------------- #
#  Setup-time guards
# -------------------------------------------------------------------------- #


def test_validation_is_silent_without_disaggregation():
    validate_vllm_disaggregation(_config(colocated=True, pipeline_parallel_size=2))


def test_validation_accepts_a_dense_layout_with_ep_one():
    validate_vllm_disaggregation(_disagg(expert_parallel_size=1))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(colocated=True), "non-colocated"),
        (dict(expose_http_server=False), "expose_http_server"),
        (dict(pipeline_parallel_size=2), "pipeline_parallel_size=1"),
        (dict(expert_parallel_size=4), "exceeds its tensor_parallel_size"),
        (
            dict(disagg_frontend={"prefill_router": "kv_cache_aware"}),
            "frontend_tokenize",
        ),
        (
            dict(
                disagg_frontend={"bidirectional_kv_xfer": True},
                kv_transfer={"kv_connector": "MooncakeConnector"},
            ),
            "not supported by the MooncakeConnector",
        ),
    ],
)
def test_validation_refuses_configs_that_would_run_wrong(kwargs, match):
    colocated = kwargs.pop("colocated", False)
    cfg = resolve_vllm_role_overrides(
        _config(disaggregation={"enabled": True}, colocated=colocated, **kwargs)
    )
    with pytest.raises(ValueError, match=match):
        validate_vllm_disaggregation(cfg)


def test_validation_refuses_a_user_supplied_kv_transfer_config():
    cfg = _config(disaggregation={"enabled": True})
    cfg["vllm_kwargs"] = {"kv_transfer_config": {"kv_connector": "NixlConnector"}}
    with pytest.raises(ValueError, match="owned by NeMo RL"):
        validate_vllm_disaggregation(resolve_vllm_role_overrides(cfg))


def test_validation_refuses_a_non_default_refit_transport():
    cfg = _config(disaggregation={"enabled": True})
    cfg["refit_transport"] = "nixl"
    with pytest.raises(ValueError, match="refit_transport=null"):
        validate_vllm_disaggregation(resolve_vllm_role_overrides(cfg))


def test_hybrid_ssm_models_need_equal_tp_across_roles():
    mamba = SimpleNamespace(model_type="nemotron_h")
    dense = SimpleNamespace(model_type="qwen3")
    assert hybrid_ssm_requires_homogeneous_tp(mamba)
    assert not hybrid_ssm_requires_homogeneous_tp(dense)
    assert hybrid_ssm_requires_homogeneous_tp(
        SimpleNamespace(model_type="custom", hybrid_override_pattern="M*M*")
    )
    uneven = _disagg(prefill_engine={"tensor_parallel_size": 4})
    with pytest.raises(ValueError, match="Mamba/SSM layers"):
        validate_vllm_disaggregation(uneven, hf_config=mamba)
    validate_vllm_disaggregation(uneven, hf_config=dense)
    validate_vllm_disaggregation(_disagg(), hf_config=mamba)


def test_kv_cache_aware_needs_frontend_tokenize_and_is_then_accepted():
    cfg = resolve_vllm_role_overrides(
        _config(
            disaggregation={"enabled": True, "frontend_tokenize": True},
            disagg_frontend={"prefill_router": "kv_cache_aware"},
        )
    )
    validate_vllm_disaggregation(cfg)
    assert DisaggConfig.model_validate(cfg["disaggregation"]).frontend_tokenize


def test_bidirectional_kv_xfer_is_refused_with_in_flight_weight_updates():
    from nemo_rl.models.generation.vllm.config import (
        validate_vllm_disaggregation_for_refit,
    )

    cfg = _disagg(disagg_frontend={"bidirectional_kv_xfer": True})
    # Fine without in-flight updates; fine with them while bidirectional is off
    # or disaggregation itself is off.
    validate_vllm_disaggregation_for_refit(cfg, in_flight_weight_updates=False)
    validate_vllm_disaggregation_for_refit(_disagg(), in_flight_weight_updates=True)
    validate_vllm_disaggregation_for_refit(_config(), in_flight_weight_updates=True)
    with pytest.raises(ValueError, match="bidirectional_kv_xfer"):
        validate_vllm_disaggregation_for_refit(cfg, in_flight_weight_updates=True)

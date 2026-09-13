"""A failed worker wake-up must abort preparation, not become a boolean."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_rl.models.generation.vllm import vllm_generation


@pytest.mark.parametrize("async_engine", [False, True])
@pytest.mark.parametrize("tags", [None, ["weights"], ["kv_cache"]])
@pytest.mark.parametrize("failure_stage", ["dispatch", "result"])
def test_wakeup_propagates_errors(
    monkeypatch: pytest.MonkeyPatch,
    async_engine: bool,
    tags: list[str] | None,
    failure_stage: str,
) -> None:
    error = RuntimeError("worker CUDA allocation failed")
    dispatch = Mock(return_value=[object()])
    get = Mock(return_value=[None])
    (dispatch if failure_stage == "dispatch" else get).side_effect = error
    policy = SimpleNamespace(
        cfg={
            "colocated": {"enabled": True},
            "vllm_cfg": {"async_engine": async_engine},
        },
        worker_group=SimpleNamespace(run_all_workers_single_data=dispatch),
    )
    monkeypatch.setattr(vllm_generation.ray, "get", get)
    kwargs = {} if tags is None else {"tags": tags}
    with pytest.raises(RuntimeError) as caught:
        vllm_generation.VllmGeneration.prepare_for_generation(policy, **kwargs)
    assert caught.value is error
    dispatch.assert_called_once_with(
        "wake_up_async" if async_engine else "wake_up",
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        **kwargs,
    )


@pytest.mark.parametrize("results", [[None, True], [None, None], [None, False]])
def test_wakeup_checks_worker_results(
    monkeypatch: pytest.MonkeyPatch, results: list[bool | None]
) -> None:
    policy = SimpleNamespace(
        cfg={"colocated": {"enabled": True}, "vllm_cfg": {"async_engine": True}},
        worker_group=SimpleNamespace(run_all_workers_single_data=Mock(return_value=[])),
    )
    monkeypatch.setattr(vllm_generation.ray, "get", Mock(return_value=results))
    if False in results:
        with pytest.raises(RuntimeError, match="wake-up reported failure"):
            vllm_generation.VllmGeneration.prepare_for_generation(policy)
    else:
        assert vllm_generation.VllmGeneration.prepare_for_generation(policy) is True


def test_noncolocated_does_not_wake() -> None:
    dispatch = Mock()
    policy = SimpleNamespace(
        cfg={"colocated": {"enabled": False}},
        worker_group=SimpleNamespace(run_all_workers_single_data=dispatch),
    )
    assert vllm_generation.VllmGeneration.prepare_for_generation(policy) is True
    dispatch.assert_not_called()

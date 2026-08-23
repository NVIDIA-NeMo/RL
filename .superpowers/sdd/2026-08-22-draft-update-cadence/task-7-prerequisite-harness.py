"""Dependency-free behavioral gate for the Task 7 receipt/science prerequisite."""

import ast
import hashlib
import inspect
import math
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional


def _node(path: Path, name: str, *, class_name: str | None = None) -> ast.AST:
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        owner = next(
            item
            for item in body
            if isinstance(item, ast.ClassDef) and item.name == class_name
        )
        body = owner.body
    return next(
        item
        for item in body
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == name
    )


def _receipt_contract() -> None:
    path = Path("nemo_rl/weight_sync/interfaces.py")
    request_node = _node(path, "DraftApplyRequest")
    namespace = {
        "dataclass": __import__("dataclasses").dataclass,
        "hashlib": hashlib,
        "Mapping": Mapping,
        "Path": Path,
    }
    exec(compile(ast.Module([request_node], []), str(path), "exec"), namespace)
    request_cls = namespace["DraftApplyRequest"]
    with tempfile.TemporaryDirectory() as root:
        snapshot = (Path(root) / "draft.bin").resolve()
        snapshot.write_bytes(b"draft")
        request = request_cls(
            version=3,
            snapshot_path=str(snapshot),
            sha256=hashlib.sha256(b"draft").hexdigest(),
        )
        assert request.version == 3
        try:
            request_cls(
                version=True, snapshot_path=str(snapshot), sha256=request.sha256
            )
        except ValueError:
            pass
        else:
            raise AssertionError("boolean draft version was accepted")
    print("GREEN: draft apply identity is typed and digest-bound")


def _version_contract() -> None:
    path = Path("nemo_rl/experience/sync_rollout_actor.py")
    source = path.read_text()
    tree = ast.parse(source)
    nodes = [
        item
        for item in tree.body
        if (
            isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id
                in {
                    "ACCEPTED_TOKEN_COUNT_KEY",
                    "DRAFT_TOKEN_COUNT_KEY",
                    "APPLIED_DRAFT_VERSION_KEY",
                }
                for target in item.targets
            )
        )
        or (
            isinstance(item, ast.ClassDef) and item.name == "ServingDraftVersionTracker"
        )
    ]
    namespace = {"Mapping": Mapping, "Any": object, "math": math}
    exec(compile(ast.Module(nodes, []), str(path), "exec"), namespace)
    tracker = namespace["ServingDraftVersionTracker"]()
    tracker.publish(2)
    stamped = tracker.stamp(
        {
            namespace["ACCEPTED_TOKEN_COUNT_KEY"]: 6.0,
            namespace["DRAFT_TOKEN_COUNT_KEY"]: 10.0,
        },
        expected_version=2,
    )
    assert stamped[namespace["APPLIED_DRAFT_VERSION_KEY"]] == 2
    try:
        tracker.publish(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale serving version was accepted")
    print("GREEN: rollout science binds counts to a monotonic serving version")


def _stale_failure_contract() -> None:
    path = Path("nemo_rl/weight_sync/ipc_weight_synchronizer.py")
    method = _node(path, "sync_weights", class_name="IPCWeightSynchronizer")

    class _Selection:
        draft = True

    class _Ray:
        fail = False
        results: list[bool | None] = [True]

        @classmethod
        def get(cls, _value: object) -> list[bool | None]:
            if cls.fail:
                raise RuntimeError("transfer failed")
            return cls.results

    namespace = {
        "Any": Any,
        "DraftApplyRequest": object,
        "Optional": Optional,
        "Timer": object,
        "WeightSyncSelection": lambda: _Selection(),
        "nullcontext": nullcontext,
        "ray": _Ray,
    }
    exec(compile(ast.Module([method], []), str(path), "exec"), namespace)

    class _Policy:
        def offload_before_refit(self) -> None:
            pass

        def offload_after_refit(self) -> None:
            pass

        def stream_weights_via_ipc_zmq(self, **_kwargs: object) -> list[object]:
            return []

    class _Generation:
        def prepare_for_generation(self, **_kwargs: object) -> None:
            pass

        def update_weights_via_ipc_zmq(self, **_kwargs: object) -> list[object]:
            return []

    sync = type("Sync", (), {})()
    sync._policy = _Policy()
    sync._generation = _Generation()
    sync._stale = True
    sync.validate_selection = lambda _selection: None
    sync._compute_buffer_size = lambda: 1
    method_fn = namespace["sync_weights"]
    apply_receipt = {
        "successful": True,
        "version": 4,
        "snapshot_path": "/immutable/draft.bin",
        "sha256": "a" * 64,
    }
    request = type("Request", (), {"receipt": lambda _self: apply_receipt})()
    result = method_fn(sync, draft_apply_request=request)
    assert result == {
        "successful": True,
        "draft_apply_receipt": apply_receipt,
    }
    assert sync._stale is False
    _Ray.fail = True
    try:
        method_fn(sync)
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed transfer did not raise")
    assert sync._stale is True, "failed transfer left synchronizer falsely fresh"
    _Ray.fail = False
    sync._stale = False

    class _ChangedRequest:
        def receipt(self) -> Mapping[str, object]:
            raise RuntimeError("draft apply snapshot changed")

    try:
        method_fn(sync, draft_apply_request=_ChangedRequest())
    except RuntimeError:
        pass
    else:
        raise AssertionError("changed snapshot did not fail before transfer")
    assert sync._stale is True, "invalid apply request left synchronizer falsely fresh"
    _Ray.results = [None]
    try:
        method_fn(sync)
    except RuntimeError:
        pass
    else:
        raise AssertionError("receiver without explicit success was accepted")
    assert sync._stale is True
    print("GREEN: transfer failure restores stale state")


def _vllm_collective_test_double_contract() -> None:
    path = Path("tests/unit/models/generation/test_vllm_backend.py")
    tree = ast.parse(path.read_text())
    test_names = {
        "test_collective_target_only_receiver_omits_draft_then_full_sync_restores_it",
        "test_collective_mtp_selection_controls_unprefixed_drafter_update",
    }
    for test_node in (
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in test_names
    ):
        receive_node = next(
            item
            for item in test_node.body
            if isinstance(item, ast.FunctionDef) and item.name == "receive"
        )
        namespace: dict[str, object] = {}
        exec(compile(ast.Module([receive_node], []), str(path), "exec"), namespace)
        signature = inspect.signature(namespace["receive"])
        signature.bind(
            iterator=iter(()),
            group=object(),
            src=0,
            post_unpack_func=lambda _weights: None,
        )
    print("GREEN: vLLM collective test doubles accept production keywords")


def _refit_adapter_contract() -> None:
    path = Path("nemo_rl/algorithms/grpo.py")
    method = _node(path, "refit_policy_generation")

    class _Selection:
        pass

    namespace = {
        "Any": Any,
        "ColocatablePolicyInterface": object,
        "DraftApplyRequest": object,
        "GenerationInterface": object,
        "Mapping": Mapping,
        "Optional": Optional,
        "Timer": object,
        "WeightSyncSelection": _Selection,
        "nullcontext": nullcontext,
    }
    exec(compile(ast.Module([method], []), str(path), "exec"), namespace)

    class _Synchronizer:
        kwargs: dict[str, object] | None = None

        def sync_weights(self, **kwargs: object) -> Mapping[str, object]:
            self.kwargs = kwargs
            return {"successful": True, "draft_apply_receipt": {"version": 5}}

    synchronizer = _Synchronizer()
    generation = type("Generation", (), {"weight_synchronizer": synchronizer})()
    request = object()
    result = namespace["refit_policy_generation"](
        object(),
        generation,
        False,
        draft_apply_request=request,
    )
    assert result["successful"] is True
    assert synchronizer.kwargs is not None
    assert synchronizer.kwargs["draft_apply_request"] is request
    print("GREEN: refit adapter preserves typed apply receipt payload")


if __name__ == "__main__":
    try:
        _receipt_contract()
        _version_contract()
        _stale_failure_contract()
        _vllm_collective_test_double_contract()
        _refit_adapter_contract()
    except (StopIteration, KeyError) as error:
        raise AssertionError(
            "RED: receipt/science producer contract is absent"
        ) from error

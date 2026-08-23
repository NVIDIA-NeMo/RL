"""Dependency-free RED/GREEN harness for Task 6 producer selection."""

import ast
import copy
import subprocess
import typing
from pathlib import Path


class Selection:
    def __init__(self, draft: bool = True) -> None:
        self.draft = draft


def _method(source: str, *, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    class_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        copy.deepcopy(node)
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _compile_policy_method(source: str) -> typing.Callable[..., object]:
    method = _method(
        source,
        class_name="Policy",
        method_name="broadcast_weights_for_collective",
    )
    namespace: dict[str, object] = {
        "Any": typing.Any,
        "Optional": typing.Optional,
        "WeightSyncSelection": Selection,
        "ray": type("Ray", (), {"ObjectRef": object}),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "lm_policy.py", "exec"),
        namespace,
    )
    return typing.cast(typing.Callable[..., object], namespace[method.name])


def _run_policy_red_and_green() -> None:
    base_source = subprocess.check_output(
        ["git", "show", "156f9905c:nemo_rl/models/policy/lm_policy.py"],
        text=True,
    )
    base_method = _compile_policy_method(base_source)
    policy = type(
        "Policy",
        (),
        {
            "worker_group": type(
                "WorkerGroup",
                (),
                {"run_all_workers_single_data": lambda _self, *_args, **_kwargs: []},
            )(),
        },
    )()
    try:
        base_method(policy, selection=Selection(draft=False))
    except TypeError:
        print("RED: base producer rejects component selection")
    else:  # pragma: no cover
        raise AssertionError("base producer unexpectedly accepts selection")

    calls: list[tuple[str, dict[str, object]]] = []
    policy.worker_group = type(
        "WorkerGroup",
        (),
        {
            "run_all_workers_single_data": lambda _self, name, **kwargs: (
                calls.append((name, kwargs)) or ["future"]
            )
        },
    )()
    source = Path("nemo_rl/models/policy/lm_policy.py").read_text()
    method = _compile_policy_method(source)
    assert method(policy, selection=Selection()) == ["future"]
    assert calls == [
        (
            "broadcast_weights_for_collective",
            {"kv_scales": None, "buffer_size_bytes": None, "num_buffers": None},
        )
    ]
    print("GREEN: selection accepted and default call shape preserved")


def _run_megatron_green() -> None:
    source = Path("nemo_rl/models/policy/workers/megatron_policy_worker.py").read_text()
    methods = {
        name: _method(
            source,
            class_name="MegatronPolicyWorkerImpl",
            method_name=name,
        )
        for name in ("stream_weights_via_ipc_zmq", "broadcast_weights_for_collective")
    }
    for method in methods.values():
        method.decorator_list = []
    methods["stream_weights_via_ipc_zmq"].body = [
        node
        for node in methods["stream_weights_via_ipc_zmq"].body
        if not isinstance(node, ast.ImportFrom)
    ]
    payloads: dict[str, list[list[tuple[str, bytes]]]] = {
        "ipc": [],
        "collective": [],
    }
    namespace: dict[str, object] = {
        "Optional": typing.Optional,
        "WeightSyncSelection": Selection,
        "torch": type("Torch", (), {"Tensor": object}),
        "stream_weights_via_ipc_zmq_impl": lambda *, params_generator, **_kwargs: (
            payloads["ipc"].append(list(params_generator))
        ),
        "packed_broadcast_producer": lambda *, iterator, **_kwargs: payloads[
            "collective"
        ].append(list(iterator)),
    }
    exec(
        compile(
            ast.Module(body=list(methods.values()), type_ignores=[]),
            "megatron_policy_worker.py",
            "exec",
        ),
        namespace,
    )
    preflight_calls: list[str] = []
    worker = type("Worker", (), {})()
    worker.rank = 0
    worker.zmq_socket = object()
    worker.model_update_group = object()
    worker.maybe_init_zmq = lambda: None
    worker._preflight_draft_weights_for_refit = lambda: (
        preflight_calls.append("pp") or (("draft.weight", b"draft"),),
        None,
    )
    worker._iter_params_with_optional_kv_scales = lambda **kwargs: iter(
        [("target.weight", b"target")] + list(kwargs["draft_weights"])
    )
    for draft in (True, False, True):
        namespace["stream_weights_via_ipc_zmq"](worker, selection=Selection(draft))
        namespace["broadcast_weights_for_collective"](
            worker, selection=Selection(draft)
        )
    expected_names = [
        ["target.weight", "draft.weight"],
        ["target.weight"],
        ["target.weight", "draft.weight"],
    ]
    for transport_payloads in payloads.values():
        assert [[name for name, _ in payload] for payload in transport_payloads] == (
            expected_names
        )
        assert [
            sum(len(value) for _, value in payload) - len(b"target")
            for payload in transport_payloads
        ] == [5, 0, 5]
    assert preflight_calls == ["pp", "pp", "pp", "pp"]
    print(
        "GREEN: Megatron IPC and collective target-only transfers skip draft "
        "preflight, names, and bytes; full recovery verified"
    )


if __name__ == "__main__":
    _run_policy_red_and_green()
    _run_megatron_green()

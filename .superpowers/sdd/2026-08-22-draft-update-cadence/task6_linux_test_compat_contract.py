"""Dependency-free contract for Task 6 Ray actor test compatibility."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
TEST_FILE = ROOT / "tests/unit/models/policy/test_lm_policy_collective.py"
WORKER_FILE = ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
TARGET_TESTS = {
    "test_megatron_worker_target_only_skips_draft_preflight_and_payload": (
        "stream_weights_via_ipc_zmq"
    ),
    "test_megatron_collective_target_only_skips_draft_preflight_and_payload": (
        "broadcast_weights_for_collective"
    ),
}


def _functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }


def _is_impl_reference(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "MegatronPolicyWorkerImpl"
        and isinstance(node.value, ast.Name)
        and node.value.id == "worker_module"
    )


def main() -> None:
    test_functions = _functions(ast.parse(TEST_FILE.read_text()))
    worker_tree = ast.parse(WORKER_FILE.read_text())
    worker_impl = next(
        node
        for node in worker_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerImpl"
    )
    production_methods = {
        node.name for node in worker_impl.body if isinstance(node, ast.FunctionDef)
    }

    for test_name, method_name in TARGET_TESTS.items():
        assert method_name in production_methods
        test = test_functions[test_name]
        assert not any(
            isinstance(node, ast.Attribute) and node.attr == "__ray_metadata__"
            for node in ast.walk(test)
        ), f"RED: {test_name} depends on private Ray actor metadata"
        worker_class_assignments = [
            node
            for node in test.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "worker_cls"
                for target in node.targets
            )
        ]
        assert len(worker_class_assignments) == 1
        assert _is_impl_reference(worker_class_assignments[0].value), (
            f"RED: {test_name} does not instantiate the production implementation"
        )
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method_name
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "worker"
            for node in ast.walk(test)
        )
        assert not any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "worker"
                and target.attr == method_name
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
            for node in ast.walk(test)
        ), f"RED: {test_name} replaces the production method under test"

    print("TASK6_LINUX_TEST_COMPAT_GREEN")


if __name__ == "__main__":
    main()

"""Dependency-free gate for the real synchronous cadence integration."""

import ast
from pathlib import Path


def _function_source(path: Path, name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def _adaptive_runtime_contract() -> None:
    path = Path("nemo_rl/algorithms/draft_cadence_runtime.py")
    source = _function_source(path, "resolve_cadence_schedule_config")
    assert 'schedule.mode == "adaptive"' not in source, (
        "RED: runtime still rejects adaptive scheduling"
    )


def _sync_controller_contract() -> None:
    path = Path("nemo_rl/algorithms/grpo_sync.py")
    source = _function_source(path, "grpo_train_sync")
    required = {
        "prepare_sync_draft_decision(": "selected-rollout decision preparation",
        "capture_draft_science=": "rollout science capture",
        "expected_applied_draft_version=": "serving-version binding",
        "capture_draft_update_receipt=": "worker update receipt capture",
        "apply_scheduled_refit(": "transaction-bound refit finalization",
        "publish_applied_draft_version.remote": "serving-version publication",
    }
    for needle, capability in required.items():
        assert needle in source, f"RED: sync loop lacks {capability}"
    assert "worker_receipt=None" not in source, (
        "RED: sync loop still fabricates an absent worker receipt"
    )
    assert "apply_receipt=None" not in source, (
        "RED: sync loop still fabricates an absent apply receipt"
    )
    finalizer = _function_source(path, "apply_scheduled_refit")
    assert "write_durable_apply_receipt(" in finalizer, (
        "RED: apply success is not crash-safe before version publication"
    )


def _initial_identity_contract() -> None:
    worker = Path("nemo_rl/models/policy/workers/megatron_policy_worker.py").read_text()
    policy = Path("nemo_rl/models/policy/tq_policy.py").read_text()
    assert "def capture_current_draft_state_receipt(" in worker, (
        "RED: worker cannot capture the initial/resumed draft identity"
    )
    assert "def capture_current_draft_state_receipt(" in policy, (
        "RED: controller cannot collect the initial/resumed draft identity"
    )
    runtime = Path("nemo_rl/algorithms/draft_cadence_runtime.py").read_text()
    assert "def write_draft_apply_identity(" in runtime, (
        "RED: controller cannot durably bind a selected draft identity"
    )


if __name__ == "__main__":
    _adaptive_runtime_contract()
    _initial_identity_contract()
    _sync_controller_contract()
    print("GREEN: adaptive sync loop consumes truthful science and receipts")

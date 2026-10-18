#!/usr/bin/env python3
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

"""Validate generation-replica and LLM timing wiring in a source overlay."""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
from typing import Dict


GYM_ROOT = Path("3rdparty/Gym-workspace/Gym")
SCHEMA_PATH = GYM_ROOT / "nemo_gym/openai_utils.py"
PROXY_PATH = GYM_ROOT / "responses_api_models/vllm_model/app.py"
OPENHANDS_PATH = GYM_ROOT / (
    "responses_api_agents/swe_agents/swe_openhands_setup/OpenHands/"
    "openhands/agenthub/nemo_gym_client.py"
)
OPENHANDS_HELPER_PATH = GYM_ROOT / (
    "responses_api_agents/swe_agents/swe_openhands_setup/OpenHands/"
    "openhands/utils/nemo_gym_timeline.py"
)
VLLM_WORKER_PATH = Path(
    "nemo_rl/models/generation/vllm/vllm_worker_async.py"
)
TRTLLM_SERVER_PATH = Path(
    "nemo_rl/models/generation/trtllm/trtllm_http_server.py"
)
LLM_TIMING_FIELDS = (
    "nemo_llm_arrival_ts_us",
    "nemo_llm_queued_ts_us",
    "nemo_llm_first_scheduled_ts_us",
    "nemo_llm_first_token_ts_us",
    "nemo_llm_last_token_ts_us",
    "nemo_llm_response_ready_ts_us",
    "nemo_gym_llm_response_received_ts_us",
    "nemo_gym_tokenize_duration_us",
    "nemo_gym_response_ready_ts_us",
)


def fail(message: str) -> None:
    raise SystemExit(f"replica wiring validation failed: {message}")


def read_source(root: Path, relative_path: Path) -> tuple[Path, str, ast.Module]:
    path = root / relative_path
    if not path.is_file():
        fail(f"missing {path}")
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        fail(f"cannot parse {path}: {error}")
    return path, source, tree


def find_class(tree: ast.Module, name: str, path: Path) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    fail(f"{path}: class {name} not found")


def find_method(cls: ast.ClassDef, name: str, path: Path) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    fail(f"{path}: {cls.name}.{name} not found")


def assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store):
            names.add(child.attr)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
    return names


def validate_schema(root: Path) -> None:
    path, _, tree = read_source(root, SCHEMA_PATH)
    cls = find_class(tree, "NeMoGymChatCompletion", path)
    fields = {
        node.target.id
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    required_fields = {"nemo_generation_replica_idx", *LLM_TIMING_FIELDS}
    missing_fields = required_fields - fields
    if missing_fields:
        fail(
            f"{path}: NeMoGymChatCompletion lacks {sorted(missing_fields)}"
        )


def validate_proxy(root: Path) -> None:
    path, source, tree = read_source(root, PROXY_PATH)
    cls = find_class(tree, "VLLMModel", path)
    post_init = find_method(cls, "_post_init", path)
    resolve_client = find_method(cls, "_resolve_client", path)
    chat_completions = next(
        (
            node
            for node in cls.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "chat_completions"
        ),
        None,
    )
    if chat_completions is None:
        fail(f"{path}: VLLMModel.chat_completions not found")

    if "_session_id_to_replica_idx" not in assigned_names(post_init):
        fail(f"{path}: _post_init does not initialize replica-index state")
    if "_session_id_to_replica_idx" not in source:
        fail(f"{path}: replica-index map is absent")
    if "nemo_generation_replica_idx" not in assigned_names(chat_completions):
        fail(f"{path}: chat_completions never attaches the replica index")
    for field in (
        "nemo_gym_llm_response_received_ts_us",
        "nemo_gym_tokenize_duration_us",
        "nemo_gym_response_ready_ts_us",
    ):
        if field not in assigned_names(chat_completions):
            fail(f"{path}: chat_completions never attaches {field}")

    namespace = {
        "Dict": Dict,
        "hashlib": hashlib,
        "NeMoGymAsyncOpenAI": object,
        "Request": object,
        "SESSION_ID_KEY": "session_id",
    }
    module = ast.fix_missing_locations(ast.Module(body=[resolve_client], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)

    class Router:
        _resolve_client = namespace["_resolve_client"]

        def __init__(self) -> None:
            self._clients = ["replica-0", "replica-1", "replica-2", "replica-3"]
            self._session_id_to_client = {}
            self._session_id_to_replica_idx = {}

        def _maybe_rebind_endpoint(self) -> None:
            """Endpoint-file rebinding (newer Gym revisions) is a no-op here."""

    class Request:
        def __init__(self, session_id: str) -> None:
            self.session = {"session_id": session_id}

    # The selection policy is revision-specific (round-robin in older Gym,
    # session-hash in newer Gym) and must be preserved, not checked here. What
    # the timeline needs is that the retained index names the selected client
    # and that a repeated session keeps both.
    router = Router()
    sessions = [f"session-{idx}" for idx in range(8)]
    observed = {sid: router._resolve_client(Request(sid)) for sid in sessions}
    for sid, client in observed.items():
        idx = router._session_id_to_replica_idx.get(sid)
        if not isinstance(idx, int) or isinstance(idx, bool):
            fail(f"{path}: selected client index is not retained for {sid}")
        if router._clients[idx] != client:
            fail(f"{path}: retained index {idx} does not name the selected client for {sid}")
    for sid in sessions:
        if router._resolve_client(Request(sid)) != observed[sid]:
            fail(f"{path}: repeated session {sid} is not sticky")


def validate_openhands(root: Path) -> None:
    path, source, _ = read_source(root, OPENHANDS_PATH)
    helper_path, helper_source, _ = read_source(root, OPENHANDS_HELPER_PATH)
    if "nemo_generation_replica_idx" not in source:
        fail(f"{path}: proxy response field is never read")
    if 'provider_specific_fields["generation_replica_idx"]' not in source:
        fail(f"{path}: replica index is not added to provider fields")
    if '"generation_replica_idx"' not in source:
        fail(f"{path}: replica index is not copied into model_call metadata")
    for field in LLM_TIMING_FIELDS:
        if field not in helper_source:
            fail(f"{helper_path}: response timing field {field} is never read")
    if "nemo_gym_llm_timing_metrics(model_response_json)" not in source:
        fail(f"{path}: response timing helper is not called")
    if '"first_scheduled_ts_us"' not in source:
        fail(f"{path}: first scheduling timestamp is not copied into metadata")
    for field in (
        "queued_ts_us",
        "llm_response_ready_ts_us",
        "gym_llm_response_received_ts_us",
        "gym_tokenize_duration_us",
        "gym_response_ready_ts_us",
        "openhands_response_received_ts_us",
        "openhands_parse_validate_duration_us",
    ):
        if field not in source and field not in helper_source:
            fail(f"{path}: detailed timing field {field} is not propagated")


def validate_vllm_worker(root: Path) -> None:
    path, source, _ = read_source(root, VLLM_WORKER_PATH)
    for token in (
        "enable_per_request_metrics",
        'getattr(final_res, "metrics", None)',
        "scheduled_ts",
        "queued_ts",
        "nemo_llm_first_scheduled_ts_us",
        "nemo_llm_response_ready_ts_us",
        "time.time_ns() - time.monotonic_ns()",
    ):
        if token not in source:
            fail(f"{path}: missing LLM timing wiring token {token!r}")


def validate_trtllm_server(root: Path) -> None:
    path, source, _ = read_source(root, TRTLLM_SERVER_PATH)
    for token in (
        "NRL_TRTLLM_EMIT_TIMELINE_FIELDS",
        "nemo_llm_arrival_ts_us",
        "nemo_llm_queued_ts_us",
        "nemo_llm_first_scheduled_ts_us",
        "nemo_llm_first_token_ts_us",
        "nemo_llm_last_token_ts_us",
        "nemo_llm_response_ready_ts_us",
    ):
        if token not in source:
            fail(f"{path}: missing TRT-LLM timing wiring token {token!r}")


DISAGG_RESPONSE_FIELDS = (
    "nemo_fe_recv_ts_us",
    "nemo_fe_fwd_ts_us",
    "nemo_ctx_arrival_ts_us",
    "nemo_ctx_queued_ts_us",
    "nemo_ctx_first_scheduled_ts_us",
    "nemo_ctx_done_ts_us",
    "nemo_ctx_cached_tokens",
    "nemo_ctx_computed_tokens",
    "nemo_ctx_first_begin",
    "nemo_ctx_num_chunks",
)


def validate_disagg_propagation(root: Path) -> None:
    """PD-disaggregation frontend/context fields reach the model_call event."""
    path, _, tree = read_source(root, SCHEMA_PATH)
    cls = find_class(tree, "NeMoGymChatCompletion", path)
    fields = {
        node.target.id
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    missing = set(DISAGG_RESPONSE_FIELDS) - fields
    if missing:
        fail(f"{path}: NeMoGymChatCompletion lacks {sorted(missing)}")
    helper_path, helper_source, _ = read_source(root, OPENHANDS_HELPER_PATH)
    for field in DISAGG_RESPONSE_FIELDS:
        if field not in helper_source:
            fail(f"{helper_path}: disaggregation field {field} is never read")
    client_path, client_source, _ = read_source(root, OPENHANDS_PATH)
    if "nemo_gym_disagg_metrics(model_response_json)" not in client_source:
        fail(f"{client_path}: disaggregation helper is not called")
    for field in ("ctx_cached_prompt_tokens", "ctx_first_scheduled_ts_us", "frontend_recv_ts_us"):
        if f'"{field}"' not in client_source:
            fail(f"{client_path}: {field} is not copied into model_call metadata")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Overlay root containing 3rdparty/Gym-workspace/Gym.",
    )
    parser.add_argument(
        "--generation-backend",
        choices=("vllm", "trtllm"),
        default="vllm",
        help="Which generation server must export the request milestones.",
    )
    parser.add_argument(
        "--nemo-rl-root",
        type=Path,
        help=(
            "NeMo-RL root holding the generation server when it is not part of "
            "the overlay (e.g. the image's TRT-LLM server). Defaults to --source-root."
        ),
    )
    args = parser.parse_args()
    nemo_rl_root = args.nemo_rl_root or args.source_root

    validate_schema(args.source_root)
    validate_proxy(args.source_root)
    validate_openhands(args.source_root)
    if args.generation_backend == "trtllm":
        validate_trtllm_server(nemo_rl_root)
        validate_disagg_propagation(args.source_root)
    else:
        validate_vllm_worker(nemo_rl_root)
    print("generation-replica and LLM timing wiring validation passed")


if __name__ == "__main__":
    main()

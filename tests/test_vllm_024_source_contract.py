from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASYNC_WORKER = ROOT / "nemo_rl/models/generation/vllm/vllm_worker_async.py"


def load_tree() -> ast.Module:
    return ast.parse(ASYNC_WORKER.read_text(encoding="utf-8"))


def load_patches_module():
    path = ROOT / "nemo_rl/models/generation/vllm/patches.py"
    spec = importlib.util.spec_from_file_location("nemo_rl_vllm_patches_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_async_http_adapter_imports_vllm_024_tokenization_service() -> None:
    imports = {
        (node.module, alias.name)
        for node in ast.walk(load_tree())
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert (
        "vllm.entrypoints.serve.tokenize.serving",
        "ServingTokenization",
    ) in imports
    assert (
        "vllm.entrypoints.serve.tokenize.serving",
        "OpenAIServingTokenization",
    ) not in imports


def test_async_render_override_uses_vllm_024_parser_argument() -> None:
    preprocess_chat = next(
        node
        for node in ast.walk(load_tree())
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "preprocess_chat"
    )
    arguments = {
        argument.arg
        for argument in [*preprocess_chat.args.args, *preprocess_chat.args.kwonlyargs]
    }

    assert "parser" in arguments
    assert "tool_parser" not in arguments
    assert "reasoning_parser" not in arguments


def test_tokenization_service_constructor_does_not_receive_engine_client() -> None:
    assignment = next(
        node
        for node in ast.walk(load_tree())
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "serving_tokenization_kwargs"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "dict"
    keyword_names = {keyword.arg for keyword in assignment.value.keywords}

    assert "models" in keyword_names
    assert "openai_serving_render" in keyword_names
    assert "engine_client" not in keyword_names


def test_ray_patch_uses_vllm_024_extra_env_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    patches = load_patches_module()
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text(
        "class RayDistributedExecutor:\n"
        "    def _init_executor(self):\n"
        "        self._init_workers_ray(placement_group)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _path: str(ray_executor))
    monkeypatch.setenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "EXISTING_ENV")

    patches._patch_vllm_init_workers_ray(
        "/opt/nemo-rl-venvs/vllm/bin/python",
        ["CUSTOM_ENV", "EXISTING_ENV"],
    )

    patched = ray_executor.read_text(encoding="utf-8")
    assert '_init_workers_ray(placement_group, runtime_env={"py_executable": ' in patched
    assert set(
        patches.os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"].split(",")
    ) == {
        "CUSTOM_ENV",
        "EXISTING_ENV",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NCCL_CUMEM_ENABLE",
        "NCCL_NVLS_ENABLE",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
    }

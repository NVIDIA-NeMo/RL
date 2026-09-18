# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import ast
import hashlib
import os

from omegaconf import OmegaConf
import pytest

from nemo_rl.utils.config import load_config
from tools.image_tools_processor_assets import copy_processor_assets


def make_checkpoint(model: Path) -> dict[str, bytes]:
    model.mkdir()
    files = {
        "config.json": b'{"model_type": "custom_super"}',
        "processor_config.json": b'{"processor_class": "CustomProcessor"}',
        "preprocessor_config.json": b'{"size": 448}',
        "processing_nemotron_h_omni.py": b"from .configuration_radio import RadioConfig\n",
        "image_processing_nemotron_h_omni.py": b"# custom image processor\n",
        "configuration_radio.py": b"class RadioConfig: pass\n",
    }
    for name, data in files.items():
        (model / name).write_bytes(data)
    (model / "model-00001-of-00064.safetensors").write_bytes(b"do not copy weights")
    (model / "model.safetensors.index.json").write_bytes(b"do not copy weight index")
    (model / "tokenizer.json").write_bytes(b"caller saves corrected tokenizer")
    return files


def test_preserves_custom_code_and_settings_without_weights(tmp_path):
    model, target = tmp_path / "model", tmp_path / "export"
    files = make_checkpoint(model)
    hashes = copy_processor_assets(model, target=target)
    assert {path.name for path in target.iterdir()} == set(files)
    for name, original in files.items():
        assert (target / name).read_bytes() == original
        assert (model / name).read_bytes() == original
        assert hashes[name] == hashlib.sha256(original).hexdigest()


def test_refuses_existing_export_without_modifying_it(tmp_path):
    model, target = tmp_path / "model", tmp_path / "export"
    make_checkpoint(model)
    target.mkdir()
    marker = target / "keep"
    marker.write_text("existing run")
    with pytest.raises(FileExistsError):
        copy_processor_assets(model, target=target)
    assert list(target.iterdir()) == [marker]
    assert marker.read_text() == "existing run"


def test_missing_processor_is_rejected_before_creating_export(tmp_path):
    model, target = tmp_path / "model", tmp_path / "export"
    make_checkpoint(model)
    (model / "processing_nemotron_h_omni.py").unlink()
    with pytest.raises(FileNotFoundError, match="processor asset"):
        copy_processor_assets(model, target=target)
    assert not target.exists()


def test_driver_validates_export_through_training_loader():
    root = Path(__file__).resolve().parents[3]
    driver = (root / "tools/image_tools_train_hsg.sh").read_text()
    copied = driver.index("asset_hashes = copy_processor_assets(")
    saved = driver.index("tokenizer.save_pretrained(target)")
    loaded = driver.index(
        "processor = get_tokenizer(tokenizer_config, get_processor=True)"
    )
    ready = driver.index("IMAGE_TOOLS_TRAIN_PREFLIGHT_OK")
    assert copied < saved < loaded < ready
    assert "processor.tokenizer.encode(text)" in driver


def test_preflight_passes_resolved_plain_dict_to_training_loader(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    for key in (
        "MODEL_CHECKPOINT",
        "VLLM_TOKENIZER",
        "TRAIN_MANIFEST",
        "EVAL_MANIFEST",
        "CHECKPOINT_DIR",
        "RUN_LOG_DIR",
        "WANDB_RUN_NAME",
    ):
        monkeypatch.setenv(key, "/fixture/" + key)
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    recipe = (
        root
        / "examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-image-tools-8n4g-megatron-tp8ep16cp2-async.v1.yaml"
    )
    monkeypatch.setenv("IMAGE_TOOLS_CONFIG", str(recipe))
    driver = (root / "tools/image_tools_train_hsg.sh").read_text()
    preflight = driver.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    tree = ast.parse(preflight)
    # Execute the actual config/load statements. Only the GPU-dependent loader
    # is substituted; it enforces the same plain-dict contract as get_tokenizer.
    statements = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {"cfg", "tokenizer_config", "processor"}
            for target in node.targets
        )
    ]
    calls = []

    def checked_loader(config, *, get_processor):
        assert get_processor is True
        assert type(config) is dict
        assert type(config["chat_template_kwargs"]) is dict
        assert config["chat_template_kwargs"]["enable_thinking"] is True
        assert config["name"] == "/fixture/VLLM_TOKENIZER"
        assert config["chat_template"].startswith(str(root))
        calls.append(config)
        return object()

    namespace = {
        "os": os,
        "OmegaConf": OmegaConf,
        "load_config": load_config,
        "get_tokenizer": checked_loader,
    }
    exec(
        compile(
            ast.Module(body=statements, type_ignores=[]),
            str(root / "tools/image_tools_train_hsg.sh"),
            "exec",
        ),
        namespace,
    )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "name,expected",
    [
        ("NemotronH_Omni_Reasoning_V3Processor", True),
        ("NemotronH_Nano_Omni_Reasoning_V3Processor", True),
        ("NemotronNanoVLV2Processor", True),
        ("UnknownProcessor", False),
    ],
)
def test_explicit_placeholder_dispatch_without_gpu_imports(name, expected):
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / "nemo_rl/data/multimodal_utils.py").read_text())
    statements = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_PLACEHOLDER_STYLE_PROCESSOR_NAMES"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name in {"uses_image_placeholder", "get_pad_to_max_shape"}
        )
    ]
    namespace = {"Any": object}
    exec(
        compile(
            ast.Module(body=statements, type_ignores=[]), "multimodal_utils.py", "exec"
        ),
        namespace,
    )
    processor = type(name, (), {})()
    assert namespace["uses_image_placeholder"](processor) is expected
    assert namespace["get_pad_to_max_shape"](processor, "pixel_values") is expected
    assert namespace["get_pad_to_max_shape"](processor, "imgs_sizes") is False

from __future__ import annotations

import pytest

from nemo_rl.models.multi_lora.config import MultiLoRAConfig


def _adapter(name: str = "a") -> dict:
    return {"name": name, "data": {}}


def test_from_dict_keeps_only_runtime_fields() -> None:
    cfg = MultiLoRAConfig.from_dict(
        {
            "enabled": True,
            "batch_size_per_adapter": 8,
            "adapters": [_adapter()],
        }
    )

    assert cfg.enabled is True
    assert cfg.batch_size_per_adapter == 8
    assert [adapter.name for adapter in cfg.adapters] == ["a"]


@pytest.mark.parametrize(
    "obsolete_key",
    [
        "schedule",
        "global_batch_size",
        "steps_per_adapter",
        "storage_device",
        "execution_mode",
        "hbm_budget_gb",
    ],
)
def test_removed_top_level_options_are_ignored(obsolete_key: str) -> None:
    cfg = MultiLoRAConfig.from_dict(
        {"enabled": True, "adapters": [_adapter()], obsolete_key: "unused"}
    )

    cfg.validate()
    assert not hasattr(cfg, obsolete_key)


@pytest.mark.parametrize(
    "obsolete_key", ["lora_cfg", "optimizer", "weight", "nemo_gym", "pin"]
)
def test_removed_adapter_options_are_rejected(obsolete_key: str) -> None:
    adapter = _adapter()
    adapter[obsolete_key] = "unused"

    with pytest.raises(TypeError, match=obsolete_key):
        MultiLoRAConfig.from_dict({"enabled": True, "adapters": [adapter]})


def test_validate_rejects_empty_adapters() -> None:
    cfg = MultiLoRAConfig.from_dict({"enabled": True, "adapters": []})

    with pytest.raises(ValueError, match="at least 1 entry"):
        cfg.validate()


def test_validate_rejects_duplicate_names() -> None:
    cfg = MultiLoRAConfig.from_dict(
        {"enabled": True, "adapters": [_adapter(), _adapter()]}
    )

    with pytest.raises(ValueError, match="Duplicate adapter names"):
        cfg.validate()


def test_validate_rejects_nonpositive_batch_size() -> None:
    cfg = MultiLoRAConfig.from_dict(
        {
            "enabled": True,
            "batch_size_per_adapter": 0,
            "adapters": [_adapter()],
        }
    )

    with pytest.raises(ValueError, match="batch_size_per_adapter"):
        cfg.validate()
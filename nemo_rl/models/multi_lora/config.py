"""Configuration consumed by the packed multi-adapter SFT path."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MultiLoRAAdapter:
    """A single LoRA adapter slot in a multi-adapter run."""

    name: str
    data: dict


@dataclass(frozen=True)
class MultiLoRAConfig:
    """The fields that the native concurrent SFT implementation reads."""

    enabled: bool
    batch_size_per_adapter: int = 1
    adapters: list[MultiLoRAAdapter] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "MultiLoRAConfig":
        adapters = [MultiLoRAAdapter(**a) for a in d.get("adapters", [])]
        return cls(
            enabled=d.get("enabled", False),
            batch_size_per_adapter=d.get("batch_size_per_adapter", 1),
            adapters=adapters,
        )

    def validate(self) -> None:
        if len(self.adapters) < 1:
            raise ValueError("multi_lora.adapters must have at least 1 entry")
        names = [a.name for a in self.adapters]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate adapter names: {names}")
        if self.batch_size_per_adapter < 1:
            raise ValueError(
                f"multi_lora.batch_size_per_adapter must be >= 1, got "
                f"{self.batch_size_per_adapter}"
            )

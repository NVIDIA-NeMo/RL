# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field, model_validator


class ComponentSpec(BaseModel, extra="forbid"):
    factory: str
    config: dict[str, Any] = Field(default_factory=dict)

    def build(self, **dependencies):
        module, name = self.factory.split(":")
        factory = getattr(import_module(module), name)
        config = factory.Config.model_validate(self.config)
        return factory(config=config, **dependencies)


class Prompt(BaseModel, extra="forbid"):
    id: str
    turns: int = Field(gt=0)
    turn_seconds: list[float] = Field(min_length=1)


class Scenario(BaseModel, extra="forbid"):
    prompts: list[Prompt] = Field(min_length=1)
    siblings_per_prompt: int = Field(gt=0)
    policy: ComponentSpec
    generation: ComponentSpec
    refit: ComponentSpec

    @model_validator(mode="after")
    def check_prompts(self) -> Self:
        if len({p.id for p in self.prompts}) != len(self.prompts):
            raise ValueError("Prompt IDs must be unique")
        for prompt in self.prompts:
            if len(prompt.turn_seconds) != self.siblings_per_prompt:
                raise ValueError("Specify one delay per sibling")
            if any(seconds <= 0 for seconds in prompt.turn_seconds):
                raise ValueError("Turn delays must be positive")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate(yaml.safe_load(path.read_text()))

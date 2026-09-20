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

"""Unit tests for Automodel packing boundary translation."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from nemo_rl.models.automodel.packing import (
    _flash_attn_kwargs_as_mapping,
    _model_expects_native_packing_args,
    _promote_flash_attn_kwargs_to_native_packing,
)


@dataclass
class _FlashAttnKwargs:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


class _NativePackingModel(nn.Module):
    """Fake Automodel-style model that accepts top-level cu_seqlens."""

    def __init__(self, model_type: str = "nemotron_h"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        qkv_format: Optional[str] = None,
        **kwargs: Any,
    ):
        del input_ids, cu_seqlens, qkv_format, kwargs
        return None


class _HfFlashAttnOnlyModel(nn.Module):
    """Fake HF FA2 model that only documents nested flash_attn_kwargs."""

    def forward(
        self,
        input_ids: torch.Tensor,
        flash_attn_kwargs: Optional[dict[str, Any]] = None,
    ):
        del input_ids, flash_attn_kwargs
        return None


class _NemotronByNameModel(nn.Module):
    """Model that only exposes packing via **kwargs but is Nemotron-typed."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="nemotron_h")

    def forward(self, input_ids: torch.Tensor, **kwargs: Any):
        del input_ids, kwargs
        return None


def _sample_flash_kwargs() -> _FlashAttnKwargs:
    return _FlashAttnKwargs(
        cu_seqlens_q=torch.tensor([0, 32, 64], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 32, 64], dtype=torch.int32),
        max_seqlen_q=32,
        max_seqlen_k=32,
    )


class TestFlashAttnKwargsAsMapping:
    def test_dataclass_and_dict(self):
        flash = _sample_flash_kwargs()
        as_map = _flash_attn_kwargs_as_mapping(flash)
        assert torch.equal(as_map["cu_seqlens_q"], flash.cu_seqlens_q)
        assert as_map["max_seqlen_q"] == 32

        as_map2 = _flash_attn_kwargs_as_mapping({"cu_seqlens_q": flash.cu_seqlens_q})
        assert torch.equal(as_map2["cu_seqlens_q"], flash.cu_seqlens_q)



class TestModelExpectsNativePackingArgs:
    def test_explicit_cu_seqlens_forward(self):
        assert _model_expects_native_packing_args(_NativePackingModel()) is True

    def test_hf_flash_attn_only_forward(self):
        assert _model_expects_native_packing_args(_HfFlashAttnOnlyModel()) is False

    def test_nemotron_model_type(self):
        assert _model_expects_native_packing_args(_NemotronByNameModel()) is True

    def test_magicmock_defaults_to_nested(self):
        # Existing automodel train tests use MagicMock models and expect nested
        # flash_attn_kwargs to remain. Detection must stay conservative.
        assert _model_expects_native_packing_args(MagicMock()) is False


class TestPromoteFlashAttnKwargsToNativePacking:
    def test_promotes_top_level_args_and_qkv_format(self):
        flash = _sample_flash_kwargs()
        model_batch = {
            "input_ids": torch.randint(0, 100, (1, 64)),
            "flash_attn_kwargs": flash,
        }
        _promote_flash_attn_kwargs_to_native_packing(model_batch)

        assert "flash_attn_kwargs" not in model_batch
        assert torch.equal(model_batch["cu_seqlens"], flash.cu_seqlens_q)
        assert torch.equal(model_batch["cu_seqlens_q"], flash.cu_seqlens_q)
        assert torch.equal(model_batch["cu_seqlens_kv"], flash.cu_seqlens_k)
        assert model_batch["max_seqlen"] == 32
        assert model_batch["max_seqlen_q"] == 32
        assert model_batch["max_seqlen_kv"] == 32
        assert model_batch["qkv_format"] == "thd"
        # Packed layout stays [1, T]; Automodel squeezes when qkv_format is thd.
        assert model_batch["input_ids"].shape == (1, 64)


@pytest.mark.automodel
class TestBuildModelBatchPackingTranslation:
    """Exercise _build_model_batch with native vs HF FA2 fake models."""

    @pytest.fixture(autouse=True)
    def _import_train(self):
        try:
            import nemo_automodel  # noqa: F401
        except ImportError:
            pytest.skip("nemo_automodel not available")

        from nemo_rl.models.automodel.data import ProcessedInputs
        from nemo_rl.models.automodel.train import _build_model_batch

        self.ProcessedInputs = ProcessedInputs
        self._build_model_batch = _build_model_batch

    def _processed(self):
        flash = _sample_flash_kwargs()
        return self.ProcessedInputs(
            input_ids=torch.randint(0, 1000, (1, 64)),
            seq_len=64,
            attention_mask=None,
            position_ids=torch.arange(64).unsqueeze(0),
            flash_attn_kwargs=flash,
            vlm_kwargs={},
        )

    def test_promotes_for_native_cu_seqlens_model(self):
        model_batch = self._build_model_batch(
            _NativePackingModel(),
            self._processed(),
            is_reward_model=False,
            allow_flash_attn_args=True,
            clone_model_tensors=False,
        )
        assert "flash_attn_kwargs" not in model_batch
        assert "cu_seqlens" in model_batch
        assert model_batch["qkv_format"] == "thd"

    def test_keeps_nested_for_hf_flash_attn_only_model(self):
        model_batch = self._build_model_batch(
            _HfFlashAttnOnlyModel(),
            self._processed(),
            is_reward_model=False,
            allow_flash_attn_args=True,
            clone_model_tensors=False,
        )
        assert "flash_attn_kwargs" in model_batch
        assert "cu_seqlens" not in model_batch
        assert "qkv_format" not in model_batch

    def test_promotes_for_nemotron_model_type(self):
        processed = self._processed()
        model_batch = self._build_model_batch(
            _NemotronByNameModel(),
            processed,
            is_reward_model=False,
            allow_flash_attn_args=True,
            clone_model_tensors=False,
        )
        assert "flash_attn_kwargs" not in model_batch
        assert torch.equal(
            model_batch["cu_seqlens"],
            processed.flash_attn_kwargs.cu_seqlens_q,
        )
        assert model_batch["qkv_format"] == "thd"

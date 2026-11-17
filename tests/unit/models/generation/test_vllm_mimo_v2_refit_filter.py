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

"""Refit filter for the text-only MiMo-V2 rollout model.

The trainer holds the full multimodal model; vLLM runs MiMoV2ForCausalLM (no towers) and refuses
tower weight names, so refit drops them.
"""

from types import SimpleNamespace

import torch

from nemo_rl.models.generation.vllm.vllm_backend import (
    _filter_mimo_v2_non_lm_weights,
    _is_mimo_v2_text_only,
)


def test_text_only_is_detected_from_the_architecture():
    assert _is_mimo_v2_text_only(SimpleNamespace(architectures=["MiMoV2ForCausalLM"]))
    assert not _is_mimo_v2_text_only(SimpleNamespace(architectures=["MiMoV2OmniForCausalLM"]))
    assert not _is_mimo_v2_text_only(SimpleNamespace(architectures=None))
    assert not _is_mimo_v2_text_only(SimpleNamespace())


def test_filter_drops_tower_weights_and_keeps_language_model_weights():
    t = torch.zeros(1)
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
        "visual.blocks.0.attn.qkv.weight",
        "audio_encoder.layers.0.fc1.weight",
        "speech_embeddings.0.weight",
    ]
    kept = [k for k, _ in _filter_mimo_v2_non_lm_weights((n, t) for n in names)]
    assert kept == names[:3]

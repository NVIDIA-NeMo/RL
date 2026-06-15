# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the IntentTrain / IntentBench dataset loader.

The audio+video sample-shape contract (every prompt carries one
``{type:video}`` + one ``{type:audio}`` + a text item, so the chat template
emits both ``<|VIDEO|>`` and ``<|AUDIO|>`` placeholders) is exercised end to
end by the functional test ``tests/functional/audio_visual_grpo_megatron.sh``
and by the vLLM-utils unit tests. The dedicated unit check for it required
``ffmpeg`` to fabricate an mp4 with an audio track, so it is intentionally not
included here — the unit suite stays ffmpeg-free.
"""

import pytest


class TestIntentDataset:
    def test_intent_invalid_split_raises(self):
        from nemo_rl.data.datasets.response_datasets.intent import IntentDataset

        with pytest.raises(ValueError, match="Invalid split"):
            IntentDataset(split="test")

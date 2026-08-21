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
#
# Extends nemo_automodel.components.speculative.dspark.markov_head (3rdparty
# Automodel r0.6.0). The head classes are re-exported unchanged; the local
# delta is reduced-draft-vocab support for speculators-format checkpoints
# (draft_vocab_size < target vocab): previous tokens are target-space ids, so
# the prev-token embedding spans the FULL target vocab while the bias
# projection outputs draft-vocab logits. Upstream heads assume one vocab for
# both sides.
from torch import nn

from nemo_automodel.components.speculative.dspark.markov_head import (  # noqa: F401
    GatedMarkovHead,
    RNNHead,
    VanillaMarkov as _UpstreamVanillaMarkov,
)


class VanillaMarkov(_UpstreamVanillaMarkov):
    """Upstream VanillaMarkov with a split embedding/output vocabulary."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        embed_vocab_size: int | None = None,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        # Reduced-draft-vocab checkpoints (draft_vocab_size < target vocab)
        # embed previous tokens over the FULL target vocab (token ids are
        # target-space) while the bias projection outputs draft-vocab logits.
        self.embed_vocab_size = int(embed_vocab_size or vocab_size)
        if self.embed_vocab_size != self.vocab_size:
            self.markov_w1 = nn.Embedding(self.embed_vocab_size, self.markov_rank)


def build_markov_head(config) -> nn.Module | None:
    markov_rank = int(config.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank == 0:
        return None

    markov_head_type = str(config.markov_head_type).lower()
    if markov_head_type == "vanilla":
        # Reduced-draft-vocab checkpoints: bias output over draft_vocab_size,
        # prev-token embedding over the full target vocab.
        output_vocab = int(
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        return VanillaMarkov(
            vocab_size=output_vocab,
            markov_rank=markov_rank,
            embed_vocab_size=config.vocab_size,
        )
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    assert False, f"Unsupported markov_head_type: {markov_head_type!r}"


__all__ = [
    "VanillaMarkov",
    "GatedMarkovHead",
    "RNNHead",
    "build_markov_head",
]

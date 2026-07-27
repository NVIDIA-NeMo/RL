# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Published layer names must be pipeline-GLOBAL (Bug 8).

Megatron's ``named_parameters()`` reports each layer's index within the local
pipeline stage. With ``pp_size=2`` and 48 layers, both stages report
``decoder.layers.0..23``. MX's shard table is keyed by name and tiebroken
first-writer-wins, so colliding names mean one stage's offer displaces the
other's: half the model is never refit, and where the second stage wins,
destination layer N is filled from global layer N+24 at a valid address with a
self-consistent digest. Every digest gate passes, because bytes that are never
requested are never checked.

These tests pin the invariant that makes that impossible: the index in the
published name is the global one, read from ``TransformerLayer.layer_number``.
"""

from __future__ import annotations

import pytest

from nemo_rl.distributed.mx_megatron_helpers import (
    globalize_megatron_layer_name,
    map_local_layers_to_global,
)


class _Layer:
    """Stands in for a Megatron ``TransformerLayer``.

    Only ``layer_number`` matters, and it is 1-based and global, exactly as
    Megatron assigns it.
    """

    def __init__(self, layer_number: int) -> None:
        self.layer_number = layer_number
        # Submodules inherit the attribute in real Megatron, which is why the
        # mapping has to require a ModuleList index tail rather than just the
        # presence of layer_number.
        self.self_attention = _SubModule(layer_number)


class _SubModule:
    def __init__(self, layer_number: int) -> None:
        self.layer_number = layer_number


class _Block:
    """Stands in for ``decoder``, holding this stage's layers."""

    def __init__(self, layer_numbers: list[int]) -> None:
        self.layers = list(layer_numbers)


class _Model:
    """Minimal stand-in exposing ``named_modules`` / ``named_parameters``."""

    def __init__(
        self,
        layer_numbers: list[int],
        *,
        prefix: str = "",
        extra_params: list[str] | None = None,
        drop_layer_number: bool = False,
    ) -> None:
        self._layer_numbers = layer_numbers
        self._prefix = prefix
        self._extra_params = extra_params or []
        self._drop_layer_number = drop_layer_number

    def named_modules(self):
        p = self._prefix
        yield f"{p}decoder".rstrip("."), _Block(self._layer_numbers)
        for local_index, global_number in enumerate(self._layer_numbers):
            path = f"{p}decoder.layers.{local_index}"
            if self._drop_layer_number:
                yield path, _SubModule(0).__class__.__new__(_SubModule)
                continue
            layer = _Layer(global_number)
            yield path, layer
            yield f"{path}.self_attention", layer.self_attention

    def named_parameters(self):
        p = self._prefix
        for local_index in range(len(self._layer_numbers)):
            for leaf in ("self_attention.linear_qkv.weight", "mlp.linear_fc1.weight"):
                yield f"{p}decoder.layers.{local_index}.{leaf}", None
        for extra in self._extra_params:
            yield f"{p}{extra}", None


def _publish_names(model: _Model, *, pp_size: int) -> list[str]:
    """Names as ``collect_megatron_publish_set`` would publish them."""
    mapping = map_local_layers_to_global(model, pp_size=pp_size)
    out = []
    for raw_name, _param in model.named_parameters():
        name = (
            raw_name[len("module.") :] if raw_name.startswith("module.") else raw_name
        )
        out.append(globalize_megatron_layer_name(raw_name, name, mapping))
    return out


def _layer_indices(names: list[str]) -> list[int]:
    return sorted(
        {
            int(n.split("decoder.layers.")[1].split(".")[0])
            for n in names
            if "decoder.layers." in n
        }
    )


def test_second_pipeline_stage_publishes_its_global_indices():
    """Stage 1 of 2 owns global layers 24-47 and must say so."""
    stage1 = _Model(list(range(25, 49)))  # 1-based layer_number 25..48
    assert _layer_indices(_publish_names(stage1, pp_size=2)) == list(range(24, 48))


def test_the_two_stages_no_longer_collide():
    stage0 = _Model(list(range(1, 25)))
    stage1 = _Model(list(range(25, 49)))
    names = _publish_names(stage0, pp_size=2) + _publish_names(stage1, pp_size=2)
    # 48 distinct layers, and no name published twice.
    assert _layer_indices(names) == list(range(48))
    assert len(names) == len(set(names))


def test_single_stage_names_are_untouched():
    """At pp_size=1 local index equals global index; this is Topology A."""
    model = _Model(list(range(1, 5)))
    before = [n for n, _ in model.named_parameters()]
    assert _publish_names(model, pp_size=1) == before


def test_non_layer_params_are_untouched():
    model = _Model(
        list(range(25, 27)),
        extra_params=[
            "embedding.word_embeddings.weight",
            "decoder.final_layernorm.weight",
        ],
    )
    names = _publish_names(model, pp_size=2)
    assert "embedding.word_embeddings.weight" in names
    assert "decoder.final_layernorm.weight" in names


def test_wrapped_models_resolve_through_the_module_prefix():
    """`module.` prefixed names must still map, and publish unprefixed."""
    model = _Model(list(range(25, 29)), prefix="module.")
    names = _publish_names(model, pp_size=2)
    assert all(not n.startswith("module.") for n in names)
    assert _layer_indices(names) == [24, 25, 26, 27]


def test_virtual_pipeline_and_uneven_stages_come_out_right():
    """Non-contiguous ownership is why the offset is read, not computed.

    A rank holding global layers 4-7 and 20-23 cannot be described by
    ``pp_rank * layers_per_stage``; reading ``layer_number`` handles it.
    """
    model = _Model([5, 6, 7, 8, 21, 22, 23, 24])
    assert _layer_indices(_publish_names(model, pp_size=4)) == [
        4,
        5,
        6,
        7,
        20,
        21,
        22,
        23,
    ]


def test_missing_layer_number_under_pp_is_an_error_not_a_silent_local_name():
    model = _Model(list(range(1, 5)), drop_layer_number=True)
    with pytest.raises(RuntimeError) as excinfo:
        map_local_layers_to_global(model, pp_size=2)
    assert "layer_number" in str(excinfo.value)


def test_missing_layer_number_without_pp_is_fine():
    model = _Model(list(range(1, 5)), drop_layer_number=True)
    assert map_local_layers_to_global(model, pp_size=1) == {}

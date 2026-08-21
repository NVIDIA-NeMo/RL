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

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.vllm.speculator_runtime import (
    DraftRuntimeAdapter,
    ModelUpdateCoverage,
    ModelUpdateManifest,
    RunnerFamily,
    SpeculatorRuntimeError,
)


@pytest.mark.parametrize("speculator_type", ["dflash", "dspark"])
def test_runtime_adapter_prefers_public_accessor(speculator_type: str) -> None:
    accessed = object()
    runner = SimpleNamespace(
        get_draft_model=lambda: accessed,
        drafter=SimpleNamespace(model=object()),
        speculator=SimpleNamespace(model=object()),
    )

    adapter = DraftRuntimeAdapter.resolve(
        runner,
        speculator_type=speculator_type,
        vllm_version="0.27.1",
        pp_rank=0,
        pp_size=1,
    )

    assert adapter.runner_family is RunnerFamily.ACCESSOR
    assert adapter.model is accessed
    assert adapter.is_owner


@pytest.mark.parametrize(
    "attribute,family",
    [("drafter", RunnerFamily.DRAFTER), ("speculator", RunnerFamily.SPECULATOR)],
)
def test_runtime_adapter_supports_legacy_runner_layouts(
    attribute: str, family: RunnerFamily
) -> None:
    draft_model = object()
    runner = SimpleNamespace(**{attribute: SimpleNamespace(model=draft_model)})

    adapter = DraftRuntimeAdapter.resolve(
        runner,
        speculator_type="eagle3",
        vllm_version="0.25.1",
        pp_rank=0,
        pp_size=1,
    )

    assert adapter.runner_family is family
    assert adapter.model is draft_model


@pytest.mark.parametrize("speculator_type", ["dflash", "dspark"])
def test_runtime_adapter_rejects_unproven_pipeline_parallelism(
    speculator_type: str,
) -> None:
    runner = SimpleNamespace(get_draft_model=object)

    with pytest.raises(SpeculatorRuntimeError) as error:
        DraftRuntimeAdapter.resolve(
            runner,
            speculator_type=speculator_type,
            vllm_version="0.27.1",
            pp_rank=0,
            pp_size=2,
        )

    message = str(error.value)
    assert speculator_type in message
    assert "0.27.1" in message
    assert RunnerFamily.ACCESSOR.value in message
    assert "PP=2" in message


def test_eagle3_non_owner_participates_without_a_local_draft_model() -> None:
    runner = SimpleNamespace(drafter=SimpleNamespace(model=None))

    adapter = DraftRuntimeAdapter.resolve(
        runner,
        speculator_type="eagle3",
        vllm_version="0.25.1",
        pp_rank=0,
        pp_size=2,
    )

    assert not adapter.is_owner
    assert adapter.model is None


def test_owner_without_a_draft_model_is_a_setup_error() -> None:
    runner = SimpleNamespace(get_draft_model=lambda: None)

    with pytest.raises(SpeculatorRuntimeError, match="owner.*unavailable"):
        DraftRuntimeAdapter.resolve(
            runner,
            speculator_type="eagle3",
            vllm_version="0.27.1",
            pp_rank=0,
            pp_size=1,
        )


def test_model_update_manifest_is_ordered_and_component_aware() -> None:
    state_dict_info = {
        "model.layers.0.weight": ((2, 3), torch.bfloat16),
        "draft.model.layers.0.weight": ((4, 3), torch.float32),
        "model.norm.weight": ((3,), torch.bfloat16),
    }

    manifest = ModelUpdateManifest.from_state_dict_info(
        state_dict_info,
        target_owner_ranks=(0, 1),
        draft_owner_ranks=(1,),
    )

    assert manifest.target.ordered_names == (
        "model.layers.0.weight",
        "model.norm.weight",
    )
    assert manifest.target.byte_count == 18
    assert manifest.target.owner_ranks == (0, 1)
    assert manifest.target.loader == "target.load_weights"
    assert manifest.draft is not None
    assert manifest.draft.ordered_names == ("draft.model.layers.0.weight",)
    assert manifest.draft.byte_count == 48
    assert manifest.draft.owner_ranks == (1,)
    assert manifest.draft.loader == "draft.load_weights"
    assert manifest.draft.finalizer == "process_weights_after_loading"


def test_model_update_coverage_requires_every_input_exactly_once() -> None:
    manifest = ModelUpdateManifest.from_state_dict_info(
        {
            "model.weight": ((2,), torch.float32),
            "draft.model.weight": ((2,), torch.float32),
        },
        target_owner_ranks=(0,),
        draft_owner_ranks=(0,),
    )
    coverage = ModelUpdateCoverage(manifest, rank=0)

    coverage.record_loaded(("model.weight",))
    with pytest.raises(SpeculatorRuntimeError, match="missing keys"):
        coverage.require_complete()

    coverage.record_loaded(("draft.model.weight",))
    coverage.require_complete()

    with pytest.raises(SpeculatorRuntimeError, match="duplicate keys"):
        coverage.record_loaded(("model.weight",))

    coverage = ModelUpdateCoverage(manifest, rank=0)
    with pytest.raises(SpeculatorRuntimeError, match="duplicate keys"):
        coverage.record_loaded(("model.weight", "model.weight"))


def test_non_owner_records_draft_skip_but_still_requires_transport_coverage() -> None:
    manifest = ModelUpdateManifest.from_state_dict_info(
        {
            "model.weight": ((2,), torch.float32),
            "draft.model.weight": ((2,), torch.float32),
        },
        target_owner_ranks=(0, 1),
        draft_owner_ranks=(1,),
    )
    coverage = ModelUpdateCoverage(manifest, rank=0)

    coverage.record_loaded(("model.weight",))
    coverage.record_owner_skip(("draft.model.weight",), component="draft")
    coverage.require_complete()

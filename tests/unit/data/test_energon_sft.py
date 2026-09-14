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

import io
from copy import deepcopy
from typing import Any

import pytest
import torch
from PIL import Image

# nemo_rl/data/energon/{cookers,task_encoders,types,sft_dataloader} import
# megatron.energon at module scope, and it ships only in the `mcore` extra.
# importorskip must run before those imports: the mcore mark is applied in
# pytest_collection_modifyitems, too late to prevent a collection error.
pytest.importorskip("megatron.energon")

pytestmark = pytest.mark.mcore

from nemo_rl.algorithms.sft import prepare_sft_batch  # noqa: E402
from nemo_rl.data.collate_fn import rl_collate_fn  # noqa: E402
from nemo_rl.data.energon.config import (  # noqa: E402
    EnergonLoaderConfig,
    EnergonSourceConfig,
)
from nemo_rl.data.energon.multimodal.cookers.generic import (  # noqa: E402
    cook_conversation,
)
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (  # noqa: E402
    GenericSFTTaskEncoder,
    HFMultimodalSFTProcessorAdapter,
)
from nemo_rl.data.energon.multimodal.types import (  # noqa: E402
    CanonicalSFTSample,
    MediaRef,
)
from nemo_rl.data.energon.sft_dataloader import (  # noqa: E402
    EnergonSFTDataLoader,
    _identity_fingerprint,
    _loader_config,
    _loader_identity,
    build_energon_sft_loader,
)
from nemo_rl.data.interfaces import TaskDataSpec  # noqa: E402
from nemo_rl.data.llm_message_utils import (  # noqa: E402
    message_log_to_flat_messages,
)
from nemo_rl.data.multimodal_utils import PackedTensor  # noqa: E402
from nemo_rl.data.processors import sft_processor  # noqa: E402


class _FakeTokenizer:
    pad_token_id = 0
    bos_token = None
    eos_token = "<eos>"
    model_input_names = ["input_ids", "attention_mask"]
    name_or_path = "fake-tokenizer"
    chat_template = "fake-template"


class _FakeImageProcessor:
    model_input_names = ["pixel_values", "image_grid_thw"]


class _FakeQwenProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = _FakeImageProcessor()
    model_input_names = [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    ]
    name_or_path = "fake-qwen3-vl"
    pad_token_id = 0
    bos_token = None
    eos_token = "<eos>"

    def __init__(self):
        self.messages = None
        self.tools = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = deepcopy(messages)
        self.tools = kwargs.get("tools")
        assert kwargs["tokenize"] is False

        def render_content(content):
            if isinstance(content, str):
                return content
            return "".join(
                part.get("text", "") if part["type"] == "text" else f"<{part['type']}>"
                for part in content
            )

        return "".join(
            f"<{message['role']}>{render_content(message.get('content', ''))}"
            f"</{message['role']}>"
            for message in messages
        )

    def __call__(self, *, text, images=None, **kwargs):
        text = text[0] if isinstance(text, list) else text
        input_ids = torch.tensor(
            [[ord(character) % 251 + 1 for character in text]], dtype=torch.long
        )
        processed = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": torch.zeros_like(input_ids),
        }
        if images:
            processed["pixel_values"] = torch.ones(len(images), 4)
            processed["image_grid_thw"] = torch.tensor(
                [[1, 2, 2]] * len(images), dtype=torch.long
            )
        return processed


class NemotronH_Nano_Omni_Reasoning_V3Processor(_FakeQwenProcessor):
    model_input_names = ["input_ids", "pixel_values", "imgs_sizes", "num_frames"]

    def __call__(self, *, text, images=None, **kwargs):
        processed = {"input_ids": torch.tensor([[10, 20, 21, 22]])}
        if images:
            processed["pixel_values"] = torch.ones(len(images), 3, 4, 4)
        return processed


def _rgb_image(level: int) -> Image.Image:
    """Build media the way Energon does.

    ``GenericSFTTaskEncoder`` decodes with ``SampleDecoder(image_decode="pilrgb")``,
    so a cooked ``MediaRef`` carries a PIL image -- not a raw tensor.
    """
    return Image.new("RGB", (4, 4), color=(level, level, level))


def _sample(*, with_tools=False):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "media_index": 0},
                {"type": "text", "text": "Compare these."},
                {"type": "image", "media_index": 1},
            ],
        },
        {"role": "assistant", "content": "They match."},
    ]
    tools = None
    if with_tools:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "done",
                },
                {"role": "assistant", "content": "Final answer."},
            ]
        )
        tools = [{"type": "function", "function": {"name": "lookup"}}]
    return CanonicalSFTSample(
        __key__="sample-0",
        __restore_key__=("sample-0",),
        messages=messages,
        media=[
            MediaRef("image", _rgb_image(255)),
            MediaRef("image", _rgb_image(0)),
        ],
        tools=tools,
    )


def _hf_messages(sample: CanonicalSFTSample) -> list[dict[str, Any]]:
    """Rewrite a canonical conversation into the shape the HF path expects.

    The two backends deliberately do not share a message convention. Energon
    keeps media out of line -- a content part carries ``media_index`` into
    ``sample.media`` -- while ``sft_processor`` reads the payload inline off the
    part itself (``{"type": "image", "image": <PIL.Image>}``). Inlining here
    keeps the parity test comparing the two backends rather than the two
    conventions.
    """
    messages = []
    for message in sample.messages:
        content = message.get("content")
        if not isinstance(content, list):
            messages.append(message)
            continue
        inlined = []
        for part in content:
            part = dict(part)
            media_index = part.pop("media_index", None)
            if media_index is not None:
                media_ref = sample.media[media_index]
                part[media_ref.modality] = media_ref.value
            inlined.append(part)
        messages.append({**message, "content": inlined})
    return messages


def _adapter(processor, *, max_sequence_length=1024):
    return HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=max_sequence_length,
        add_bos=False,
        add_eos=False,
        add_generation_prompt=False,
    )


def _encoder(adapter, *, include_source_ids=False):
    return GenericSFTTaskEncoder(
        adapter=adapter,
        cooker_functions=[cook_conversation],
        include_source_ids=include_source_ids,
    )


def test_qwen_adapter_returns_tokenized_message_log_with_model_inputs():
    processor = _FakeQwenProcessor()
    encoded = _adapter(processor).encode(_sample(with_tools=True))

    user_content = processor.messages[0]["content"]
    assert [part["type"] for part in user_content] == ["image", "text", "image"]
    assert user_content[0]["image"].getpixel((0, 0)) == (255, 255, 255)
    assert user_content[2]["image"].getpixel((0, 0)) == (0, 0, 0)
    assert processor.tools == [{"type": "function", "function": {"name": "lookup"}}]

    flat = message_log_to_flat_messages(encoded.message_log)
    assert [message["role"] for message in encoded.message_log] == [
        "user",
        "assistant",
        "assistant",
        "tool",
        "assistant",
    ]
    assert isinstance(flat["pixel_values"], PackedTensor)
    assert isinstance(flat["image_grid_thw"], PackedTensor)
    # mm_token_type_ids rides along with the multimodal turn only: get_formatted_message_log
    # attaches it where the processor ran (the user message), and the text-only turns are
    # tokenized without it. Batching zero-fills the rest, which is the correct text type.
    assert (
        flat["mm_token_type_ids"].shape[0]
        == encoded.message_log[0]["token_ids"].shape[0]
    )


def test_hf_and_energon_backends_agree_on_the_same_conversation():
    """Both backends feed prepare_sft_batch; the prepared tensors must match.

    Each backend is handed the conversation in its own message convention (see
    ``_hf_messages``) -- the claim under test is that the two agree on the
    prepared tensors, not on the input shape.

    Truncation is deliberately not compared: above max_sequence_length the
    Energon encoder also empties every PackedTensor (generic_sft.py:225-228)
    while the HF path leaves media alone, so the fixture stays short.
    """
    processor = _FakeQwenProcessor()
    encoder = _encoder(_adapter(processor))
    sample = _sample()

    energon = encoder.batch([_adapter(processor).encode(sample)])
    hf = rl_collate_fn(
        [
            sft_processor(
                {"task_name": "sft", "messages": _hf_messages(sample)},
                TaskDataSpec(),
                processor,
                max_seq_length=1024,
                idx=0,
                add_bos=False,
                add_eos=False,
            )
        ]
    )

    kwargs = dict(
        tokenizer=processor,
        only_unmask_final=False,
        make_sequence_length_divisible_by=8,
    )
    a = prepare_sft_batch(energon, **kwargs)
    b = prepare_sft_batch(hf, **kwargs)

    for key in ("input_ids", "token_mask", "input_lengths", "sample_mask"):
        assert torch.equal(a[key], b[key]), key


def test_prepare_sft_batch_builds_mask_from_energon_message_log():
    processor = _FakeQwenProcessor()
    adapter = _adapter(processor)
    encoder = _encoder(adapter)
    encoded = adapter.encode(_sample())

    batch = encoder.batch([encoded])
    prepared = prepare_sft_batch(
        batch,
        tokenizer=processor,
        only_unmask_final=False,
        make_sequence_length_divisible_by=8,
    )

    expected_ids = torch.cat([message["token_ids"] for message in encoded.message_log])
    expected_mask = torch.cat(
        [
            torch.full_like(message["token_ids"], int(message["role"] == "assistant"))
            for message in encoded.message_log
        ]
    )
    assert torch.equal(prepared["input_ids"][0, : encoded.length], expected_ids)
    assert torch.equal(prepared["token_mask"][0, : encoded.length], expected_mask)
    assert prepared["input_ids"].shape[1] % 8 == 0


def test_nano_adapter_keeps_placeholder_side_tensors():
    encoded = _adapter(NemotronH_Nano_Omni_Reasoning_V3Processor()).encode(_sample())

    flat = message_log_to_flat_messages(encoded.message_log)
    assert isinstance(flat["pixel_values"], PackedTensor)
    assert isinstance(flat["imgs_sizes"], PackedTensor)
    assert isinstance(flat["num_frames"], PackedTensor)


def test_task_encoder_batches_heterogeneous_message_logs():
    processor = _FakeQwenProcessor()
    adapter = _adapter(processor)
    encoder = _encoder(adapter)
    assert encoder.decoder.config()["image_decode"] == "pilrgb"

    encoded = adapter.encode(_sample())
    text_only = deepcopy(encoded)
    for message in text_only.message_log:
        message["content"] = "text only"
        for key in ("pixel_values", "image_grid_thw"):
            message.pop(key, None)

    batch = encoder.batch([encoded, text_only])
    prepared = prepare_sft_batch(
        batch,
        tokenizer=processor,
        only_unmask_final=False,
        make_sequence_length_divisible_by=8,
    )

    assert list(batch) == ["message_log", "loss_multiplier"]
    assert prepared["input_ids"].shape[0] == 2
    assert len(prepared["pixel_values"]) == 2
    assert prepared["pixel_values"].tensors[1] is None


def test_task_encoder_runs_split_encode_and_batch_lifecycle_methods():
    adapter = _adapter(_FakeQwenProcessor())
    encoder = _encoder(adapter, include_source_ids=True)

    preencoded = encoder.preencode_sample(_sample())
    postencoded = encoder.postencode_sample(preencoded)
    batch = encoder.batch([postencoded])

    assert encoder.encode_batch(batch) is batch
    assert batch["source_ids"] == ["sample-0"]
    # Stage 1 does not override select_samples_to_pack, so this falls through to
    # Energon's base implementation.
    with pytest.raises(
        NotImplementedError, match="Packing only effective when overridden"
    ):
        encoder.select_samples_to_pack([preencoded])


class _FakeLoader:
    def __init__(self):
        self.restored = None

    def __iter__(self):
        return iter([{"value": 1}])

    def __len__(self):
        return 1

    def save_state_rank(self):
        return {"next": 1}

    def restore_state_rank(self, state):
        self.restored = state


def test_loader_state_is_fingerprinted_and_restored_before_iteration():
    identity = {"split_role": "train", "loader": {"num_workers": 4}}
    raw_loader = _FakeLoader()
    loader = EnergonSFTDataLoader(raw_loader, identity=identity)
    state = loader.state_dict()
    checkpoint = io.BytesIO()
    torch.save(state, checkpoint)
    checkpoint.seek(0)
    # The identity travels through a weights-only load with the rest of the
    # checkpoint, so it may hold JSON containers and scalars only.
    state = torch.load(checkpoint, weights_only=True)

    restored_raw_loader = _FakeLoader()
    restored = EnergonSFTDataLoader(restored_raw_loader, identity=identity)
    restored.load_state_dict(state)

    assert restored_raw_loader.restored == {"next": 1}
    assert list(restored) == [{"value": 1}]
    with pytest.raises(RuntimeError, match="before iteration"):
        restored.load_state_dict(state)


def test_rejected_restore_names_the_settings_that_changed():
    state = EnergonSFTDataLoader(
        _FakeLoader(),
        identity={"split_role": "train", "loader": {"num_workers": 4}},
    ).state_dict()

    mismatched = EnergonSFTDataLoader(
        _FakeLoader(),
        identity={"split_role": "validation", "loader": {"num_workers": 2}},
    )
    with pytest.raises(ValueError) as failure:
        mismatched.load_state_dict(state)
    assert "loader.num_workers 4 -> 2" in str(failure.value)
    assert "split_role 'train' -> 'validation'" in str(failure.value)

    # A state whose identity is missing still fails, with the generic reason.
    del state["identity"]
    with pytest.raises(ValueError, match="dataset, processor, or loader settings"):
        EnergonSFTDataLoader(
            _FakeLoader(), identity={"split_role": "validation"}
        ).load_state_dict(state)


def test_energon_config_disables_sequence_packing():
    config = EnergonLoaderConfig(model_family="qwen")
    assert config.model_family == "qwen"
    assert config.packing_buffer_size is None
    assert config.max_samples_per_sequence is None
    assert config.processor_adapter == "hf_multimodal"
    assert config.topology_mapper == "default"
    assert config.task_encoder.name == "generic_sft"
    assert [cooker.name for cooker in config.cookers] == ["generic_conversation"]

    source = EnergonSourceConfig(
        path="/data/prepared", split="train", virtual_epoch_length=10
    )
    assert source.virtual_epoch_length == 10
    with pytest.raises(ValueError):
        EnergonLoaderConfig(model_family="qwen", packing_buffer_size=10)
    with pytest.raises(ValueError):
        EnergonLoaderConfig(model_family="qwen", max_samples_per_sequence=2)
    with pytest.raises(ValueError):
        EnergonLoaderConfig.model_validate({})
    with pytest.raises(ValueError):
        EnergonLoaderConfig.model_validate({"model_family": "unsupported"})


def _identity(
    *,
    loader_config: EnergonLoaderConfig | None = None,
    batch_size: int = 8,
    shuffle: bool | None = True,
    logical_rank: int = 0,
) -> dict:
    config = loader_config or EnergonLoaderConfig(model_family="qwen")
    return _loader_identity(
        source=EnergonSourceConfig(
            path="/data/prepared", split="train", virtual_epoch_length=10
        ),
        loader_config=config,
        adapter_fingerprint="same-processor",
        split_role="train",
        batch_size=batch_size,
        shuffle=shuffle,
        topology={
            "mapper": config.topology_mapper,
            "placement": "same-placement",
            "logical_rank": logical_rank,
            "logical_world_size": 2,
        },
    )


def test_identity_pins_the_whole_loader_config_and_component_selection():
    generic = EnergonLoaderConfig(model_family="qwen")
    subflavored = EnergonLoaderConfig.model_validate(
        {
            "model_family": "qwen",
            "cookers": [
                {
                    "name": "generic_conversation",
                    "has_subflavors": {"cook": "conversation"},
                }
            ],
        }
    )

    assert _identity()["loader"] == generic.model_dump(mode="json")
    assert _identity()["registries"]
    assert _identity()["state_format_version"] == 2

    fingerprints = {
        _identity_fingerprint(_identity(loader_config=config))
        for config in (generic, subflavored)
    }
    assert len(fingerprints) == 2


def test_identity_refuses_a_changed_batch_size_or_shuffle():
    # Energon cannot rescale a restored offset through GroupBatchDataset and
    # data.shuffle also reorders shard slices, so either change would resume
    # mid-stream and silently replay samples.
    baseline = _identity()

    assert baseline["batch_size"] == 8
    assert baseline["shuffle"] is True
    for changed in (_identity(batch_size=4), _identity(shuffle=False)):
        assert _identity_fingerprint(changed) != _identity_fingerprint(baseline)


def test_train_loader_rejects_shuffle_false():
    # get_train_dataset shards by slice and Energon asserts a single slice
    # iterator when it does not shuffle over epochs, so shuffle=false is not a
    # valid training config. Reject it here instead of failing on the first
    # sample inside an Energon worker.
    with pytest.raises(ValueError, match="requires shuffle=true"):
        build_energon_sft_loader(
            data_config={"energon": {"model_family": "qwen"}, "shuffle": False},
            source=EnergonSourceConfig(
                path="/data/prepared", split="train", virtual_epoch_length=10
            ),
            processor=_FakeQwenProcessor(),
            batch_size=2,
            max_sequence_length=128,
            split_role="train",
            logical_rank=0,
            logical_world_size=1,
            placement_fingerprint="same-placement",
        )


def test_identity_binds_the_loader_to_one_logical_shard():
    # Two shards of one run must not accept each other's state.
    assert _identity_fingerprint(_identity(logical_rank=0)) != _identity_fingerprint(
        _identity(logical_rank=1)
    )


def test_config_parses_registry_keys_and_validates_packing_options():
    config = EnergonLoaderConfig.model_validate(
        {
            "model_family": "qwen",
            "task_encoder": "generic_sft",
            "cookers": ["generic_conversation"],
        }
    )
    assert config.task_encoder.name == "generic_sft"
    assert config.cookers[0].name == "generic_conversation"


def test_config_rejects_options_for_the_generic_task_encoder():
    with pytest.raises(ValueError, match="has no configurable options"):
        _loader_config(
            {
                "model_family": "qwen",
                "task_encoder": {
                    "name": "generic_sft",
                    "options": {"patch_dim": 14},
                },
            }
        )

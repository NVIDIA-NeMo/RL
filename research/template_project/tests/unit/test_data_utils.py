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
"""Unit tests for template_project.data_utils."""

import pytest
import torch
from template_project.data_utils import create_batch_from
from transformers import AutoTokenizer


@pytest.fixture
def tokenizer():
    """Fixture to create a tokenizer with proper padding token."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def test_create_batch_from_single_sentence(tokenizer):
    """Test create_batch_from with a single sentence."""
    sentences = ["Hello world"]

    batch = create_batch_from(tokenizer, sentences)

    assert "input_ids" in batch
    assert "input_lengths" in batch
    assert "token_mask" in batch
    assert "sample_mask" in batch
    assert batch["input_ids"].shape[0] == 1
    assert batch["input_lengths"].shape[0] == 1
    assert batch["sample_mask"].shape[0] == 1


def test_create_batch_from_multiple_sentences(tokenizer):
    """Test create_batch_from with multiple sentences."""
    sentences = ["Hello world", "This is a test", "Another sentence here"]

    batch = create_batch_from(tokenizer, sentences)

    assert batch["input_ids"].shape[0] == 3
    assert batch["input_lengths"].shape[0] == 3
    assert batch["sample_mask"].shape[0] == 3
    assert batch["token_mask"].shape == batch["input_ids"].shape


def test_create_batch_from_padding(tokenizer):
    """Test that create_batch_from correctly pads sequences."""
    sentences = ["short", "this is a much longer sentence"]

    batch = create_batch_from(tokenizer, sentences)

    # All sequences should have the same length (padded to max)
    assert batch["input_ids"].shape[1] == batch["input_ids"].shape[1]
    # Input lengths should reflect the actual (unpadded) lengths
    assert batch["input_lengths"][0] < batch["input_lengths"][1]


def test_create_batch_from_dtypes(tokenizer):
    """Test that create_batch_from produces correct data types."""
    sentences = ["Hello world"]

    batch = create_batch_from(tokenizer, sentences)

    assert batch["input_ids"].dtype == torch.long
    assert batch["input_lengths"].dtype == torch.int32
    assert batch["sample_mask"].dtype == torch.float32
    assert batch["token_mask"].dtype == torch.long


def test_create_batch_from_sample_mask_all_ones(tokenizer):
    """Test that sample_mask is all ones."""
    sentences = ["Hello", "World", "Test"]

    batch = create_batch_from(tokenizer, sentences)

    assert torch.all(batch["sample_mask"] == 1.0)
    assert batch["sample_mask"].shape[0] == len(sentences)


def test_create_batch_from_token_mask_all_ones(tokenizer):
    """Test that token_mask is all ones."""
    sentences = ["Hello world", "Test sentence"]

    batch = create_batch_from(tokenizer, sentences)

    assert torch.all(batch["token_mask"] == 1)
    assert batch["token_mask"].shape == batch["input_ids"].shape


def test_create_batch_from_input_lengths(tokenizer):
    """Test that input_lengths correctly represent non-padded token counts."""
    sentences = ["a b c", "x y"]

    batch = create_batch_from(tokenizer, sentences)

    # Compute expected lengths from attention mask
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    enc = tok(sentences, add_special_tokens=False, return_tensors="pt", padding=True)
    expected_lengths = enc["attention_mask"].sum(dim=1).to(torch.int32)

    assert torch.all(batch["input_lengths"] == expected_lengths)

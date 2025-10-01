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

import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor


class SmolVLMVisionEmbeddingsReference(nn.Module):
    """
    Previous (correct) implementation in transformers<=4.54.1. Copied from https://github.com/huggingface/transformers/blob/4.54.1/src/transformers/models/smolvlm/modeling_smolvlm.py#L101-L156

    Remove this test once upstream bug is fixed.
    """

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(
        self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor
    ) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = (
            max_im_h // self.patch_size,
            max_im_w // self.patch_size,
        )
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(
                fractional_coords_h, boundaries, right=True
            )
            bucket_coords_w = torch.bucketize(
                fractional_coords_w, boundaries, right=True
            )

            pos_ids = (
                bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w
            ).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


def test_smolvlm_embeddings_differ_from_reference():
    # Remove once https://github.com/huggingface/transformers/issues/41190 is fixed and adopted.

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    model = model.to(device)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": "Can you describe this image?"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    inputs = {
        k: v.to(dtype=torch.bfloat16)
        if isinstance(v, torch.Tensor) and v.is_floating_point()
        else v
        for k, v in inputs.items()
    }

    patch_size = model.model.vision_model.patch_size
    pixel_values = inputs["pixel_values"]  # (bsz, num_images, 3, H, W)
    bsz, num_images, _, H, W = pixel_values.shape
    pixel_values = pixel_values.view(bsz * num_images, *pixel_values.shape[2:])

    patch_attention_mask = torch.ones(
        (
            bsz,
            pixel_values.size(2) // patch_size,
            pixel_values.size(3) // patch_size,
        ),
        device=pixel_values.device,
        dtype=torch.bool,
    )

    # Get buggy/current embeddings module from installed transformers
    embeddings_buggy = model.model.vision_model.embeddings

    with torch.no_grad():
        out_buggy = embeddings_buggy(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
        )

    # Build reference embeddings and copy weights for apples-to-apples comparison
    ref = SmolVLMVisionEmbeddingsReference(model.model.vision_model.config)
    ref = ref.to(device=device, dtype=torch.bfloat16)

    # Copy the conv and embedding weights
    ref.patch_embedding.load_state_dict(embeddings_buggy.patch_embedding.state_dict())
    ref.position_embedding.load_state_dict(
        embeddings_buggy.position_embedding.state_dict()
    )

    with torch.no_grad():
        out_ref = ref(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
        )

    # Assert outputs differ due to the upstream bug
    are_equal = torch.allclose(out_buggy.float(), out_ref.float(), atol=0, rtol=0)
    assert not are_equal, (
        "If this fails, that means the upstream bug has been fixed. You can close this issue: https://github.com/huggingface/transformers/issues/41190"
    )

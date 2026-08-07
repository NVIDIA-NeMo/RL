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
#!/usr/bin/env python3

"""Verifies the convert_lora_to_hf.py CLI merge-path output.

Checks that the merged model has the same key structure as the original HF
model, and that at least one weight differs (confirming the LoRA adapter was
actually folded in).

Usage:
    python tests/functional/_cli_test_verify_lora_merge.py <model_name> <hf_output_path>
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tests.functional.converter_test_utils import get_model_state_dict

model_name, out_path = sys.argv[1], sys.argv[2]

orig = get_model_state_dict(
    AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
)
merged = get_model_state_dict(
    AutoModelForCausalLM.from_pretrained(
        out_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
)

assert set(merged.keys()) == set(orig.keys()), (
    f"Key mismatch.\n"
    f"  Extra:   {set(merged.keys()) - set(orig.keys())}\n"
    f"  Missing: {set(orig.keys()) - set(merged.keys())}"
)

any_diff = any(
    isinstance(orig[k], torch.Tensor)
    and not torch.allclose(orig[k], merged[k], rtol=1e-5, atol=1e-5)
    for k in orig
)
assert any_diff, "LoRA-merged output is identical to original — adapter was not applied"

print(
    "✓ convert_lora_to_hf CLI (merge) output has correct key structure and differs from original"
)

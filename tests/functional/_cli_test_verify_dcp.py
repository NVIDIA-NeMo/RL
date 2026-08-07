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

"""Verifies that the convert_dcp_to_hf.py CLI output matches the original HF model.

Usage:
    python tests/functional/_cli_test_verify_dcp.py <model_name> <hf_output_path>
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tests.functional.converter_test_utils import (
    assert_state_dicts_equal,
    get_model_state_dict,
)

model_name, out_path = sys.argv[1], sys.argv[2]

orig = get_model_state_dict(
    AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
)
conv = get_model_state_dict(
    AutoModelForCausalLM.from_pretrained(
        out_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
)
assert_state_dicts_equal(orig, conv, "original", "cli_dcp_to_hf")
print("✓ convert_dcp_to_hf CLI output matches original HF model")

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

"""Stable logical sampling identities, independent of dispatch order."""

import hashlib


def derive_sampling_seed(seed: int, *identity: int) -> int:
    """Derive a nonnegative int64 seed from a run seed and integer identity."""
    payload = ":".join(str(int(value)) for value in (seed, *identity))
    digest = hashlib.sha256(f"nemo-rl-sampling-v1:{payload}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

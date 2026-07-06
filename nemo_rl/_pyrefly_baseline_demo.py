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

"""Demonstration of the pyrefly baseline gate — NOT for merge.

This file is added by a demo PR to show that `pyrefly check --baseline
pyrefly-baseline.json` fails on a NEW type error (one not present in the
committed baseline snapshot), both in pre-commit and in CI. The follow-up
commit fixes the error so the check goes green again.
"""


def add_one(x: int) -> str:
    # Intentional bug for the demo: this returns an int, but the signature
    # promises str. pyrefly flags this as a new error (not in the baseline).
    return x + 1

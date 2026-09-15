# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Parse the first valid judge verdict without treating missing output as false."""


def first_verdict(text: str, *, equal_label: str, not_equal_label: str) -> bool:
    if not equal_label or not not_equal_label or equal_label == not_equal_label:
        raise ValueError("Judge verdict labels must be distinct and nonempty")
    equal = text.find(equal_label)
    unequal = text.find(not_equal_label)
    if equal < 0 and unequal < 0:
        raise ValueError("Judge response contains no valid verdict")
    return equal >= 0 and (unequal < 0 or equal < unequal)

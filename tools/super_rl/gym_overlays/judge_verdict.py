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
"""Accept only complete, unambiguous judge verdicts as rewards."""

import json
from typing import Any


def first_verdict(text: str, *, equal_label: str, not_equal_label: str) -> bool:
    """Return the sole verdict kind, rejecting missing or conflicting labels."""
    if not equal_label or not not_equal_label or equal_label == not_equal_label:
        raise ValueError("Judge verdict labels must be distinct and nonempty")
    equal = text.find(equal_label)
    unequal = text.find(not_equal_label)
    if equal < 0 and unequal < 0:
        raise ValueError("Judge response contains no valid verdict")
    if equal >= 0 and unequal >= 0:
        raise ValueError("Judge response contains conflicting verdicts")
    return equal >= 0


def response_verdict(response: Any, *, equal_label: str, not_equal_label: str) -> bool:
    """Keep failure shape and token accounting, never the judge's full answer."""
    status = getattr(response, "status", None)
    incomplete = getattr(response, "incomplete_details", None)
    try:
        if status != "completed" or incomplete is not None:
            raise ValueError("Judge response is not complete")
        return first_verdict(
            response.output_text,
            equal_label=equal_label,
            not_equal_label=not_equal_label,
        )
    except ValueError as error:
        usage = getattr(response, "usage", None)
        details = {
            "status": status,
            "incomplete_reason": getattr(incomplete, "reason", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "output_text_chars": len(response.output_text),
        }
        raise ValueError(f"{error}; {json.dumps(details)}") from error

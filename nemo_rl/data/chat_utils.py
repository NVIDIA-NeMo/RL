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

"""Validation shared by chat dataset adapters, processors, and collators."""

from numbers import Integral
from typing import Any, Sequence


def normalize_message_loss_mask(
    messages: Sequence[dict[str, Any]],
    message_loss_mask: Sequence[int] | None,
) -> list[int]:
    """Validate turn selection, defaulting to all assistant messages.

    A zero excludes the message from supervision; it never removes the message
    from the rendered conversation. Supplied masks must have one binary integer
    per message and may select only assistant turns.
    """
    if message_loss_mask is None:
        return [int(message.get("role") == "assistant") for message in messages]
    if not isinstance(message_loss_mask, (list, tuple)):
        raise TypeError(
            "message_loss_mask must be a list or tuple aligned with messages, got "
            f"{type(message_loss_mask).__name__}"
        )
    if len(message_loss_mask) != len(messages):
        raise ValueError(
            "message_loss_mask must have one entry per message, got "
            f"{len(message_loss_mask)} entries for {len(messages)} messages"
        )

    normalized: list[int] = []
    for message_i, (message, raw_value) in enumerate(zip(messages, message_loss_mask)):
        value: object = raw_value
        if not isinstance(value, Integral) or int(value) not in (0, 1):
            raise ValueError(
                "message_loss_mask entries must be binary integers; "
                f"entry {message_i} is {value!r}"
            )
        include = int(value)
        if include and message.get("role") != "assistant":
            raise ValueError(
                "message_loss_mask may select only assistant messages; "
                f"entry {message_i} has role {message.get('role')!r}"
            )
        normalized.append(include)
    return normalized

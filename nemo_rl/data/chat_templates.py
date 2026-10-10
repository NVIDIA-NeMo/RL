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

## a reference to frequently used chat templates for convenience
class COMMON_CHAT_TEMPLATES:
    ### simple template which prepends a role header to the content
    simple_role_header = "{% for message in messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    ### passthrough template which just concatenates the content of the messages with no special tokens
    passthrough_prompt_response = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )


def find_rendered_message_content_span(
    text: str,
    content: str,
    cursor: int = 0,
) -> tuple[int, int, str] | None:
    """Locate the same canonical nonempty content key on every model surface.

    Trimming the logical source *before* searching ensures templates preserving
    and trimming boundary whitespace share one coordinate origin. Callers must
    anchor ``cursor`` and the search surface to the corresponding logical turn.
    Interior whitespace is never changed.
    """
    canonical = content.strip()
    if not canonical:
        return None
    position = text.find(canonical, cursor)
    if position < 0:
        return None
    return position, position + len(canonical), canonical

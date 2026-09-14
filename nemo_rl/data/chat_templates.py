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
    rendered: str,
    probe_rendered: str,
    sentinel: str,
) -> tuple[int, int] | None:
    """Map one sentinel-probed message-content insertion onto a real render.

    ``probe_rendered`` must be produced from the same complete conversation as
    ``rendered``, with exactly one message's content replaced by ``sentinel``.
    The text surrounding that unique sentinel structurally anchors the content
    insertion, so content repeated in role headers, tools, or other turns cannot
    be mistaken for the target. The returned span covers the template-rendered
    content, including any escaping and excluding whitespace removed by the
    template. ``None`` means the probe was ambiguous or changed text outside the
    target insertion.
    """
    if probe_rendered.count(sentinel) != 1:
        return None

    probe_start = probe_rendered.index(sentinel)
    probe_end = probe_start + len(sentinel)
    prefix = probe_rendered[:probe_start]
    suffix = probe_rendered[probe_end:]
    if not rendered.startswith(prefix) or (suffix and not rendered.endswith(suffix)):
        return None

    content_start = len(prefix)
    content_end = len(rendered) - len(suffix) if suffix else len(rendered)
    if content_end < content_start:
        return None
    return content_start, content_end

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

"""Judge prompt text for the GenRM / DWRL VLM runners (`examples/run_dwrl_vlm.py`, `examples/run_dwrl_pairwise_vlm.py`,
`examples/run_genrm_pairwise_vlm.py`) and the `examples/run_eval_genrm_dwrl_openai*.py` eval scripts: the prompt
templates, their default principles, `flatten_to_single_turn`, which renders the conversation as the templates'
`{context}`, `build_openai_judge_content`, which turns a rendered prompt with media into OpenAI chat content parts,
and `build_bt_turn`, the BT question turn that DWRL training appends to the judge's generation.

The templates are `str.format` strings (literal braces doubled). The default principle fills `{principle}` when a
datum has none.

The eval scripts run in a slim OpenAI-client environment, so keep this module's imports light (no vLLM, Ray or
Megatron); media decoding and tokenization belong in `genrm_multimodal_utils`."""

import re

from nemo_rl.data.multimodal_utils import AUDIO_CONTENT_TYPES, IMAGE_CONTENT_TYPES, VIDEO_CONTENT_TYPES

# Pointwise (run_dwrl_vlm.py): fields {context}, {response}, {principle}.
DWRL_POINTWISE_RUBRIC_TEMPLATE = """You are an expert evaluation judge specializing in the assessment of LLM responses. You are impartial, rigorous, and consistent. Given the conversation context and a single assistant response to the user's latest query, you will follow the evaluation plan and scoring guidelines exactly as written below.

#### Conversation Context ####
{context}

#### Response to be Scored ####
[The Start of Response]
{response}
[The End of Response]

#### Evaluation Plan ####
{principle}

#### Scoring Guidelines ####
Based on the evaluation plan above, assign a score using these scales:

**Individual Helpfulness Scores (1-5):**
- 5: Extremely Helpful - Completely aligned with what the user was asking for
- 4: Mostly Helpful - Generally useful with minor room for improvement
- 3: Partially Helpful - Misses the overall goal in some way
- 2: Borderline Unhelpful - Mostly doesn't capture what the user wanted
- 1: Not Helpful - Completely missed the essence of the request

#### Output Format ####
First, evaluate each rubric individually. Then synthesize an overall judgment.
Provide your judgment as JSON:
```json
{{
    "rubric_evaluations": [
        {{
            "rubric_id": 1,
            "response_analysis": "Your analysis of the Response for this rubric",
            "score_1": <1-5>
        }}
    ],
    "overall": {{
        "response_analysis": "Your overall analysis of the Response considering all rubrics",
        "score_1": <1-5>
    }}
}}
```"""

GENRM_POINTWISE_DEFAULT_PRINCIPLE = """You will be given one or more evaluation criteria (rubrics).
Evaluate the Response on EACH criterion individually first, then synthesize an overall judgment.
Criteria:

1. Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide your answer before judging any answers. When evaluating the assistant's response, compare the assistant's response with your answer. You must identify and correct any mistakes or inaccurate information. Then consider if the assistant's response is helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when the user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user rather than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Then consider the creativity and novelty of the assistant's response when needed. Finally, identify any missing important information in the assistant's response which would be beneficial to include when responding to the user prompt."""

# Pairwise (run_dwrl_pairwise_vlm.py, run_genrm_pairwise_vlm.py): fields {context}, {response_1}, {response_2}, {principle}.
DWRL_AND_GENRM_PAIRWISE_RUBRIC_TEMPLATE = """You are an expert evaluation judge specializing in comparative assessment of LLM responses. You are impartial, rigorous, and consistent. Given the conversation context and two assistant responses to the user's latest query, you will follow the evaluation plan and scoring guidelines exactly as written below.

#### Conversation Context ####
{context}

#### Responses to be Scored ####
[The Start of Response 1]
{response_1}
[The End of Response 1]

[The Start of Response 2]
{response_2}
[The End of Response 2]

#### Evaluation Plan ####
{principle}

#### Scoring Guidelines ####
Based on the evaluation plan above, assign scores using these scales:

**Individual Helpfulness Scores (1-5):**
- 5: Extremely Helpful - Completely aligned with what the user was asking for
- 4: Mostly Helpful - Generally useful with minor room for improvement
- 3: Partially Helpful - Misses the overall goal in some way
- 2: Borderline Unhelpful - Mostly doesn't capture what the user wanted
- 1: Not Helpful - Completely missed the essence of the request

**Comparative Ranking (1-6):**
- 1: Response 1 is much better than Response 2
- 2: Response 1 is better than Response 2
- 3: Response 1 is slightly better than Response 2
- 4: Response 2 is slightly better than Response 1
- 5: Response 2 is better than Response 1
- 6: Response 2 is much better than Response 1

#### Output Format ####
First, evaluate each rubric individually. Then synthesize an overall judgment.
Provide your judgment as JSON:
```json
{{
    "rubric_evaluations": [
        {{
            "rubric_id": 1,
            "response_1_analysis": "Your analysis of Response 1 for this rubric",
            "response_2_analysis": "Your analysis of Response 2 for this rubric",
            "score_1": <1-5>,
            "score_2": <1-5>,
            "ranking": <1-6>
        }}
    ],
    "overall": {{
        "response_1_analysis": "Your overall analysis of Response 1 considering all rubrics",
        "response_2_analysis": "Your overall analysis of Response 2 considering all rubrics",
        "score_1": <1-5>,
        "score_2": <1-5>,
        "ranking": <1-6>
    }}
}}
```"""

GENRM_PAIRWISE_DEFAULT_PRINCIPLE = """You will be given one or more evaluation criteria (rubrics).
Evaluate both responses on EACH criterion individually first, then synthesize an overall judgment.
Criteria:

1. Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide your answer before judging any answers. When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information. Then consider if the assistants' answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when the user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user rather than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Then consider the creativity and novelty of the assistants' answers when needed. Finally, identify any missing important information in the assistants' answers which would be beneficial to include when responding to the user prompt."""

# Placeholders understood by the Super 3.5 VL chat template and processor. A literal marker in the text is
# kept where it is (the template adds none of its own), and the processor expands it to <img><image>*N</img>.
IMAGE_MARKER = "<image>"
VIDEO_MARKER = "<video>"
# Sound placeholder of models with a sound encoder (Omni); `genrm_multimodal_utils` expands it to
# <so_start> + marker*N + <so_end>.
AUDIO_MARKER = "<so_embedding>"

def _media_marker(part_type: str) -> str:
    """Inline placeholder for a media content part (kept at the position it had in the conversation)."""
    if part_type in IMAGE_CONTENT_TYPES:
        return IMAGE_MARKER
    if part_type in VIDEO_CONTENT_TYPES:
        return VIDEO_MARKER
    if part_type in AUDIO_CONTENT_TYPES:
        return AUDIO_MARKER
    raise ValueError(f"unsupported content part type {part_type!r}")

def flatten_to_single_turn(message_log):
    """Flatten the conversation context into the text of the judge prompt.

    Media parts become inline `<image>` / `<video>` markers; the raw parts are returned, in order, for
    `extract_all_modals` to decode."""
    def strip_prefix(text: str) -> str:
        if text.lower().startswith("user:"):
            return text[5:].strip()
        if text.lower().startswith("assistant:"):
            return text[10:].strip()
        if text.lower().startswith("system:"):
            return text[7:].strip()

        return text

    ret = ""
    multimodal = []
    for idx, message in enumerate(message_log):
        if message["role"] == "system":
            ret += '\n[The Start of the System Prompt for this Conversation]\n' + strip_prefix(message["content"]) + '\n[The End of the System Prompt for this Conversation]' + "\n\n"
        elif message["role"] == "user":
            if isinstance(message["content"], str):
                ret += "User: " + strip_prefix(message["content"]) + "\n\n"
            else:
                for payload in message["content"]:
                    if payload["type"] == "text":
                        ret += "User: " + strip_prefix(payload["text"]) + "\n\n"
                    else:
                        ret += "User: " + _media_marker(payload["type"]) + "\n\n"
                        multimodal.append(payload)
        elif message["role"] == "assistant":
            #resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", message["content"]).strip()
            #resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
            resp_no_thinking = message["content"].split('</think>')[-1].lstrip()
            ret += "Assistant: " + strip_prefix(resp_no_thinking.strip()) + "\n\n"

    return ret.strip(), multimodal

_MARKER_PATTERN = re.compile("(" + "|".join(re.escape(m) for m in (IMAGE_MARKER, VIDEO_MARKER, AUDIO_MARKER)) + ")")

def media_part_url(part: dict) -> str:
    """The URL (usually a base64 data URL) of a media content part. It sits under a different key depending on the part
    type (image_url, image, input_image, audio_url, ...), either directly or as {"url": ...}."""
    value = next((part[key] for key in (part["type"], "url", "image_url", "image", "video_url", "video", "audio_url", "audio")
                  if part.get(key) is not None), None)
    if value is None:
        raise ValueError(f"no media payload found in content part {part['type']!r}")
    return value["url"] if isinstance(value, dict) else value

def _openai_media_part(part: dict) -> dict:
    """A conversation media part as an OpenAI chat-completions content part (vLLM decodes the URL and does the
    model-specific placeholder expansion)."""
    url = media_part_url(part)
    if part["type"] in IMAGE_CONTENT_TYPES:
        return {"type": "image_url", "image_url": {"url": url}}
    if part["type"] in VIDEO_CONTENT_TYPES:
        return {"type": "video_url", "video_url": {"url": url}}
    if part["type"] in AUDIO_CONTENT_TYPES:
        return {"type": "audio_url", "audio_url": {"url": url}}
    raise ValueError(f"unsupported content part type {part['type']!r}")

def build_openai_judge_content(template: str, context, **fields):
    """The judge prompt as the `content` of an OpenAI chat-completions user message (for the eval scripts' vLLM server).

    `context` is the conversation (a message list, flattened with `flatten_to_single_turn`) or already flat text; `fields`
    fill the rest of `template`. Without media the content is the prompt string. With media it is the prompt text, its
    markers laid out as in training (`build_user_message`), followed by the media parts: Super's chat template adds no
    markers of its own when the text already has them, so the model sees the training prompt, and vLLM matches the
    parts to the markers in order."""
    context, media = flatten_to_single_turn(context) if isinstance(context, list) else (context, [])
    prompt = template.format(context=context, **fields)
    if not media:
        return prompt
    num_markers = len(_MARKER_PATTERN.findall(context))
    if num_markers != len(media):  # e.g. a text part that itself contains a literal "<image>"
        raise ValueError(f"the context has {num_markers} media markers for {len(media)} media parts")
    if VIDEO_MARKER in prompt:
        # training puts the video marker on its own line
        before, _, after = prompt.partition(VIDEO_MARKER)
        prompt = "\n".join([before, VIDEO_MARKER, after])
    return [{"type": "text", "text": prompt}] + [_openai_media_part(part) for part in media]

def build_bt_turn(tokenizer, bt_prompt: str) -> str:
    """The DWRL BT question turn that `dwrl.py` / `dwrl_pairwise.py` append to the judge's generation.

    The turn is `bt_prompt` rendered by the tokenizer's own chat template (thinking off). The generation ends with the
    eos token, so the turn starts with what that template puts between the end of an assistant turn and the next turn
    ("\\n" for ChatML templates such as Super 3.5's and Qwen3's): the conversation then reads exactly as a chat
    template renders it at inference (the eval scripts, Gym)."""
    chat_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    bt_message = {"role": "user", "content": bt_prompt}
    rendered = tokenizer.apply_chat_template([bt_message], tokenize=False, add_generation_prompt=True, add_special_tokens=False, enable_thinking=False)
    bt_turn = rendered.replace("<|im_start|>system\n<|im_end|>\n", "").replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "")
    # The same turn after an assistant turn: the text between the assistant's eos and the turn is the separator.
    end_of_assistant_turn = "ASSISTANT_TURN_PROBE" + chat_tokenizer.eos_token
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "ASSISTANT_TURN_PROBE"}, bt_message],
        tokenize=False, add_generation_prompt=True, add_special_tokens=False, enable_thinking=False,
    )
    if end_of_assistant_turn not in probe or not probe.endswith(bt_turn):
        raise ValueError(f"the chat template does not end an assistant turn with the eos token {chat_tokenizer.eos_token!r} "
                         f"and then render the BT turn {bt_turn!r} as on its own: {probe!r}")
    separator = probe[probe.index(end_of_assistant_turn) + len(end_of_assistant_turn):len(probe) - len(bt_turn)]
    return separator + bt_turn

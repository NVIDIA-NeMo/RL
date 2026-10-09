"""Install required-prefix-token processing on a vLLM OnlineRenderer."""

import asyncio
from copy import deepcopy
from types import MethodType
from typing import Any

from .prefix import replace_prefix_tokens

_PATCH_MARKER = "_nemo_rl_required_prefix_tokens_installed"


def _get_request_max_tokens(request: Any) -> int | None:
    if request.max_completion_tokens is not None:
        return request.max_completion_tokens
    return request.max_tokens


def _set_request_max_tokens(request: Any, max_tokens: int) -> None:
    if request.max_completion_tokens is not None:
        request.max_completion_tokens = max_tokens
    elif request.max_tokens is not None:
        request.max_tokens = max_tokens


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def install_renderer_patch(
    online_renderer: Any,
    validation_error_type: type[Exception],
    capture_runtime: Any | None = None,
) -> bool:
    """Wrap ``preprocess_chat`` once and return whether a patch was installed."""
    if getattr(online_renderer, _PATCH_MARKER, False):
        return False

    original_preprocess_chat = online_renderer.preprocess_chat

    async def preprocess_chat(
        renderer_self: Any,
        request: Any,
        messages: list[Any],
        default_template: str | None,
        default_template_content_format: Any,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        parser: Any = None,
        *,
        skip_mm_cache: bool = False,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        kv_transfer_params = getattr(request, "kv_transfer_params", None)
        reuses_prefill_prompt = isinstance(kv_transfer_params, dict) and bool(
            kv_transfer_params.get("prompt_token_ids")
        )
        admission = getattr(request, "ng_capture", None)
        capture_prefix_token_ids: list[int] = []
        if admission is not None:
            if capture_runtime is None:
                raise RuntimeError("ng_capture is unavailable on this vLLM endpoint")
            if not isinstance(admission, dict):
                admission = admission.model_dump(mode="json")
            prefill_prompt_token_ids = (
                list(kv_transfer_params["prompt_token_ids"])
                if reuses_prefill_prompt
                else None
            )
            capture_prefix_token_ids = await asyncio.to_thread(
                capture_runtime.resolve_prefix,
                admission,
                prefill_prompt_token_ids=prefill_prompt_token_ids,
            )
            if admission.get("mode") == "token_in":
                request.required_prefix_token_ids = capture_prefix_token_ids
        required_prefix_token_ids = getattr(request, "required_prefix_token_ids", None)
        messages_for_prefix_render = deepcopy(messages)
        actual_max_tokens = _get_request_max_tokens(request)

        # vLLM validates prompt + requested output length while rendering. The
        # final prompt length is only known after the token splice, so defer the
        # real clamp until then.
        if actual_max_tokens is not None:
            _set_request_max_tokens(request, 1)

        try:
            result = await original_preprocess_chat(
                request=request,
                messages=messages,
                default_template=default_template,
                default_template_content_format=default_template_content_format,
                default_template_kwargs=default_template_kwargs,
                tool_dicts=tool_dicts,
                parser=parser,
                skip_mm_cache=skip_mm_cache,
            )

            if required_prefix_token_ids is None or reuses_prefill_prompt:
                final_prompt_token_ids = result[1][0]["prompt_token_ids"]
            else:
                last_assistant_index = None
                for index in reversed(range(len(messages_for_prefix_render))):
                    if _message_role(messages_for_prefix_render[index]) == "assistant":
                        last_assistant_index = index
                        break

                if last_assistant_index is None:
                    prefix_messages = messages_for_prefix_render
                else:
                    prefix_messages = messages_for_prefix_render[
                        : last_assistant_index + 1
                    ]

                prefix_request = request.model_copy(
                    update={"add_generation_prompt": False}
                )
                prefix_result = await original_preprocess_chat(
                    request=prefix_request,
                    messages=prefix_messages,
                    default_template=default_template,
                    default_template_content_format=default_template_content_format,
                    default_template_kwargs=default_template_kwargs,
                    tool_dicts=tool_dicts,
                    parser=parser,
                    skip_mm_cache=skip_mm_cache,
                )
                template_prefix_token_ids = prefix_result[1][0]["prompt_token_ids"]
                engine_prompt = result[1][0]
                final_prompt_token_ids = replace_prefix_tokens(
                    tokenizer=renderer_self.renderer.tokenizer,
                    model_prefix_token_ids=list(required_prefix_token_ids),
                    template_prefix_token_ids=template_prefix_token_ids,
                    template_token_ids=engine_prompt["prompt_token_ids"],
                )
                engine_prompt["prompt_token_ids"] = final_prompt_token_ids

            if actual_max_tokens is not None:
                remaining_tokens = renderer_self.model_config.max_model_len - len(
                    final_prompt_token_ids
                )
                if remaining_tokens <= 0:
                    message = (
                        f"Prompt length ({len(final_prompt_token_ids)}) fills or "
                        "exceeds this model's maximum context length "
                        f"({renderer_self.model_config.max_model_len}). No room "
                        "for output tokens."
                    )
                    raise validation_error_type(
                        message,
                        parameter="input_tokens",
                        value=len(final_prompt_token_ids),
                    )
                _set_request_max_tokens(
                    request, min(actual_max_tokens, remaining_tokens)
                )

            is_pd_prefill = (
                isinstance(kv_transfer_params, dict)
                and bool(kv_transfer_params.get("do_remote_decode"))
                and not reuses_prefill_prompt
            )
            if admission is not None and not is_pd_prefill:
                capture_runtime.record_prompt(
                    request,
                    admission=admission,
                    prefix_token_ids=capture_prefix_token_ids,
                    prompt_token_ids=final_prompt_token_ids,
                )

            return result
        except BaseException:
            if actual_max_tokens is not None:
                _set_request_max_tokens(request, actual_max_tokens)
            raise

    online_renderer.preprocess_chat = MethodType(preprocess_chat, online_renderer)
    setattr(online_renderer, _PATCH_MARKER, True)
    return True

"""Custom CrewAI-compatible LLM wrapper for local Hugging Face models."""

from __future__ import annotations

from typing import Any, Iterable

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel


class HuggingFaceLocalLLM(BaseLLM):
    """Thin adapter over a transformers pipeline that satisfies CrewAI's BaseLLM."""

    def __init__(
        self,
        *,
        model_name: str,
        generation_pipeline,
        tokenizer,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        super().__init__(
            model=model_name,
            temperature=temperature,
            provider="huggingface",
        )
        self._pipeline = generation_pipeline
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._top_p = top_p

    def supports_stop_words(self) -> bool:
        """The transformers pipeline does not currently respect stop words."""

        return False

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
    ) -> str:
        formatted_messages = self._format_messages(messages)
        prompt = self._build_prompt(formatted_messages)
        self._emit_call_started_event(
            formatted_messages,
            tools,
            callbacks,
            available_functions,
            from_task,
            from_agent,
        )

        prompt_tokens = self._token_count(prompt)

        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=self._max_new_tokens,
                temperature=self.temperature,
                top_p=self._top_p,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                return_full_text=False,
            )

            if not outputs:
                raise RuntimeError("Pipeline returned no completions")

            completion = outputs[0]["generated_text"].strip()
            completion = self._maybe_parse_structured_output(
                completion,
                response_model,
            )

            if isinstance(completion, str):
                completion_tokens = self._token_count(completion)
            else:
                completion_tokens = self._token_count(str(completion))

            self._track_token_usage_internal(
                {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )

            self._emit_call_completed_event(
                response=completion,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=formatted_messages,
            )

            if tools and available_functions:
                # The lightweight local model does not support tool calling yet.
                pass

            if isinstance(completion, BaseModel):
                return completion.model_dump_json()

            return completion

        except Exception as exc:  # noqa: BLE001
            self._emit_call_failed_event(str(exc), from_task=from_task, from_agent=from_agent)
            raise

    def _maybe_parse_structured_output(
        self,
        completion: str,
        response_model: Any | None,
    ) -> str | Any:
        if response_model is None:
            return completion
        return self._validate_structured_output(completion, response_model)

    def _token_count(self, text: str) -> int:
        encoded = self._tokenizer(text, add_special_tokens=False)
        token_ids = encoded.get("input_ids", [])
        return len(token_ids)

    def _build_prompt(self, messages: Iterable[dict[str, Any]]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001
                pass

        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role.capitalize()}: {content}\n")
        parts.append("Assistant:")
        return "".join(parts)

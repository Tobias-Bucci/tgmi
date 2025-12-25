"""HTTP client abstraction for the Gemini API."""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

import requests

from .constants import SUGGESTED_MODELS


class GeminiClient:
    """Encapsulate HTTP calls to the Google Gemini API."""

    API_BASES = (
        "https://generativelanguage.googleapis.com/v1",
        "https://generativelanguage.googleapis.com/v1beta",
    )

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: int,
        max_output_tokens_retry: Optional[int] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = self.API_BASES[0]
        self.timeout = timeout
        self.max_output_tokens_retry = max_output_tokens_retry

    def send_message(
        self,
        history: List[Dict[str, object]],
        message: str,
        *,
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        allow_max_tokens_retry: bool = True,
        google_search: bool = False,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        contents = [*history]
        contents.append({"role": "user", "parts": [{"text": message}]})

        payload: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["system_instruction"] = {
                "role": "system",
                "parts": [{"text": system_instruction}],
            }
        if generation_config:
            payload["generationConfig"] = copy.deepcopy(generation_config)
        
        if google_search:
            payload["tools"] = [{"googleSearch": {}}]

        bases_to_try = [self.api_base] + [base for base in self.API_BASES if base != self.api_base]
        if system_instruction or google_search:
            beta_base = self.API_BASES[-1]
            if beta_base in bases_to_try:
                bases_to_try.remove(beta_base)
            bases_to_try.insert(0, beta_base)
        not_found_messages: List[str] = []

        for base in bases_to_try:
            url = f"{base}/models/{self.model}:generateContent"
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except requests.HTTPError as http_err:
                detail = self._extract_error(http_err.response) or str(http_err)
                status_code = http_err.response.status_code if http_err.response is not None else None
                if status_code == 404:
                    not_found_messages.append(detail)
                    continue
                if (
                    status_code == 400
                    and (system_instruction is not None or google_search)
                    and (
                        "systemInstruction" in (detail or "")
                        or "system_instruction" in (detail or "")
                        or "tools" in (detail or "")
                    )
                    and base != self.API_BASES[-1]
                ):
                    continue
                raise RuntimeError(detail) from http_err
            except requests.RequestException as exc:
                raise RuntimeError(str(exc)) from exc

            data = response.json()
            prompt_feedback = data.get("promptFeedback")
            if isinstance(prompt_feedback, dict):
                block_reason = prompt_feedback.get("blockReason")
                if block_reason:
                    safety_info = self._format_safety_details(prompt_feedback.get("safetyRatings"))
                    message = f"Prompt blocked by safety filters ({block_reason})."
                    if safety_info:
                        message += f" Details: {safety_info}"
                    raise RuntimeError(message)
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("No candidates returned")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason")
            if finish_reason == "SAFETY":
                safety_info = self._format_safety_details(candidate.get("safetyRatings"))
                message = "Response blocked by safety filters."
                if safety_info:
                    message += f" Details: {safety_info}"
                raise RuntimeError(message)
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            texts: List[str] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("text"):
                    texts.append(part["text"])
                elif "functionCall" in part:
                    func = part["functionCall"] or {}
                    name = func.get("name", "function")
                    raw_args = func.get("args") or func.get("arguments")
                    if isinstance(raw_args, (dict, list)):
                        args_text = json.dumps(raw_args, ensure_ascii=False, indent=2)
                    else:
                        args_text = str(raw_args) if raw_args is not None else "{}"
                    texts.append(f"[Function call requested: {name}]\n{args_text}")
                elif "inlineData" in part:
                    data_info = part["inlineData"] or {}
                    mime = data_info.get("mimeType", "application/octet-stream")
                    size = len(data_info.get("data", ""))
                    texts.append(f"[Inline data payload received: {mime}, {size} base64 chars]")
            reply = "\n\n".join(filter(None, texts))
            if not reply:
                if (
                    finish_reason == "MAX_TOKENS"
                    and allow_max_tokens_retry
                ):
                    upgraded_config: Dict[str, Any]
                    retry_limit = self.max_output_tokens_retry
                    if isinstance(retry_limit, int) and retry_limit > 0:
                        if generation_config is None:
                            upgraded_config = {"maxOutputTokens": retry_limit}
                        else:
                            upgraded_config = copy.deepcopy(generation_config)
                            current_max = upgraded_config.get("maxOutputTokens")
                            if not isinstance(current_max, int) or current_max < retry_limit:
                                upgraded_config["maxOutputTokens"] = retry_limit
                            else:
                                upgraded_config = {}
                    else:
                        upgraded_config = {}
                    if upgraded_config:
                        return self.send_message(
                            history,
                            message,
                            system_instruction=system_instruction,
                            generation_config=upgraded_config,
                            allow_max_tokens_retry=False,
                        )
                if finish_reason == "MAX_TOKENS":
                    suggestions = [
                        "• Shorten the latest request or remove repeated context",
                        "• Clear old chat history via :c to free up tokens",
                        "• Increase the token limit via the settings menu (:o) or config.json",
                    ]
                    hint = "\n".join(suggestions)
                    return (
                        "⚠️ Gemini stopped because it hit the maximum output token limit before returning any text.\n"
                        f"Try one of these adjustments:\n{hint}"
                    )
                safety_info = self._format_safety_details(candidate.get("safetyRatings"))
                if safety_info:
                    raise RuntimeError(f"Model produced no text. Safety ratings: {safety_info}")
                if finish_reason and finish_reason != "STOP":
                    raise RuntimeError(
                        f"Model finished with reason '{finish_reason}' but returned no textual content."
                    )
                raise RuntimeError("Empty response returned")

            self.api_base = base
            return reply

        suggestions = ", ".join(SUGGESTED_MODELS)
        detail_suffix = f" Details: {not_found_messages[-1]}" if not_found_messages else ""
        raise RuntimeError(
            f"Model '{self.model}' not found on the Gemini API endpoints. Try enabling the Generative Language API "
            f"or switch to another model via the settings menu (:o). Suggestions: {suggestions}.{detail_suffix}"
        )

    @staticmethod
    def _extract_error(response: Optional[requests.Response]) -> Optional[str]:
        if response is None:
            return None
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or None

        if isinstance(payload, dict):
            if "error" in payload and isinstance(payload["error"], dict):
                error_obj = payload["error"]
                message = error_obj.get("message")
                status = error_obj.get("status")
                if status and message:
                    return f"{status}: {message}"
                return message or None
            if "message" in payload:
                return str(payload["message"])
        return response.text.strip() or None

    def update_key(self, api_key: str) -> None:
        self.api_key = api_key

    def update_model(self, model: str) -> None:
        self.model = model

    def update_timeout(self, timeout: int) -> None:
        self.timeout = timeout

    def update_max_output_tokens_retry(self, value: Optional[int]) -> None:
        self.max_output_tokens_retry = value

    @staticmethod
    def _format_safety_details(ratings: Optional[List[Dict[str, Any]]]) -> str:
        if not ratings:
            return ""
        details: List[str] = []
        for rating in ratings:
            if not isinstance(rating, dict):
                continue
            category = rating.get("category") or "unknown"
            severity = rating.get("severity") or rating.get("probability") or rating.get("probabilityScore")
            if isinstance(severity, dict):
                severity = severity.get("value")
            details.append(f"{category}: {severity}")
        return " | ".join(details)

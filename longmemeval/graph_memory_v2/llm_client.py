from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


def safe_json_fragment(text: str) -> Any:
    payload = str(text or "").lstrip("\ufeff").strip()
    if not payload:
        raise ValueError("empty LLM response")

    if payload.startswith("```"):
        payload = re.sub(
            r"^```(?:json)?\s*",
            "",
            payload,
            flags=re.IGNORECASE,
        )
        payload = re.sub(r"\s*```$", "", payload).strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", payload):
        try:
            value, _ = decoder.raw_decode(payload[match.start():])
            return value
        except json.JSONDecodeError:
            continue

    raise ValueError("unable to parse JSON from LLM response")


class LLMJSONError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        text: str,
        finish_reason: str,
        attempts: int,
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.attempts = attempts
        preview = text[:800].replace("\n", "\\n")
        super().__init__(
            f"{message}; finish_reason={finish_reason!r}; "
            f"attempts={attempts}; response_chars={len(text)}; "
            f"response_preview={preview!r}"
        )


@dataclass
class ChatResult:
    text: str
    raw: Dict[str, Any]
    elapsed_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    reasoning_content: str = ""


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 300,
        retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.retries = max(1, int(retries))

    @property
    def _is_deepseek(self) -> bool:
        return (
            "deepseek" in self.model_name.casefold()
            or "deepseek.com" in self.base_url.casefold()
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format_json: bool = False,
        thinking: Optional[bool] = None,
    ) -> ChatResult:
        url = self.base_url + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": int(max_tokens),
        }

        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        if thinking is not None and self._is_deepseek:
            payload["thinking"] = {
                "type": "enabled" if thinking else "disabled"
            }

        last_error: Optional[Exception] = None
        started = time.time()

        for attempt in range(self.retries):
            try:
                response = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )

                if (
                    response.status_code in {400, 404, 422}
                    and "response_format" in payload
                ):
                    payload.pop("response_format", None)
                    if attempt + 1 < self.retries:
                        continue

                response.raise_for_status()
                raw = response.json()
                choices = raw.get("choices") or []
                if not choices:
                    raise ValueError("LLM response contains no choices")

                choice = choices[0] or {}
                message = choice.get("message") or {}
                content = message.get("content")

                if isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict):
                            parts.append(
                                str(
                                    part.get("text")
                                    or part.get("content")
                                    or ""
                                )
                            )
                        else:
                            parts.append(str(part))
                    text = "".join(parts).strip()
                else:
                    text = str(content or "").strip()

                usage = raw.get("usage") or {}
                return ChatResult(
                    text=text,
                    raw=raw,
                    elapsed_seconds=time.time() - started,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(
                        usage.get("completion_tokens") or 0
                    ),
                    finish_reason=str(
                        choice.get("finish_reason") or ""
                    ),
                    reasoning_content=str(
                        message.get("reasoning_content") or ""
                    ),
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(8, 2 ** attempt))

        raise RuntimeError(
            f"LLM call failed after {self.retries} attempts: {last_error}"
        )

    def json(
        self,
        prompt: str,
        *,
        system: str = "Return only valid JSON.",
        max_tokens: int = 4096,
    ) -> tuple[Any, ChatResult]:
        system_prompt = (
            system.rstrip()
            + "\nReturn exactly one valid JSON object. "
              "Do not use Markdown fences or explanatory prose."
        )

        last_result: Optional[ChatResult] = None
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            result = self.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                response_format_json=True,
                thinking=False,
            )
            last_result = result

            try:
                return safe_json_fragment(result.text), result
            except ValueError as exc:
                last_error = exc

                if result.finish_reason == "length":
                    break

                if attempt < self.retries:
                    time.sleep(min(4, 2 ** (attempt - 1)))

        assert last_result is not None
        reason = (
            "JSON output was truncated; increase max_tokens or reduce "
            "the extraction window size"
            if last_result.finish_reason == "length"
            else f"unable to parse JSON from LLM response: {last_error}"
        )
        raise LLMJSONError(
            reason,
            text=last_result.text,
            finish_reason=last_result.finish_reason,
            attempts=self.retries,
        )

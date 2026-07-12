from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


def safe_json_fragment(text: str) -> Any:
    payload = str(text or "").strip()
    if not payload:
        raise ValueError("empty LLM response")
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload).strip()
    try:
        return json.loads(payload)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", payload):
        try:
            value, _ = decoder.raw_decode(payload[match.start() :])
            return value
        except Exception:
            continue
    raise ValueError("unable to parse JSON from LLM response")


@dataclass
class ChatResult:
    text: str
    raw: Dict[str, Any]
    elapsed_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


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
        self.retries = retries

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format_json: bool = False,
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
                if response.status_code in {400, 404, 422} and "response_format" in payload:
                    # A number of OpenAI-compatible gateways do not implement JSON mode.
                    # Fall back to prompt-enforced JSON without changing the experiment prompt.
                    payload.pop("response_format", None)
                    if attempt + 1 < self.retries:
                        continue
                response.raise_for_status()
                raw = response.json()
                choices = raw.get("choices") or []
                if not choices:
                    raise ValueError("LLM response contains no choices")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            parts.append(str(part.get("text") or part.get("content") or ""))
                        else:
                            parts.append(str(part))
                    text = "".join(parts).strip()
                else:
                    text = str(content or message.get("reasoning_content") or "").strip()
                usage = raw.get("usage") or {}
                return ChatResult(
                    text=text,
                    raw=raw,
                    elapsed_seconds=time.time() - started,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(8, 2 ** attempt))
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last_error}")

    def json(
        self,
        prompt: str,
        *,
        system: str = "Return only valid JSON.",
        max_tokens: int = 4096,
    ) -> tuple[Any, ChatResult]:
        result = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            response_format_json=True,
        )
        return safe_json_fragment(result.text), result

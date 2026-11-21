"""Minimal client for the OpenAI-compatible chat endpoint of llama-server."""

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from inklab.constants import REQUEST_TIMEOUT_S
from inklab.errors import ServerError


@dataclass(frozen=True)
class ChatResult:
    """One chat completion.

    Attributes:
        content: Final answer text.
        reasoning: Reasoning text that the server split from the answer.
        prompt_tokens: Tokens in the prompt, image tokens included.
        completion_tokens: Generated tokens, reasoning included.
        seconds: Wall-clock time of the request.
        finish_reason: Why generation stopped ("stop" or "length").
    """

    content: str
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    finish_reason: str


def png_data_url(png: bytes) -> str:
    """Return a data URL for PNG bytes."""
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class ChatClient:
    """Send chat requests to one server."""

    def __init__(self, base_url: str, timeout_s: float = REQUEST_TIMEOUT_S) -> None:
        self.url = f"{base_url}/v1/chat/completions"
        self.timeout_s = timeout_s

    def chat(self, content: list[dict] | str, effort: str, max_tokens: int, temperature: float = 0.0,
             reasoning_budget: int = -1) -> ChatResult:
        """Send one user message and return the completion.

        Args:
            content: Plain text, or a list of OpenAI content parts (text and image_url).
            effort: Reasoning effort name that the chat template accepts.
            max_tokens: Limit on generated tokens, reasoning included.
            temperature: Sampling temperature; 0 is greedy.
            reasoning_budget: Limit on reasoning tokens; -1 is no limit and 0 ends reasoning at once.

        Raises:
            ServerError: If the server returns an HTTP error.
        """
        body = {
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"reasoning_effort": effort},
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": 0,
            "reasoning_budget_tokens": reasoning_budget,
        }
        request = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            raise ServerError(f"HTTP {error.code}: {error.read()[:300]!r}") from error
        return self._parse(data, time.perf_counter() - start)

    @staticmethod
    def _parse(data: dict, seconds: float) -> ChatResult:
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})
        return ChatResult(
            content=message.get("content") or "",
            reasoning=message.get("reasoning_content") or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            seconds=seconds,
            finish_reason=choice.get("finish_reason") or "",
        )

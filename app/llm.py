"""Language-model clients that return validated Pydantic objects.

Every agent asks for structured output: the client sends the JSON schema of the expected
Pydantic model and validates the answer against it (one corrective retry if invalid).

- Ollama (default, free, local): the schema is passed as the required output ``format``.
- Anthropic (paid, opt-in): ``messages.parse`` with the Pydantic model as ``output_format``.
  A ``CostTracker`` refuses any call once the per-research spending cap is reached.
"""

from __future__ import annotations

import json
import time
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The model could not produce a valid answer."""


class LLMConfigError(LLMError):
    """The provider is not usable as configured (e.g. Ollama not running, model missing)."""


class BudgetExceededError(LLMError):
    """The per-research spending cap would be exceeded."""


def recoverable(exc: Exception) -> bool:
    """A failed call the research can work around (timeout, invalid answer), as opposed to
    a missing model or an exhausted budget, which must stop the run."""
    return isinstance(exc, LLMError) and not isinstance(exc, LLMConfigError | BudgetExceededError)


class Usage(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


class CostTracker:
    """Accumulates usage for one research and enforces the spending cap (paid providers only)."""

    def __init__(self, max_cost_usd: float, input_usd_per_mtok: float = 0.0, output_usd_per_mtok: float = 0.0):
        self.max_cost_usd = max_cost_usd
        self._in, self._out = input_usd_per_mtok, output_usd_per_mtok
        self.usage = Usage()

    @property
    def paid(self) -> bool:
        return self._in > 0 or self._out > 0

    def check(self) -> None:
        if self.paid and self.usage.cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(
                f"spending cap reached: ${self.usage.cost_usd:.4f} of ${self.max_cost_usd:.2f}"
            )

    def record(self, input_tokens: int, output_tokens: int, seconds: float) -> None:
        u = self.usage
        u.calls += 1
        u.input_tokens += input_tokens
        u.output_tokens += output_tokens
        u.cost_usd += (input_tokens * self._in + output_tokens * self._out) / 1_000_000
        u.seconds += seconds


class LLMClient(Protocol):
    name: str
    model: str
    tracker: CostTracker

    def structured(self, system: str, user: str, schema: type[T]) -> T: ...


def _corrective(user: str, error: str) -> str:
    return (f"{user}\n\nYour previous answer was not valid for the required JSON schema: {error[:500]}\n"
            "Answer again with JSON that matches the schema exactly.")


class OllamaClient:
    name = "ollama"

    def __init__(self, base_url: str, model: str, tracker: CostTracker, timeout: float,
                 num_ctx: int = 8192, temperature: float = 0.1, max_output_tokens: int = 4096,
                 client: httpx.Client | None = None):
        self.model, self.tracker = model, tracker
        self._url = base_url.rstrip("/") + "/api/chat"
        self._options = {"num_ctx": num_ctx, "temperature": temperature, "num_predict": max_output_tokens}
        self._client = client or httpx.Client(timeout=timeout)

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        prompt = user
        for attempt in range(2):
            self.tracker.check()
            content = self._chat(system, prompt, schema.model_json_schema())
            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMError(f"{self.model} returned invalid {schema.__name__}: {exc}") from exc
                prompt = _corrective(user, str(exc))
        raise AssertionError("unreachable")

    def _chat(self, system: str, user: str, json_schema: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": json_schema,
            "stream": False,
            "options": self._options,
        }
        started = time.perf_counter()
        try:
            response = self._client.post(self._url, json=payload)
        except httpx.ConnectError as exc:
            raise LLMConfigError(f"Ollama is not reachable at {self._url}: is it running?") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama timed out after {self._client.timeout.read}s") from exc
        if response.status_code == 404:
            raise LLMConfigError(f"model {self.model!r} is not installed: run `ollama pull {self.model}`")
        if response.status_code >= 400:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:200]}")
        data = response.json()
        self.tracker.record(data.get("prompt_eval_count", 0), data.get("eval_count", 0),
                            time.perf_counter() - started)
        return data.get("message", {}).get("content", "")


class AnthropicClient:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, tracker: CostTracker, timeout: float,
                 max_output_tokens: int = 4096):
        import anthropic

        self.model, self.tracker = model, tracker
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._max_tokens = max_output_tokens

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        prompt = user
        for attempt in range(2):
            self.tracker.check()
            started = time.perf_counter()
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    output_format=schema,
                )
            except self._anthropic.AuthenticationError as exc:
                raise LLMConfigError("Anthropic rejected the API key") from exc
            except self._anthropic.APIConnectionError as exc:
                raise LLMError(f"cannot reach the Anthropic API: {exc}") from exc
            except self._anthropic.APIStatusError as exc:
                raise LLMError(f"Anthropic API error {exc.status_code}") from exc
            self.tracker.record(response.usage.input_tokens, response.usage.output_tokens,
                                time.perf_counter() - started)
            if response.stop_reason == "refusal":
                raise LLMError("the model declined this request")
            if response.parsed_output is not None:
                return response.parsed_output
            if attempt == 0:
                prompt = _corrective(user, f"stop_reason={response.stop_reason}")
        raise LLMError(f"{self.model} did not return a valid {schema.__name__}")


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient(settings.ollama_base_url, settings.ollama_model, CostTracker(0.0),
                            settings.llm_timeout_seconds, settings.ollama_num_ctx,
                            settings.ollama_temperature, settings.llm_max_output_tokens)
    if settings.llm_provider == "anthropic":
        tracker = CostTracker(settings.max_cost_usd_per_research, settings.anthropic_input_usd_per_mtok,
                              settings.anthropic_output_usd_per_mtok)
        return AnthropicClient(settings.anthropic_api_key.get_secret_value(), settings.anthropic_model,
                               tracker, settings.llm_timeout_seconds, settings.llm_max_output_tokens)
    raise LLMConfigError("the OpenAI provider is not implemented yet: use LLM_PROVIDER=ollama or anthropic")


def schema_hint(schema: type[BaseModel]) -> str:
    """Compact schema text to include in prompts (small local models follow it better)."""
    return json.dumps(schema.model_json_schema(), separators=(",", ":"))

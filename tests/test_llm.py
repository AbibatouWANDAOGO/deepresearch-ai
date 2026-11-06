import json

import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm import (
    BudgetExceededError,
    CostTracker,
    LLMConfigError,
    LLMError,
    OllamaClient,
    build_llm,
)


class Answer(BaseModel):
    city: str
    population: int


def ollama(handler) -> OllamaClient:
    return OllamaClient("http://localhost:11434", "qwen2.5:3b", CostTracker(0.0), timeout=5,
                        client=httpx.Client(transport=httpx.MockTransport(handler)))


def chat_response(content: str, prompt_tokens=50, output_tokens=10):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content},
                                     "prompt_eval_count": prompt_tokens, "eval_count": output_tokens})


def test_ollama_sends_schema_and_returns_validated_object():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return chat_response('{"city": "Paris", "population": 2100000}')

    client = ollama(handler)
    answer = client.structured("system text", "user text", Answer)
    assert answer == Answer(city="Paris", population=2_100_000)
    assert seen["format"] == Answer.model_json_schema() and seen["stream"] is False
    assert seen["messages"][0] == {"role": "system", "content": "system text"}
    assert seen["options"]["num_ctx"] == 8192
    assert client.tracker.usage.calls == 1 and client.tracker.usage.input_tokens == 50


def test_invalid_answer_gets_one_corrective_retry():
    answers = iter(['{"city": "Paris"}', '{"city": "Paris", "population": 1}'])
    prompts = []

    def handler(request):
        prompts.append(json.loads(request.content)["messages"][1]["content"])
        return chat_response(next(answers))

    assert ollama(handler).structured("s", "question", Answer).population == 1
    assert "not valid for the required JSON schema" in prompts[1]


def test_two_invalid_answers_raise():
    with pytest.raises(LLMError, match="invalid Answer"):
        ollama(lambda r: chat_response("not json")).structured("s", "u", Answer)


def test_ollama_not_running_is_a_config_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)
    with pytest.raises(LLMConfigError, match="is it running"):
        ollama(handler).structured("s", "u", Answer)


def test_missing_model_tells_how_to_install_it():
    with pytest.raises(LLMConfigError, match="ollama pull qwen2.5:3b"):
        ollama(lambda r: httpx.Response(404, json={"error": "model not found"})).structured("s", "u", Answer)


def test_cost_tracker_blocks_paid_calls_beyond_the_cap():
    tracker = CostTracker(max_cost_usd=0.01, input_usd_per_mtok=4.0, output_usd_per_mtok=20.0)
    tracker.record(input_tokens=1000, output_tokens=200, seconds=1.0)  # $0.004 + $0.004
    assert tracker.usage.cost_usd == pytest.approx(0.008)
    tracker.check()
    tracker.record(input_tokens=1000, output_tokens=100, seconds=1.0)
    with pytest.raises(BudgetExceededError):
        tracker.check()


def test_free_provider_is_never_blocked():
    tracker = CostTracker(0.0)
    tracker.record(10**9, 10**9, 1.0)
    tracker.check()


def test_build_llm_defaults_to_ollama_and_refuses_openai(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    assert isinstance(build_llm(Settings(_env_file=None)), OllamaClient)
    openai = Settings(_env_file=None, llm_provider="openai", allow_paid_apis=True, openai_api_key="k", openai_model="m")
    with pytest.raises(LLMConfigError, match="not implemented"):
        build_llm(openai)

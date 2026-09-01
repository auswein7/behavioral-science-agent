"""Unit tests for the provider adapter layer, the ollama SDK mocked."""

from types import SimpleNamespace

import pytest

from src import adapters
from src.adapters import ChatMessage, ChatResponse, OllamaChatModel
from src.errors import ConfigurationError


class FakeOllama:
    def __init__(self, local_models=("some-model:latest",), content="hello"):
        self.local_models = list(local_models)
        self.content = content
        self.done_reason = "stop"
        self.pulled: list[str] = []
        self.chat_calls: list[dict] = []

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self.local_models])

    def pull(self, name):
        self.pulled.append(name)

    def chat(self, model, messages, options):
        self.chat_calls.append({"model": model, "messages": messages, "options": options})
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content),
            done_reason=self.done_reason,
            prompt_eval_count=10,
            eval_count=5,
        )


@pytest.fixture
def fake(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(adapters, "ollama", fake)
    return fake


class TestEnsureAvailable:
    def test_present_model_passes(self, fake):
        fake.local_models = ["m1:latest"]
        OllamaChatModel("m1").ensure_available()
        assert fake.pulled == []

    def test_missing_model_pulled_when_auto_pull(self, fake):
        OllamaChatModel("absent", auto_pull=True).ensure_available()
        assert fake.pulled == ["absent"]

    def test_missing_model_is_typed_error_without_auto_pull(self, fake):
        with pytest.raises(ConfigurationError):
            OllamaChatModel("absent").ensure_available()

    def test_hint_appended_to_error(self, fake):
        with pytest.raises(ConfigurationError, match="create it first"):
            OllamaChatModel("absent", unavailable_hint="create it first").ensure_available()


class TestInvoke:
    def test_images_lifted_into_payload(self, fake):
        model = OllamaChatModel("m1")
        model.invoke([ChatMessage(role="user", content="p", images=(b"jpegbytes",))])
        [call] = fake.chat_calls
        assert call["messages"] == [{"role": "user", "content": "p", "images": [b"jpegbytes"]}]

    def test_text_message_has_no_images_key(self, fake):
        OllamaChatModel("m1").invoke([ChatMessage(role="system", content="s")])
        [call] = fake.chat_calls
        assert call["messages"] == [{"role": "system", "content": "s"}]

    def test_options_passed_through(self, fake):
        OllamaChatModel("m1").invoke(
            [ChatMessage(role="user", content="p")], temperature=0.0, seed=7
        )
        assert fake.chat_calls[0]["options"] == {"temperature": 0.0, "seed": 7}

    def test_response_is_provider_neutral(self, fake):
        # Provider-specific types never cross the adapter boundary (principle 1).
        response = OllamaChatModel("m1").invoke([ChatMessage(role="user", content="p")])
        assert isinstance(response, ChatResponse)
        assert response.content == "hello"
        assert response.model == "m1"
        assert response.done_reason == "stop"
        assert (response.prompt_eval_count, response.eval_count) == (10, 5)

    def test_none_content_normalized_to_empty(self, fake):
        fake.content = None
        response = OllamaChatModel("m1").invoke([ChatMessage(role="user", content="p")])
        assert response.content == ""

    def test_length_stop_normalized_to_truncated(self, fake):
        # The provider's literal "length" is mapped here and nowhere upstream:
        # consumers branch on ChatResponse.truncated, never on the raw string.
        fake.done_reason = "length"
        response = OllamaChatModel("m1").invoke([ChatMessage(role="user", content="p")])
        assert response.truncated is True
        assert response.done_reason == "length"

    def test_other_stop_reason_is_not_truncated(self, fake):
        response = OllamaChatModel("m1").invoke([ChatMessage(role="user", content="p")])
        assert response.truncated is False

    def test_missing_stop_reason_leaves_truncated_unknown(self, fake):
        fake.done_reason = None
        response = OllamaChatModel("m1").invoke([ChatMessage(role="user", content="p")])
        assert response.truncated is None


class _CountingModel(adapters.AbstractChatModel):
    def __init__(self, responses):
        self.responses = list(responses)
        self.ensured = 0

    def ensure_available(self):
        self.ensured += 1

    def invoke(self, messages, **options):
        return self.responses.pop(0)

    def capabilities(self):
        return {"vision": True}


class TestUsageRecordingModel:
    def test_tallies_reported_usage_across_calls(self):
        from src.adapters import ChatResponse, UsageRecordingModel

        model = UsageRecordingModel(
            _CountingModel(
                [
                    ChatResponse("a", "m", prompt_eval_count=10, eval_count=2),
                    ChatResponse("b", "m", prompt_eval_count=5, eval_count=1),
                ]
            )
        )
        for _ in range(2):
            model.invoke([ChatMessage(role="user", content="p")])
        assert model.tally.calls == 2
        assert model.tally.calls_reporting == 2
        assert (model.tally.prompt_tokens, model.tally.completion_tokens) == (15, 3)

    def test_unreported_usage_counts_the_call_but_not_tokens(self):
        # A backend with no telemetry: the tally shows 0 calls_reporting so
        # a zero total reads as unknown, never as "free" (principle 6).
        from src.adapters import ChatResponse, UsageRecordingModel

        model = UsageRecordingModel(_CountingModel([ChatResponse("a", "m")]))
        model.invoke([ChatMessage(role="user", content="p")])
        assert model.tally.calls == 1
        assert model.tally.calls_reporting == 0

    def test_delegates_availability_and_capabilities(self):
        from src.adapters import ChatResponse, UsageRecordingModel

        inner = _CountingModel([ChatResponse("a", "m")])
        model = UsageRecordingModel(inner)
        model.ensure_available()
        assert inner.ensured == 1
        assert model.capabilities() == {"vision": True}


class TestCapabilities:
    def test_vision_declared(self, fake):
        assert OllamaChatModel("m1").capabilities()["vision"] is True

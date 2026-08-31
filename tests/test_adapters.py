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
            done_reason="stop",
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


class TestCapabilities:
    def test_vision_declared(self, fake):
        assert OllamaChatModel("m1").capabilities()["vision"] is True

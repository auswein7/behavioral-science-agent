"""Unit tests for the fairlib-backed adapter, fairlib stubbed at the module
seam. The stub mirrors the settled fair_llm issue #146 contract (first-class
images tuple on Message, boolean vision capability); the final test class
runs the same mapping against a real installed fairlib and self-skips until
the vision release is importable, making it the tandem contract check."""

import dataclasses
import sys
import types
from dataclasses import dataclass

import pytest

from src.adapters import ChatMessage, ChatResponse
from src.errors import AdapterError, ConfigurationError
from src.fairlib_adapter import FairlibChatModel


@dataclass
class StubMessage:
    role: str
    content: str
    images: tuple = ()


class StubFairlibAdapterError(Exception):
    pass


def install_stub_fairlib(monkeypatch):
    fairlib_mod = types.ModuleType("fairlib")
    core_mod = types.ModuleType("fairlib.core")
    message_mod = types.ModuleType("fairlib.core.message")
    errors_mod = types.ModuleType("fairlib.core.errors")
    message_mod.Message = StubMessage
    errors_mod.AdapterError = StubFairlibAdapterError
    fairlib_mod.core = core_mod
    core_mod.message = message_mod
    core_mod.errors = errors_mod
    for name, mod in [
        ("fairlib", fairlib_mod),
        ("fairlib.core", core_mod),
        ("fairlib.core.message", message_mod),
        ("fairlib.core.errors", errors_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


class FakeFairlibAdapter:
    def __init__(self, content="hello", vision=True, error=None):
        self.content = content
        self.vision = vision
        self.error = error
        self.invoke_calls: list[dict] = []

    def invoke(self, messages, **kwargs):
        self.invoke_calls.append({"messages": messages, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        return StubMessage(role="assistant", content=self.content)

    def get_model_capabilities(self):
        return {"vision": self.vision}


@pytest.fixture
def stub(monkeypatch):
    install_stub_fairlib(monkeypatch)


def make_model(adapter, availability_check=lambda: None):
    return FairlibChatModel(adapter, "m1", availability_check=availability_check)


class TestConstruction:
    def test_missing_fairlib_is_typed_error(self, monkeypatch):
        for name in (
            "fairlib",
            "fairlib.core",
            "fairlib.core.message",
            "fairlib.core.errors",
        ):
            monkeypatch.setitem(sys.modules, name, None)
        with pytest.raises(ConfigurationError, match="fair-llm is not installed"):
            make_model(FakeFairlibAdapter())

    def test_ensure_available_delegates_to_the_supplied_check(self, stub):
        calls = []
        make_model(
            FakeFairlibAdapter(), availability_check=lambda: calls.append(1)
        ).ensure_available()
        assert calls == [1]


class TestInvoke:
    def test_messages_mapped_to_fairlib_shape(self, stub):
        adapter = FakeFairlibAdapter()
        make_model(adapter).invoke(
            [
                ChatMessage(role="system", content="s"),
                ChatMessage(role="user", content="p", images=(b"jpegbytes",)),
            ]
        )
        [call] = adapter.invoke_calls
        assert call["messages"] == [
            StubMessage(role="system", content="s", images=()),
            StubMessage(role="user", content="p", images=(b"jpegbytes",)),
        ]

    def test_options_passed_through(self, stub):
        adapter = FakeFairlibAdapter()
        make_model(adapter).invoke(
            [ChatMessage(role="user", content="p")], temperature=0.0, seed=7
        )
        assert adapter.invoke_calls[0]["kwargs"] == {"temperature": 0.0, "seed": 7}

    def test_response_is_provider_neutral(self, stub):
        # Provider-specific types never cross the adapter boundary (principle 1).
        response = make_model(FakeFairlibAdapter()).invoke(
            [ChatMessage(role="user", content="p")]
        )
        assert isinstance(response, ChatResponse)
        assert (response.content, response.model) == ("hello", "m1")

    def test_usage_fields_none_until_fairlib_147(self, stub):
        response = make_model(FakeFairlibAdapter()).invoke(
            [ChatMessage(role="user", content="p")]
        )
        assert (
            response.done_reason,
            response.prompt_eval_count,
            response.eval_count,
        ) == (
            None,
            None,
            None,
        )

    def test_none_content_normalized_to_empty(self, stub):
        response = make_model(FakeFairlibAdapter(content=None)).invoke(
            [ChatMessage(role="user", content="p")]
        )
        assert response.content == ""

    def test_images_refused_without_vision_capability(self, stub):
        # Refusal beats silent frame-dropping (principle 6).
        adapter = FakeFairlibAdapter(vision=False)
        with pytest.raises(AdapterError, match="vision"):
            make_model(adapter).invoke(
                [ChatMessage(role="user", content="p", images=(b"x",))]
            )
        assert adapter.invoke_calls == []

    def test_text_messages_fine_without_vision_capability(self, stub):
        response = make_model(FakeFairlibAdapter(vision=False)).invoke(
            [ChatMessage(role="user", content="p")]
        )
        assert response.content == "hello"

    def test_fairlib_error_wrapped_as_typed_adapter_error(self, stub):
        adapter = FakeFairlibAdapter(error=StubFairlibAdapterError("boom"))
        with pytest.raises(AdapterError, match="boom"):
            make_model(adapter).invoke([ChatMessage(role="user", content="p")])


class TestCapabilities:
    def test_vision_mapped_from_fairlib(self, stub):
        assert make_model(FakeFairlibAdapter(vision=True)).capabilities() == {
            "vision": True
        }
        assert make_model(FakeFairlibAdapter(vision=False)).capabilities() == {
            "vision": False
        }

    def test_missing_flag_means_no_vision(self, stub):
        adapter = FakeFairlibAdapter()
        adapter.get_model_capabilities = dict
        assert make_model(adapter).capabilities() == {"vision": False}


class TestRealFairlibContract:
    """Tandem contract check: runs only against an installed fairlib whose
    Message already carries the #146 images field, and verifies our mapping
    constructs real fairlib Messages. Self-skips otherwise (workspace test
    rule: skipif, never an ignore list)."""

    def test_chatmessage_maps_onto_real_fairlib_message(self):
        message_mod = pytest.importorskip("fairlib.core.message")
        fields = {f.name for f in dataclasses.fields(message_mod.Message)}
        if "images" not in fields:
            pytest.skip("installed fairlib predates issue #146 vision support")
        adapter = FakeFairlibAdapter()
        make_model(adapter).invoke(
            [ChatMessage(role="user", content="p", images=(b"jpegbytes",))]
        )
        [call] = adapter.invoke_calls
        [sent] = call["messages"]
        assert isinstance(sent, message_mod.Message)
        assert sent.images == (b"jpegbytes",)

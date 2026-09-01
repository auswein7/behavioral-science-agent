"""Unit tests for the fairlib-backed adapter, fairlib stubbed at the module
seam. The stub mirrors the settled fair_llm issue #146 contract (first-class
images tuple on Message, boolean vision capability) plus the frozen issue
#147 sketch (Message.usage with a normalized DoneReason enum and the raw
provider string); the final test class runs the same mapping against a real
installed fairlib and self-skips per capability until each release is
importable, making it the tandem contract check."""

import dataclasses
import enum
import sys
import types
from dataclasses import dataclass

import pytest

from src.adapters import ChatMessage, ChatResponse
from src.errors import AdapterError, ConfigurationError
from src.fairlib_adapter import FairlibChatModel


class StubDoneReason(enum.StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


@dataclass(frozen=True)
class StubUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ms: float | None = None
    model: str | None = None
    done_reason: StubDoneReason | None = None
    raw_done_reason: str | None = None


@dataclass
class StubMessage:
    role: str
    content: str
    images: tuple = ()
    usage: StubUsage | None = None


class StubFairlibAdapterError(Exception):
    pass


class StubFairlibConfigurationError(Exception):
    pass


def install_stub_fairlib(monkeypatch):
    fairlib_mod = types.ModuleType("fairlib")
    core_mod = types.ModuleType("fairlib.core")
    message_mod = types.ModuleType("fairlib.core.message")
    errors_mod = types.ModuleType("fairlib.core.errors")
    message_mod.Message = StubMessage
    message_mod.DoneReason = StubDoneReason
    message_mod.Usage = StubUsage
    errors_mod.AdapterError = StubFairlibAdapterError
    errors_mod.ConfigurationError = StubFairlibConfigurationError
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
    def __init__(self, content="hello", vision=True, error=None, usage=None):
        self.content = content
        self.vision = vision
        self.error = error
        self.usage = usage
        self.invoke_calls: list[dict] = []

    def invoke(self, messages, **kwargs):
        self.invoke_calls.append({"messages": messages, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        return StubMessage(role="assistant", content=self.content, usage=self.usage)

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

    def test_usage_fields_none_when_reply_carries_no_usage(self, stub):
        # A pre-#147 fairlib, or a reply without telemetry: unknown, never
        # "no overflow" (principle 6).
        response = make_model(FakeFairlibAdapter()).invoke(
            [ChatMessage(role="user", content="p")]
        )
        assert (
            response.done_reason,
            response.prompt_eval_count,
            response.eval_count,
            response.truncated,
        ) == (
            None,
            None,
            None,
            None,
        )


class TestUsageMapping:
    """The frozen fair_llm #147 sketch mapped onto ChatResponse."""

    def test_length_sentinel_maps_to_truncated(self, stub):
        usage = StubUsage(
            prompt_tokens=110936,
            completion_tokens=20136,
            done_reason=StubDoneReason.LENGTH,
            raw_done_reason="length",
        )
        response = make_model(FakeFairlibAdapter(usage=usage)).invoke(
            [ChatMessage(role="user", content="p")]
        )
        assert response.truncated is True
        # The raw provider string rides through for error text and
        # provenance; the enum is consumed here and crosses the seam nowhere.
        assert response.done_reason == "length"
        assert (response.prompt_eval_count, response.eval_count) == (110936, 20136)

    def test_other_stop_is_not_truncated(self, stub):
        usage = StubUsage(done_reason=StubDoneReason.STOP, raw_done_reason="stop")
        response = make_model(FakeFairlibAdapter(usage=usage)).invoke(
            [ChatMessage(role="user", content="p")]
        )
        assert response.truncated is False
        assert response.done_reason == "stop"

    def test_usage_without_done_reason_leaves_truncated_unknown(self, stub):
        usage = StubUsage(prompt_tokens=7, completion_tokens=3)
        response = make_model(FakeFairlibAdapter(usage=usage)).invoke(
            [ChatMessage(role="user", content="p")]
        )
        assert response.truncated is None
        assert (response.prompt_eval_count, response.eval_count) == (7, 3)

    def test_none_content_normalized_to_empty(self, stub):
        response = make_model(FakeFairlibAdapter(content=None)).invoke(
            [ChatMessage(role="user", content="p")]
        )
        assert response.content == ""

    def test_images_refused_without_vision_capability(self, stub):
        # Refusal beats silent frame-dropping (principle 6); a capability
        # mismatch is misconfiguration, matching upstream fairlib semantics.
        adapter = FakeFairlibAdapter(vision=False)
        with pytest.raises(ConfigurationError, match="vision"):
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

    def test_fairlib_refusal_wrapped_as_typed_configuration_error(self, stub):
        # fairlib refuses capability mismatches at payload-build time with its
        # own ConfigurationError; the type must not cross the seam raw.
        adapter = FakeFairlibAdapter(
            error=StubFairlibConfigurationError("no vision declared")
        )
        with pytest.raises(ConfigurationError, match="no vision declared"):
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

    def test_non_vision_refusal_is_this_projects_configuration_error(self):
        # Upstream refuses images on a non-vision adapter with fairlib's
        # ConfigurationError; through the seam the caller must see this
        # project's ConfigurationError, never a fairlib type. No network:
        # refusal fires before any request.
        message_mod = pytest.importorskip("fairlib.core.message")
        fields = {f.name for f in dataclasses.fields(message_mod.Message)}
        if "images" not in fields:
            pytest.skip("installed fairlib predates issue #146 vision support")
        mal = pytest.importorskip("fairlib.modules.mal.local_llama_adapter")
        adapter = mal.OllamaAdapter(model_name="m1", vision=False)
        with pytest.raises(ConfigurationError, match="vision"):
            make_model(adapter).invoke(
                [ChatMessage(role="user", content="p", images=(b"x",))]
            )

    def test_real_usage_record_maps_onto_chat_response(self):
        # The frozen #147 contract against the real types: a Usage carrying
        # DoneReason.LENGTH becomes truncated=True with the raw string and
        # token counts passed through. Self-skips until the installed fairlib
        # ships #147. No network: the reply is built directly.
        message_mod = pytest.importorskip("fairlib.core.message")
        if not hasattr(message_mod, "Usage"):
            pytest.skip("installed fairlib predates issue #147 usage telemetry")
        usage = message_mod.Usage(
            prompt_tokens=11,
            completion_tokens=2,
            done_reason=message_mod.DoneReason.LENGTH,
            raw_done_reason="length",
        )

        class UsageReplyAdapter(FakeFairlibAdapter):
            def invoke(self, messages, **kwargs):
                return message_mod.Message(
                    role="assistant", content="", usage=usage
                )

        response = make_model(UsageReplyAdapter()).invoke(
            [ChatMessage(role="user", content="p")]
        )
        assert response.truncated is True
        assert response.done_reason == "length"
        assert (response.prompt_eval_count, response.eval_count) == (11, 2)

"""Fairlib-backed chat model: the second implementation of the adapter seam.

Wraps any fairlib chat adapter (fairlib.modules.mal.*) behind this repo's
AbstractChatModel, mapping ChatMessage to fairlib's Message. Built against
the settled fair_llm issue #146 contract: Message carries a first-class
images tuple of bytes, adapters declare a boolean "vision" capability via
get_model_capabilities, and callers branch on the capability, never on the
adapter type.

fair-llm is an optional dependency until the vision release ships; importing
this module without it is fine, constructing the adapter is not. Reliability
is deliberately NOT wrapped here: fairlib owns retry for its own adapters
(design principle 7), unlike src/adapters.py whose OllamaChatModel wraps the
raw SDK in call_with_retry.

Usage telemetry (fair_llm issue #147, frozen sketch 2026-09-01): replies
carry Message.usage - a Usage record with prompt_tokens, completion_tokens,
a normalized DoneReason enum (LENGTH is the one overflow sentinel) and the
verbatim raw_done_reason. The mapping here sets ChatResponse.truncated from
the sentinel and passes raw_done_reason through as done_reason; against a
fairlib that predates #147, replies simply have no usage attribute and every
accounting field stays None (the seam's honest tri-state, principle 6).

Per-call accounting (fair_llm issue #170, contract fixed in PR #175): every
one-shot call on a fairlib adapter emits one ModelInvocationEvent on the bus
bound with adapter.bind_event_bus. FairlibUsageSubscriber below turns that
stream into the per-stage UsageTally that RunProvenance records, so the
framework's own accounting is the record and this seam reads nothing off
replies for it (observability principle: stages emit, consumers subscribe).
"""

import logging
from collections.abc import Callable, Sequence
from typing import Protocol

from src.adapters import AbstractChatModel, ChatMessage, ChatResponse, UsageTally
from src.errors import AdapterError, ConfigurationError

logger = logging.getLogger(__name__)


class FairlibChatAdapter(Protocol):
    """The slice of fairlib's AbstractChatModel this wrapper relies on."""

    def invoke(self, messages: list, **kwargs: object) -> object: ...

    def get_model_capabilities(self) -> dict: ...


class EventBindingAdapter(Protocol):
    """The slice a fairlib adapter grows with PR #175: one bus per adapter."""

    def bind_event_bus(self, bus: object) -> None: ...


def _fairlib_symbols() -> tuple[type, type[Exception], type[Exception]]:
    """Resolve fairlib types lazily so the dependency stays optional."""
    try:
        from fairlib.core.errors import AdapterError as FairlibAdapterError
        from fairlib.core.errors import ConfigurationError as FairlibConfigurationError
        from fairlib.core.message import Message
    except ImportError as e:
        raise ConfigurationError(
            "fair-llm is not installed in this environment. The fairlib-backed "
            "adapter needs the vision-capable release (fair_llm issue #146); "
            "until it ships, install the in-flight branch with "
            "pip install -e <fair_llm checkout>."
        ) from e
    return Message, FairlibAdapterError, FairlibConfigurationError


def fairlib_reports_usage() -> bool:
    """True when the installed fairlib carries #147 usage telemetry on its
    Message. False for a missing fairlib too - callers that need fairlib at
    all get the ConfigurationError from construction, not from this probe."""
    try:
        import dataclasses

        from fairlib.core.message import Message
    except ImportError:
        return False
    return "usage" in {f.name for f in dataclasses.fields(Message)}


def fairlib_emits_invocation_events() -> bool:
    """True when the installed fairlib carries the #170 contract (PR #175):
    the ModelInvocationEvent type and the bus it is bound through. False for
    a missing fairlib too, on the same reasoning as fairlib_reports_usage."""
    try:
        from fairlib.core.event_bus import AgentEventBus  # noqa: F401
        from fairlib.core.events import ModelInvocationEvent  # noqa: F401
    except ImportError:
        return False
    return True


class FairlibUsageSubscriber:
    """Feeds a UsageTally from fairlib's own per-call accounting.

    Binds a fresh event bus to one fairlib adapter and subscribes to its
    ModelInvocationEvent: one event per one-shot call, outcome completed or
    failed, an optional Usage record, a request digest and never content.
    One bus per adapter is what gives the tally its stage attribution - the
    event itself carries no stage field. Unlike a reply-side wrapper this
    sees failed calls, and the tally's source says the numbers are the
    framework's.
    """

    def __init__(self, tally: UsageTally) -> None:
        self.tally = tally

    def bind(self, adapter: EventBindingAdapter) -> None:
        """Wire this subscriber to one adapter.

        Raises ConfigurationError when the installed fairlib predates the
        event contract, or when fairlib refuses the bind (an adapter whose
        identity cannot be resolved from describe_config): both are wiring
        faults, found before the first call (design principle 6)."""
        try:
            from fairlib.core.errors import (
                ConfigurationError as FairlibConfigurationError,
            )
            from fairlib.core.event_bus import AgentEventBus
            from fairlib.core.events import ModelInvocationEvent
        except ImportError as e:
            raise ConfigurationError(
                "usage accounting through fairlib events needs a fair-llm that "
                "ships ModelInvocationEvent (fair_llm issue #170, PR #175); the "
                "installed one does not."
            ) from e
        bus = AgentEventBus()
        bus.subscribe(ModelInvocationEvent, self._on_invocation)
        try:
            adapter.bind_event_bus(bus)
        except FairlibConfigurationError as e:
            raise ConfigurationError(
                f"fairlib refused to bind usage accounting to the adapter: {e}"
            ) from e
        self.tally.source = "fairlib_events"

    def _on_invocation(self, event: object) -> None:
        self.tally.calls += 1
        outcome = getattr(event, "outcome", None)
        # ModelInvocationOutcome is a str enum whose values are the contract
        # strings; comparing the value keeps the enum type out of this seam.
        if getattr(outcome, "value", outcome) != "completed":
            self.tally.calls_failed += 1
        usage = getattr(event, "usage", None)
        if usage is None:
            return
        if usage.prompt_tokens is None and usage.completion_tokens is None:
            return
        self.tally.calls_reporting += 1
        self.tally.prompt_tokens += usage.prompt_tokens or 0
        self.tally.completion_tokens += usage.completion_tokens or 0


class FairlibChatModel(AbstractChatModel):
    """Chat models reached through a fairlib adapter.

    availability_check is required because fairlib does not own model
    availability: the entry point supplies whatever probe the backend needs
    (the Ollama registry check, a token check) and passes an explicit no-op
    when there is none. Requiring it keeps the choice visible at the call
    site instead of defaulting to a silent skip (design principle 6).
    """

    def __init__(
        self,
        adapter: FairlibChatAdapter,
        model_name: str,
        availability_check: Callable[[], None],
        description: str = "chat completion",
    ) -> None:
        self._message_cls, self._fairlib_error, self._fairlib_config_error = (
            _fairlib_symbols()
        )
        # The one overflow sentinel of the #147 contract. DoneReason is a str
        # enum whose LENGTH value is the frozen string "length", so the
        # fallback compares equal to the real enum member; it exists only so
        # this class still constructs against a pre-#147 fairlib (where
        # replies carry no usage and the sentinel is never consulted).
        try:
            from fairlib.core.message import DoneReason

            self._length_sentinel: object = DoneReason.LENGTH
        except ImportError:
            self._length_sentinel = "length"
        self._adapter = adapter
        self.model_name = model_name
        self._availability_check = availability_check
        self.description = description

    def ensure_available(self) -> None:
        self._availability_check()

    def invoke(
        self, messages: Sequence[ChatMessage], **options: object
    ) -> ChatResponse:
        if any(m.images for m in messages) and not self.capabilities()["vision"]:
            # Defense in depth ahead of fairlib's own payload-time refusal,
            # with matching semantics: a capability mismatch is deterministic
            # misconfiguration, not a call failure (design principle 6).
            raise ConfigurationError(
                f"{self.model_name} received a message carrying images but does "
                f"not declare the vision capability; use a vision-capable model "
                f"or send messages without images."
            )

        fairlib_messages = [
            self._message_cls(role=m.role, content=m.content, images=m.images)
            for m in messages
        ]
        try:
            reply = self._adapter.invoke(fairlib_messages, **options)
        except self._fairlib_config_error as e:
            # fairlib refuses capability mismatches (e.g. images to a
            # non-vision adapter) at payload-build time; keep the type
            # provider-neutral crossing the seam (principle 1).
            raise ConfigurationError(
                f"{self.description} with {self.model_name} was refused by the "
                f"fairlib adapter: {e}"
            ) from e
        except self._fairlib_error as e:
            raise AdapterError(
                f"{self.description} with {self.model_name} failed in the fairlib adapter: {e}"
            ) from e

        content = getattr(reply, "content", None) or ""
        usage = getattr(reply, "usage", None)
        if usage is None:
            # Pre-#147 fairlib, or a reply without telemetry: every
            # accounting field stays None - unknown, not "no overflow".
            return ChatResponse(content=content, model=self.model_name)
        truncated = None
        if usage.done_reason is not None:
            truncated = usage.done_reason == self._length_sentinel
        return ChatResponse(
            content=content,
            model=self.model_name,
            done_reason=usage.raw_done_reason,
            prompt_eval_count=usage.prompt_tokens,
            eval_count=usage.completion_tokens,
            truncated=truncated,
        )

    def capabilities(self) -> dict[str, bool]:
        declared = self._adapter.get_model_capabilities()
        return {"vision": bool(declared.get("vision", False))}

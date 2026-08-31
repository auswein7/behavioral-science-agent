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

Known contract gap, tracked upstream as fair_llm issue #147: fairlib replies
carry no usage accounting yet, so done_reason / prompt_eval_count /
eval_count on ChatResponse are None. The screenplay stage's context-overflow
diagnostic depends on done_reason, so Ornith must stay on OllamaChatModel
until #147 lands; the captioner does not read those fields and can swap now.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Protocol

from src.adapters import AbstractChatModel, ChatMessage, ChatResponse
from src.errors import AdapterError, ConfigurationError

logger = logging.getLogger(__name__)


class FairlibChatAdapter(Protocol):
    """The slice of fairlib's AbstractChatModel this wrapper relies on."""

    def invoke(self, messages: list, **kwargs: object) -> object: ...

    def get_model_capabilities(self) -> dict: ...


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

        # done_reason / token counts stay None until fair_llm #147 attaches
        # usage accounting to the reply; see the module docstring.
        return ChatResponse(
            content=getattr(reply, "content", None) or "",
            model=self.model_name,
        )

    def capabilities(self) -> dict[str, bool]:
        declared = self._adapter.get_model_capabilities()
        return {"vision": bool(declared.get("vision", False))}

"""Model-provider access in exactly one place (design principle 1: agnosticism).

Every stage talks to a chat model through AbstractChatModel below and never
imports a provider SDK or sees a provider response type. The shapes mirror
fairlib's AbstractChatModel/Message contract deliberately, so swapping these
classes for fairlib adapters once vision support lands upstream is mechanical:
construct a different adapter at the entry point, keep every call site.

Outside this module the ollama SDK appears only in adapter-layer
infrastructure: src/reliability.py (classifying provider transport errors for
the retry primitive), src/config.py's preflight, and main.py's provenance
model-digest capture. No pipeline stage imports it.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import ollama

from src.errors import ConfigurationError
from src.reliability import call_with_retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    """One message in a chat exchange. Images ride on the message itself, not
    in provider-specific payload keys; adapters lift them into whatever shape
    their provider expects."""

    role: str
    content: str
    images: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class ChatResponse:
    """A provider-neutral completion. Carries the generation accounting the
    pipeline records in provenance and uses for typed overflow errors."""

    content: str
    model: str
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class AbstractChatModel(ABC):
    """The one interface a pipeline stage may hold a model through."""

    @abstractmethod
    def ensure_available(self) -> None:
        """Raise ConfigurationError if the model cannot serve requests."""

    @abstractmethod
    def invoke(self, messages: Sequence[ChatMessage], **options: object) -> ChatResponse:
        """Send one chat completion. Options are provider sampling controls
        (temperature, seed, num_ctx, ...) already validated by the config
        layer; adapters pass them through and never invent defaults."""

    @abstractmethod
    def capabilities(self) -> dict[str, bool]:
        """Declared capabilities ("vision", ...). Stages branch on these,
        never on adapter type (design principle 1)."""


class OllamaChatModel(AbstractChatModel):
    """Chat models served by the local Ollama daemon.

    auto_pull: pull the model from the registry when missing (the captioner
    models are public); when False a missing model is a ConfigurationError,
    with unavailable_hint appended for models that cannot be pulled (Ornith
    is not on the public registry).
    """

    def __init__(
        self,
        model_name: str,
        auto_pull: bool = False,
        description: str = "chat completion",
        unavailable_hint: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.auto_pull = auto_pull
        self.description = description
        self.unavailable_hint = unavailable_hint

    def ensure_available(self) -> None:
        try:
            local_models = {m.model for m in ollama.list().models}
        except ConnectionError as e:
            raise ConfigurationError(
                "Could not reach the local Ollama server. Make sure Ollama is installed "
                "and running (https://ollama.com/download), then try again."
            ) from e

        if any(
            name == self.model_name or name.startswith(f"{self.model_name}:")
            for name in local_models
        ):
            return

        if self.auto_pull:
            logger.info("Model %s not found locally, pulling via Ollama...", self.model_name)
            ollama.pull(self.model_name)
            return

        message = (
            f"Model '{self.model_name}' was not found in your local Ollama models "
            f"(`ollama list`) and auto-pull is disabled for it."
        )
        if self.unavailable_hint:
            message = f"{message} {self.unavailable_hint}"
        raise ConfigurationError(message)

    def invoke(self, messages: Sequence[ChatMessage], **options: object) -> ChatResponse:
        payload = []
        for message in messages:
            entry: dict = {"role": message.role, "content": message.content}
            if message.images:
                entry["images"] = list(message.images)
            payload.append(entry)

        response = call_with_retry(
            lambda: ollama.chat(model=self.model_name, messages=payload, options=dict(options)),
            description=f"{self.description} with {self.model_name}",
        )
        return ChatResponse(
            content=response.message.content or "",
            model=self.model_name,
            done_reason=getattr(response, "done_reason", None),
            prompt_eval_count=getattr(response, "prompt_eval_count", None),
            eval_count=getattr(response, "eval_count", None),
        )

    def capabilities(self) -> dict[str, bool]:
        return {"vision": True}

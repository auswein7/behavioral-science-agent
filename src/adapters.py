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
    pipeline records in provenance and uses for typed overflow errors.

    truncated is the normalized stop signal: True when the provider reported
    stopping on a context/length limit, False when it reported any other stop
    reason, None when the backend does not carry stop reasons (fairlib until
    fair_llm #147). Each adapter maps its provider's raw reason; consumers
    branch on this field and never string-match done_reason, which stays as
    the raw provider value for error text and provenance."""

    content: str
    model: str
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    truncated: bool | None = None


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


def ensure_ollama_model(
    model_name: str,
    auto_pull: bool = False,
    unavailable_hint: str | None = None,
) -> None:
    """Raise ConfigurationError unless model_name is served by the local
    Ollama daemon, pulling it first when auto_pull is set.

    A free function rather than a method because every backend that reaches
    models through the local daemon shares this probe: OllamaChatModel below,
    and the fairlib-backed model, whose adapter does not own availability and
    takes the probe as an explicit callable.
    """
    try:
        local_models = {m.model for m in ollama.list().models}
    except ConnectionError as e:
        raise ConfigurationError(
            "Could not reach the local Ollama server. Make sure Ollama is installed "
            "and running (https://ollama.com/download), then try again."
        ) from e

    if any(name == model_name or name.startswith(f"{model_name}:") for name in local_models):
        return

    if auto_pull:
        logger.info("Model %s not found locally, pulling via Ollama...", model_name)
        ollama.pull(model_name)
        return

    message = (
        f"Model '{model_name}' was not found in your local Ollama models "
        f"(`ollama list`) and auto-pull is disabled for it."
    )
    if unavailable_hint:
        message = f"{message} {unavailable_hint}"
    raise ConfigurationError(message)


@dataclass
class UsageTally:
    """Running totals of per-call usage across one stage's model calls.

    calls_reporting counts the calls whose backend actually returned token
    accounting; a tally with calls_reporting == 0 means usage is unknown for
    the stage, not zero (the same honest tri-state as ChatResponse)."""

    calls: int = 0
    calls_reporting: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class UsageRecordingModel(AbstractChatModel):
    """Delegating wrapper that tallies each reply's usage for run provenance.

    Cross-cutting accounting belongs to the adapter layer, not to stages
    (design principle 7): entry points wrap whichever backend the factory
    built, the stage sees only AbstractChatModel, and the tally feeds
    RunProvenance.stage_usage after the stage finishes."""

    def __init__(self, inner: AbstractChatModel) -> None:
        self._inner = inner
        self.tally = UsageTally()

    def ensure_available(self) -> None:
        self._inner.ensure_available()

    def invoke(self, messages: Sequence[ChatMessage], **options: object) -> ChatResponse:
        response = self._inner.invoke(messages, **options)
        self.tally.calls += 1
        if response.prompt_eval_count is not None or response.eval_count is not None:
            self.tally.calls_reporting += 1
            self.tally.prompt_tokens += response.prompt_eval_count or 0
            self.tally.completion_tokens += response.eval_count or 0
        return response

    def capabilities(self) -> dict[str, bool]:
        return self._inner.capabilities()


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
        ensure_ollama_model(
            self.model_name,
            auto_pull=self.auto_pull,
            unavailable_hint=self.unavailable_hint,
        )

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
        done_reason = getattr(response, "done_reason", None)
        return ChatResponse(
            content=response.message.content or "",
            model=self.model_name,
            done_reason=done_reason,
            prompt_eval_count=getattr(response, "prompt_eval_count", None),
            eval_count=getattr(response, "eval_count", None),
            # Ollama reports a context/length stop as the literal "length";
            # that provider knowledge lives here and nowhere upstream.
            truncated=None if done_reason is None else done_reason == "length",
        )

    def capabilities(self) -> dict[str, bool]:
        return {"vision": True}

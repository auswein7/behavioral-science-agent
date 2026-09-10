"""Backend selection: one config lever becomes one AbstractChatModel.

The CAPTION_BACKEND lever (CaptioningConfig.backend) picks which adapter the
captioning stage is handed. Entry points call build_caption_model and inject
the result; no stage ever learns which backend it got (design principle 1),
and the chosen backend is recorded in the run's provenance because it lives
on RunConfig (principle 8).

"ollama" is this repo's own adapter over the ollama SDK. "fairlib" is the
fairlib OllamaAdapter behind FairlibChatModel - the same daemon and model,
reached through the upstream fair_llm vision contract (issue #146); it
requires the vision-capable fair-llm install and raises ConfigurationError
without one. Both serve from the local Ollama daemon, so both share the same
availability probe; nothing about the lever changes what leaves the box.

Per-stage usage accounting is wired here too, because the factory is the one
place that knows the backend: a fairlib backend whose adapters emit
ModelInvocationEvent (fair_llm #170, PR #175) feeds the caller's UsageTally
from the framework's events; the built-in backend, and a fairlib that
predates the event contract, get the seam wrapper. The tally records which
source filled it.

The raw fairlib adapters the factories wrap are built by
fairlib_caption_adapter and fairlib_screenplay_adapter, exposed so the
conformance guard (tests/test_fairlib_conformance.py) runs fairlib's own
chat-model suite over exactly the objects production constructs, not a
look-alike; a construction that drifts from the guard is the drift the guard
exists to catch.
"""

import logging
from collections.abc import Callable
from functools import partial

from src.adapters import (
    AbstractChatModel,
    OllamaChatModel,
    UsageRecordingModel,
    UsageTally,
    ensure_ollama_model,
)
from src.egress import REMOTE_DESTINATIONS, EgressDecision
from src.errors import ConfigurationError, GateBlockedError
from src.fairlib_adapter import (
    FairlibChatModel,
    FairlibUsageSubscriber,
    fairlib_emits_invocation_events,
    fairlib_reports_usage,
)

logger = logging.getLogger(__name__)


def _fairlib_ollama_adapter_cls() -> type:
    """The fairlib OllamaAdapter class, or a typed error naming the install."""
    try:
        from fairlib.modules.mal.local_llama_adapter import OllamaAdapter
    except ImportError as e:
        raise ConfigurationError(
            "the fairlib backend needs a vision-capable fair-llm install "
            "(fair_llm issue #146); until it ships on PyPI, "
            "pip install -e <fair_llm checkout>. Or use the built-in "
            "'ollama' backend."
        ) from e
    return OllamaAdapter


def fairlib_caption_adapter(model_name: str) -> object:
    """The fairlib adapter the captioning stage's fairlib backend wraps: a
    vision-capable OllamaAdapter, so JPEG frames are carried, not refused."""
    return _fairlib_ollama_adapter_cls()(model_name=model_name, vision=True)


def fairlib_screenplay_adapter(model_name: str) -> object:
    """The fairlib adapter the screenplay stage's fairlib backend wraps.
    Ornith is text-only, so no vision; images sent by mistake get fairlib's
    own payload-time refusal (the upstream non-vision ruling)."""
    return _fairlib_ollama_adapter_cls()(model_name=model_name, vision=False)


def _with_seam_accounting(
    model: AbstractChatModel, tally: UsageTally | None
) -> AbstractChatModel:
    """Attach per-stage accounting to a backend that emits no accounting
    events of its own; no tally means the caller keeps no provenance."""
    if tally is None:
        return model
    return UsageRecordingModel(model, tally)


def _fairlib_model(
    adapter: object,
    model_name: str,
    availability_check: Callable[[], None],
    description: str,
    tally: UsageTally | None,
) -> AbstractChatModel:
    """Wrap one fairlib adapter at the seam, accounting from the framework's
    ModelInvocationEvent when the installed fairlib emits it.

    A fairlib that predates the event contract gets the seam wrapper instead.
    That is a degraded accounting source, so it is logged with the missing
    contract named and the tally's source records it for provenance - visible,
    never silent (design principle 6). The branch dies with the pre-#175
    fairlib once the fork pins a release that ships the event."""
    wrapper_tally = tally
    if tally is not None and fairlib_emits_invocation_events():
        FairlibUsageSubscriber(tally).bind(adapter)  # type: ignore[arg-type]
        wrapper_tally = None
    elif tally is not None:
        logger.warning(
            "installed fair-llm predates ModelInvocationEvent (fair_llm #170, "
            "PR #175): %s usage for %s is accounted by the seam wrapper, not "
            "by fairlib",
            description,
            model_name,
        )
    model = FairlibChatModel(
        adapter=adapter,  # type: ignore[arg-type]
        model_name=model_name,
        availability_check=availability_check,
        description=description,
    )
    return _with_seam_accounting(model, wrapper_tally)


def build_caption_model(
    backend: str, model_name: str, usage_tally: UsageTally | None = None
) -> AbstractChatModel:
    """Construct the captioning stage's chat model for the configured backend.

    backend is a str, not the schema's Literal, because CLI entry points read
    the lever straight from the environment; an unknown value raises below
    rather than falling back (design principle 6). usage_tally, when given,
    receives the stage's per-call accounting from whichever source the
    backend supports (module docstring)."""
    if backend == "fairlib":
        return _fairlib_model(
            adapter=fairlib_caption_adapter(model_name),
            model_name=model_name,
            availability_check=partial(ensure_ollama_model, model_name, auto_pull=True),
            description="frame caption",
            tally=usage_tally,
        )
    if backend == "ollama":
        return _with_seam_accounting(
            OllamaChatModel(model_name, auto_pull=True, description="frame caption"),
            usage_tally,
        )
    # Unreachable through load_run_config (the Literal validates there); this
    # guards entry points that read the lever straight from the environment.
    raise ConfigurationError(
        f"unknown caption backend '{backend}'; expected 'ollama' or 'fairlib'"
    )


def build_screenplay_model(
    backend: str,
    model_name: str,
    unavailable_hint: str | None = None,
    usage_tally: UsageTally | None = None,
) -> AbstractChatModel:
    """Construct the screenplay (Ornith) stage's chat model for the configured
    backend. Same str-not-Literal and usage_tally contract as
    build_caption_model.

    The fairlib backend is capability-gated on fair_llm #147: the stage's
    context-overflow diagnostic needs the reply's stop reason
    (ChatResponse.truncated), so an installed fairlib whose Message carries
    no usage telemetry is refused with the blocker named, never constructed
    into a silently degraded diagnostic (principle 6). The gate flips to
    construction by itself once the installed fair-llm ships #147.
    """
    if backend == "ollama":
        return _with_seam_accounting(
            OllamaChatModel(
                model_name,
                auto_pull=False,
                description="screenplay generation",
                unavailable_hint=unavailable_hint,
            ),
            usage_tally,
        )
    if backend == "fairlib":
        if not fairlib_reports_usage():
            raise ConfigurationError(
                "SCREENPLAY_BACKEND=fairlib is blocked on fair_llm #147 "
                "(usage telemetry): the installed fair-llm carries no stop "
                "reason on replies, and the screenplay stage's "
                "context-overflow diagnostic depends on it. Use "
                "SCREENPLAY_BACKEND=ollama, or install a fair-llm with #147."
            )
        return _fairlib_model(
            adapter=fairlib_screenplay_adapter(model_name),
            model_name=model_name,
            availability_check=partial(
                ensure_ollama_model,
                model_name,
                auto_pull=False,
                unavailable_hint=unavailable_hint,
            ),
            description="screenplay generation",
            tally=usage_tally,
        )
    raise ConfigurationError(
        f"unknown screenplay backend '{backend}'; expected 'ollama' or 'fairlib'"
    )


def fairlib_coder_adapter(model_name: str, *, num_ctx: int | None = None) -> object:
    """The fairlib adapter the coder agent runs on: a text-only OllamaAdapter.
    The coder is a fairlib SimpleAgent, so unlike the two stages above it
    takes the fairlib adapter itself, not the seam.

    Only the context size is set here. Sampling and the output budget are
    run options: the coder sends them with every call through
    SimpleAgent.arun(generation_options=) in fairlib's neutral vocabulary
    (fair_llm #186), where fairlib maps max_tokens onto Ollama's num_predict.
    num_ctx has no neutral counterpart (a hosted model's window is fixed), so
    it stays an adapter option of this backend."""
    options: dict[str, object] = {}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    return _fairlib_ollama_adapter_cls()(model_name=model_name, vision=False, options=options)


def fairlib_gemini_adapter(model_name: str) -> object:
    """The public Gemini adapter (fair_llm #186). Its endpoint is fixed by
    fairlib and cannot be moved at construction or per call; fairlib reads
    the key from GEMINI_API_KEY or GOOGLE_API_KEY. Construction is offline.

    Raises ConfigurationError when google-genai is missing or no key is set,
    before any call."""
    try:
        from fairlib.core.errors import FairlibError
        from fairlib.modules.mal.gemini_adapter import GeminiAdapter
    except ImportError as exc:
        raise ConfigurationError(
            "the Gemini coder needs fair-llm's Gemini adapter and google-genai"
        ) from exc
    try:
        return GeminiAdapter(model_name=model_name)
    except (FairlibError, ImportError) as exc:
        raise ConfigurationError(
            f"cannot construct the Gemini coder model ({exc}); set GEMINI_API_KEY"
        ) from exc


def build_coder_llm(
    provider: str,
    model_name: str,
    *,
    num_ctx: int | None = None,
    egress: EgressDecision | None = None,
) -> object:
    """Construct the coder stage's fairlib chat model for the configured
    provider. Usage accounting is not bound here: the coder's SimpleAgent
    owns the event bus and binds the model to it, so the tally attaches
    there (FairlibAgentCoder usage_tally). Sampling and the output budget are
    the coder's run options, not construction arguments.

    A remote provider is constructed only under an EgressDecision that
    authorized exactly this provider and model (ADR 0001): our gate decides
    first, and without its decision there is no remote model to call. A
    provider off the destination list is a GateBlockedError, not a fallback."""
    if provider == "ollama":
        return fairlib_coder_adapter(model_name, num_ctx=num_ctx)
    if provider in {"anthropic", "openai"}:
        raise GateBlockedError(
            f"CODER_PROVIDER={provider} is not an allowed egress destination "
            f"(ADR 0001 ruling 2 allows {sorted(REMOTE_DESTINATIONS)})"
        )
    if provider not in REMOTE_DESTINATIONS:
        raise ConfigurationError(
            f"unknown coder provider '{provider}'; expected 'ollama' or 'gemini'"
        )
    if egress is None or not egress.remote or (egress.provider, egress.model) != (
        provider,
        model_name,
    ):
        raise GateBlockedError(
            f"no egress decision authorizes {provider}/{model_name}; the egress "
            "gate (src.egress.decide_egress) must pass first (ADR 0001)"
        )
    if num_ctx is not None:
        raise ConfigurationError(
            "CODER_NUM_CTX is an Ollama option; a hosted model's context window is fixed"
        )
    return fairlib_gemini_adapter(model_name)

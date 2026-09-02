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
from src.errors import ConfigurationError
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
            adapter=_fairlib_ollama_adapter_cls()(model_name=model_name, vision=True),
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
        # Ornith is text-only, so no vision; images sent by mistake get
        # fairlib's own payload-time refusal.
        return _fairlib_model(
            adapter=_fairlib_ollama_adapter_cls()(model_name=model_name, vision=False),
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

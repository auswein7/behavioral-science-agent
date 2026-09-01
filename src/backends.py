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
"""

from functools import partial

from src.adapters import AbstractChatModel, OllamaChatModel, ensure_ollama_model
from src.errors import ConfigurationError
from src.fairlib_adapter import FairlibChatModel, fairlib_reports_usage


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


def build_caption_model(backend: str, model_name: str) -> AbstractChatModel:
    """Construct the captioning stage's chat model for the configured backend.

    backend is a str, not the schema's Literal, because CLI entry points read
    the lever straight from the environment; an unknown value raises below
    rather than falling back (design principle 6)."""
    if backend == "fairlib":
        return FairlibChatModel(
            adapter=_fairlib_ollama_adapter_cls()(model_name=model_name, vision=True),
            model_name=model_name,
            availability_check=partial(ensure_ollama_model, model_name, auto_pull=True),
            description="frame caption",
        )
    if backend == "ollama":
        return OllamaChatModel(model_name, auto_pull=True, description="frame caption")
    # Unreachable through load_run_config (the Literal validates there); this
    # guards entry points that read the lever straight from the environment.
    raise ConfigurationError(
        f"unknown caption backend '{backend}'; expected 'ollama' or 'fairlib'"
    )


def build_screenplay_model(
    backend: str, model_name: str, unavailable_hint: str | None = None
) -> AbstractChatModel:
    """Construct the screenplay (Ornith) stage's chat model for the configured
    backend. Same str-not-Literal contract as build_caption_model.

    The fairlib backend is capability-gated on fair_llm #147: the stage's
    context-overflow diagnostic needs the reply's stop reason
    (ChatResponse.truncated), so an installed fairlib whose Message carries
    no usage telemetry is refused with the blocker named, never constructed
    into a silently degraded diagnostic (principle 6). The gate flips to
    construction by itself once the installed fair-llm ships #147.
    """
    if backend == "ollama":
        return OllamaChatModel(
            model_name,
            auto_pull=False,
            description="screenplay generation",
            unavailable_hint=unavailable_hint,
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
        return FairlibChatModel(
            adapter=_fairlib_ollama_adapter_cls()(model_name=model_name, vision=False),
            model_name=model_name,
            availability_check=partial(
                ensure_ollama_model,
                model_name,
                auto_pull=False,
                unavailable_hint=unavailable_hint,
            ),
            description="screenplay generation",
        )
    raise ConfigurationError(
        f"unknown screenplay backend '{backend}'; expected 'ollama' or 'fairlib'"
    )

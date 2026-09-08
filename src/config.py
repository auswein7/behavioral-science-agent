"""Typed configuration: the one place environment variables become a RunConfig.

Centralized configuration (workspace design principle 8): entry points call
load_run_config() after load_dotenv(); nothing else in the pipeline reads
os.environ. preflight() then fails fast - a missing token, model, or GPU is a
ConfigurationError at startup, not a crash five hours in (run 1 died at stage
5 because the Ornith model was missing from Ollama; preflight exists so that
class of failure costs seconds, not a workday).

HF_TOKEN is deliberately NOT part of RunConfig: the config is embedded
verbatim in the delivered RunProvenance, and secrets never go there.
"""

import hashlib
import logging
import os
from collections.abc import Mapping

from pydantic import ValidationError

from src import prompts
from src.errors import ConfigurationError
from src.schemas import (
    CaptioningConfig,
    CoderConfig,
    RunConfig,
    SamplingConfig,
    ScreenplayConfig,
    ToneConfig,
    TranscriptionConfig,
)

logger = logging.getLogger(__name__)

# Prompt assets whose exact text a run must be reproducible from; hashed into
# RunConfig.prompt_versions (a content digest is a version id nobody forgets to bump).
_PROMPT_ASSETS = (
    "ANONYMIZATION_RULES",
    "CAPTION_ANONYMIZATION_RULES",
    "FIRST_FRAME_PROMPT",
    "CHANGE_PROMPT",
    "CODER_ROLE_PROMPT",
)


def _optional_int(value: str | None) -> int | None:
    return int(value) if value else None


def prompt_versions() -> dict[str, str]:
    """Content digests of every prompt asset, recorded in config and provenance."""
    versions = {}
    for name in _PROMPT_ASSETS:
        text = getattr(prompts, name, None)
        if isinstance(text, str):
            versions[name] = "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]
    return versions


def load_run_config(env: Mapping[str, str] | None = None) -> RunConfig:
    """Build the typed RunConfig from an environment mapping (os.environ by
    default). Any malformed value is a ConfigurationError naming the problem."""
    if env is None:
        env = os.environ

    context_raw = env.get("CAPTION_CONTEXT_CAPTIONS", "1").strip()
    max_dimension_raw = env.get("CAPTION_MAX_DIMENSION", "1280").strip()

    try:
        return RunConfig(
            transcription=TranscriptionConfig(
                whisper_model=env.get("WHISPERX_MODEL", "large-v2"),
                device=env.get("WHISPERX_DEVICE", "cpu"),
                compute_type=env.get("WHISPERX_COMPUTE_TYPE", "int8"),
                batch_size=int(env.get("WHISPERX_BATCH_SIZE", "16")),
                language=env.get("WHISPERX_LANGUAGE", "en"),
                num_speakers=_optional_int(env.get("WHISPERX_NUM_SPEAKERS")),
                min_speakers=_optional_int(env.get("WHISPERX_MIN_SPEAKERS")),
                max_speakers=_optional_int(env.get("WHISPERX_MAX_SPEAKERS")),
            ),
            tone=ToneConfig(tone_model=env.get("TONE_MODEL", "iic/emotion2vec_plus_large")),
            captioning=CaptioningConfig(
                caption_model=env.get("CAPTION_MODEL", "qwen3-vl:8b"),
                backend=env.get("CAPTION_BACKEND", "ollama"),
                fps=float(env.get("CAPTION_FPS", "1.0")),
                context_captions=None if context_raw.lower() == "all" else int(context_raw),
                max_dimension=None if max_dimension_raw.lower() == "none" else int(max_dimension_raw),
                speech_frame_offset=float(env.get("SPEECH_FRAME_OFFSET", "0.5")),
                burst_frames=int(env.get("SPEECH_BURST_FRAMES", "2")),
                burst_spacing=float(env.get("SPEECH_BURST_SPACING", "0.2")),
            ),
            screenplay=ScreenplayConfig(
                ornith_model=env.get("ORNITH_MODEL", "ornith-1.5-255k"),
                backend=env.get("SCREENPLAY_BACKEND", "ollama"),
                num_ctx=_optional_int(env.get("ORNITH_NUM_CTX")),
            ),
            sampling=SamplingConfig(
                temperature=float(env.get("SAMPLING_TEMPERATURE", "0.0")),
                seed=_optional_int(env.get("SAMPLING_SEED")),
            ),
            prompt_versions=prompt_versions(),
        )
    except (ValueError, ValidationError) as exc:
        raise ConfigurationError(f"invalid run configuration: {exc}") from exc


def _ollama_model_names() -> set[str]:
    import ollama

    try:
        listing = ollama.list()
    except Exception as exc:
        raise ConfigurationError(
            "Ollama is not reachable - is the service running? (`ollama serve`)"
        ) from exc
    return {m.model for m in listing.models}


def _model_present(name: str, available: set[str]) -> bool:
    return any(name == a or a.split(":")[0] == name for a in available)


def preflight(
    config: RunConfig,
    hf_token: str | None,
    need_transcription: bool = True,
    need_captioning: bool = True,
) -> None:
    """Validate the run can actually finish before any expensive work starts.

    Checks: HF token present and CUDA visible (when transcription runs on
    cuda), Ollama reachable, the Ornith model present (it cannot be
    auto-pulled), the caption model present or pullable (warn only - the
    captioning stage auto-pulls). Raises ConfigurationError on any hard miss.
    """
    if need_transcription:
        if not hf_token:
            raise ConfigurationError("HF_TOKEN not set - add it to .env (pyannote diarization needs it)")
        if config.transcription.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ConfigurationError(
                    "WHISPERX_DEVICE=cuda but no CUDA device is visible to torch"
                )

    available = _ollama_model_names()
    if not _model_present(config.screenplay.ornith_model, available):
        raise ConfigurationError(
            f"Ornith model '{config.screenplay.ornith_model}' is not in `ollama list` and cannot "
            f"be auto-pulled - create it first (see Modelfile.ornith)"
        )
    if need_captioning and not _model_present(config.captioning.caption_model, available):
        logger.info(
            "Caption model '%s' not present locally; the captioning stage will try to pull it",
            config.captioning.caption_model,
        )
    logger.info("Preflight passed: config valid, Ollama reachable, required models present")


def load_coder_config(env: Mapping[str, str] | None = None) -> CoderConfig:
    """Build the typed CoderConfig from an environment mapping (os.environ by
    default). Same contract as load_run_config: any malformed value is a
    ConfigurationError naming the problem; nothing else reads the levers."""
    if env is None:
        env = os.environ
    try:
        return CoderConfig(
            coder_model=env.get("CODER_MODEL", "qwen2.5:14b"),
            provider=env.get("CODER_PROVIDER", "ollama"),
            coder_id=env.get("CODER_ID", "fairlib_local"),
            context_utterances=int(env.get("CODER_CONTEXT_UTTERANCES", "5")),
            max_retries=int(env.get("CODER_MAX_RETRIES", "2")),
            max_steps=int(env.get("CODER_MAX_STEPS", "3")),
            num_ctx=_optional_int(env.get("CODER_NUM_CTX")),
            max_tokens=_optional_int(env.get("CODER_MAX_TOKENS")),
            temperature=float(env.get("SAMPLING_TEMPERATURE", "0.0")),
            seed=_optional_int(env.get("SAMPLING_SEED")),
        )
    except (ValueError, ValidationError) as exc:
        raise ConfigurationError(f"invalid coder configuration: {exc}") from exc

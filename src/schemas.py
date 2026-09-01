"""Pydantic models for every stage boundary, mirroring docs/DATA_MODEL.md.

The doc is the spec; these models are its executable form. Section numbers in
class docstrings refer to it. Intermediate (Tier B) models tolerate unknown
fields so existing stage outputs load unchanged; deliverable (Tier C) models
forbid them so nothing undeclared can leave the box.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1"

Role = Literal["participant", "confederate", "experimenter"]
SourceClass = Literal["cadet_pii", "public_domain"]
Trigger = Literal["fixed_interval", "speech_start"]
ScrubResolution = Literal["clean", "cleared_by_review", "redacted", "blocked"]

# Closed tone label set (DATA_MODEL 5.3): emotion2vec+ labels plus "unknown",
# which normalize_tone maps from a null or "<unk>" classification.
TONE_LABELS = frozenset(
    {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised", "other", "unknown"}
)


def normalize_tone(emotion: str | None) -> str:
    """Map a raw classifier label to the closed deliverable set (DATA_MODEL 5.3)."""
    if emotion is None or emotion not in TONE_LABELS:
        return "unknown"
    return emotion


class _Internal(BaseModel):
    """Base for Tier B intermediates: immutable, tolerant of extra fields."""

    model_config = ConfigDict(frozen=True, extra="ignore")


class _Deliverable(BaseModel):
    """Base for Tier C deliverables: immutable, undeclared fields forbidden."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SessionManifest(_Deliverable):
    """Per-session human-written metadata (DATA_MODEL 2.2). Must contain no PII."""

    schema_version: str = SCHEMA_VERSION
    session_id: str
    source_class: SourceClass
    task_name: str
    group_size: int = Field(ge=1)
    expected_speaker_count: int = Field(ge=1)
    speaker_roles: dict[str, Role] | None = None
    protocol_ref: str | None = None
    notes: str | None = None


class TranscriptionConfig(_Deliverable):
    whisper_model: str
    device: str
    compute_type: str
    batch_size: int
    language: str
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class ToneConfig(_Deliverable):
    tone_model: str


class CaptioningConfig(_Deliverable):
    caption_model: str
    backend: Literal["ollama", "fairlib"] = "ollama"
    fps: float
    context_captions: int | None
    max_dimension: int | None
    speech_frame_offset: float
    burst_frames: int
    burst_spacing: float


class ScreenplayConfig(_Deliverable):
    ornith_model: str
    backend: Literal["ollama", "fairlib"] = "ollama"
    num_ctx: int | None = None


class SamplingConfig(_Deliverable):
    temperature: float = 0.0
    seed: int | None = None


class RunConfig(_Deliverable):
    """Typed configuration for one run (DATA_MODEL 2.3); the .env file feeds this."""

    schema_version: str = SCHEMA_VERSION
    transcription: TranscriptionConfig
    tone: ToneConfig
    captioning: CaptioningConfig
    screenplay: ScreenplayConfig
    sampling: SamplingConfig = SamplingConfig()
    prompt_versions: dict[str, str] = Field(default_factory=dict)


class TranscriptRecord(_Internal):
    """One diarized utterance (DATA_MODEL 3.1). Tier B: text is verbatim speech."""

    index: int
    start: float
    end: float
    start_hms: str
    end_hms: str
    speaker: str
    text: str
    word_count: int
    avg_word_score: float | None = None
    low_confidence_words: list[str] = Field(default_factory=list)


class ToneRecord(TranscriptRecord):
    """TranscriptRecord plus the tone classification (DATA_MODEL 3.2)."""

    emotion: str | None = None
    emotion_scores: dict[str, float] | None = None


class CaptionRecord(_Internal):
    """One captioned frame (DATA_MODEL 3.3). Tier B: caption prose can leak appearance."""

    index: int
    timestamp: float
    timestamp_hms: str
    caption: str
    trigger: Trigger = "fixed_interval"
    mentioned_speakers: list[str] = Field(default_factory=list)
    speech_start: float | None = None
    burst_captions: list | None = None


class SpeechEvent(_Internal):
    """Merged-timeline speech event (DATA_MODEL 3.4)."""

    type: Literal["speech"]
    index: int
    timestamp: float
    start: float
    end: float
    start_hms: str
    end_hms: str
    speaker: str
    text: str
    emotion: str | None = None


class VisualEvent(_Internal):
    """Merged-timeline visual event (DATA_MODEL 3.4)."""

    type: Literal["visual"]
    index: int
    timestamp: float
    timestamp_hms: str
    caption: str
    trigger: Trigger = "fixed_interval"
    mentioned_speakers: list[str] = Field(default_factory=list)


MergedEvent = SpeechEvent | VisualEvent


class UtteranceRow(_Deliverable):
    """One row of the primary deliverable table (DATA_MODEL 4.1).

    Column order here is the delivered CSV column order: OSU's columns first,
    our added channels after. uid follows OSU's u-prefixed convention and is
    unique within a document only; prior fields are the literal string NA on
    the first row, matching their sample.
    """

    document: str
    uid: str
    ord: int = Field(ge=1)
    speaker: str
    utterance: str
    time: str
    end_time: str
    filename: str
    prior_utterance: str
    prior_speaker: str
    role: Role
    tone: str
    low_confidence: bool
    nonverbal_notes: str


class NameReplacement(_Internal):
    """One span replaced by the deterministic name scrub. Tier B: carries the name."""

    record_index: int
    entity: str
    label: str
    replacement: str


class GateFinding(_Internal):
    """One internal gate finding (DATA_MODEL 4.4 internal form). Tier B: carries excerpts."""

    category: Literal["gendered", "appearance", "likely_name"]
    term: str
    position: int
    excerpt: str


class RunProvenance(_Deliverable):
    """Reproducibility record for one run (DATA_MODEL 4.3)."""

    schema_version: str = SCHEMA_VERSION
    session_id: str
    run_id: str
    started: str
    finished: str
    host: str
    input_video_sha256: str
    config: RunConfig
    model_digests: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    software: dict[str, str] = Field(default_factory=dict)
    stage_timings: dict[str, float] = Field(default_factory=dict)


class ScrubReport(_Deliverable):
    """Delivered gate record (DATA_MODEL 4.4): counts and the decision, never excerpts."""

    schema_version: str = SCHEMA_VERSION
    session_id: str
    run_id: str
    artifact: str
    findings_by_category: dict[str, int] = Field(default_factory=dict)
    resolution: ScrubResolution
    reviewer: str | None = None
    decided: str

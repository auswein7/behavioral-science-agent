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
# osu_study: a session imported from OSU's own study transcripts. Human-subjects
# data like cadet_pii, so never cleared for egress; kept distinct so the run
# record says where the rows came from.
SourceClass = Literal["cadet_pii", "public_domain", "osu_study"]
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


class CoderConfig(_Deliverable):
    """Typed configuration for the coder stage (DATA_MODEL 4.5); the .env
    file feeds this. provider names the fairlib adapter family the agent's
    model comes from: ollama is local; gemini is the one remote destination
    ADR 0001 allows, and it runs only when the egress gate authorizes it."""

    coder_model: str
    provider: Literal["ollama", "gemini"] = "ollama"
    coder_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_]+$")
    context_utterances: int = Field(default=5, ge=0)
    max_retries: int = Field(default=2, ge=0)
    max_steps: int = Field(default=3, ge=1)
    # num_ctx is Ollama's construction-time context option and has no
    # provider-neutral counterpart (Gemini's window is fixed by the model),
    # so it stays adapter-scoped. max_tokens is the output budget and IS
    # neutral - it is the lever a truncated reply needs raised on retry.
    num_ctx: int | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 0.0
    seed: int | None = None
    # Path of the codebook document (CODER_CODEBOOK); None is the placeholder.
    codebook_path: str | None = None


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


CodeValue = Literal[0, 1]
CodeCell = Literal[0, 1, "NA"]


class CodeJudgment(_Deliverable):
    """The model's decision on one code for one utterance (DATA_MODEL 4.5):
    a binary value and the free-text rationale OSU asked for."""

    value: CodeValue
    rationale: str = Field(min_length=1)


class UtteranceCoding(_Deliverable):
    """The coder model's validated output for one utterance: one judgment
    per code in the codebook, keyed by code. The validator, not this model,
    checks that the key set equals the codebook's, because the code set is
    pluggable (DATA_MODEL 4.5) and this schema must not hardcode it."""

    uid: str
    judgments: dict[str, CodeJudgment]


class CodedUtteranceRow(UtteranceRow):
    """UtteranceRow plus the coder stage's columns (DATA_MODEL 4.5).

    codes holds one cell per codebook code: 0 or 1 from the model, or the
    literal NA for rows the coder never sends to a model (confederate and
    experimenter roles, following OSU's sample). rationales has one entry
    per coded code and is empty for NA rows. The CSV writer flattens codes
    into <CODE>_<coder_id> columns so several coders can sit side by side.
    """

    codes: dict[str, CodeCell]
    rationales: dict[str, str]
    coder_id: str
    coder_model: str
    coder_prompt_version: str
    codebook_version: str


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


class StageUsage(_Deliverable):
    """Aggregated model-call usage for one stage (DATA_MODEL 4.3).

    Token totals are None when no call in the stage reported accounting
    (unknown, not zero); calls_reporting says how many did, so a partial
    total is legible as partial. calls counts attempts and calls_failed the
    attempts that raised. source says who counted: the framework's own
    ModelInvocationEvent stream (fairlib_events, fair_llm #170) or this
    repo's reply-side wrapper (seam_wrapper); None on records written before
    the field existed."""

    calls: int
    calls_reporting: int
    calls_failed: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    source: Literal["fairlib_events", "seam_wrapper"] | None = None


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
    stage_usage: dict[str, StageUsage] = Field(default_factory=dict)


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

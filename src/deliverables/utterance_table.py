"""Build the per-utterance deliverable table (DATA_MODEL 4.1) from stage outputs.

The builder is deterministic: no model calls, pure projection and joining of
the tone-enriched transcript and the caption records. The utterance text
passes through unchanged for now - the deterministic spoken-name scrub (TODO
2.5) plugs in upstream of this module when it exists, and the scrub gate still
applies to the emitted files either way.
"""

import csv
import json
import logging
from pathlib import Path

from src.errors import DeliverableError
from src.schemas import (
    SCHEMA_VERSION,
    CaptionRecord,
    SessionManifest,
    ToneRecord,
    UtteranceRow,
    normalize_tone,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "deliverables"

# Below this mean word-alignment score an utterance is flagged low_confidence
# in the delivered table (the word lists themselves stay Tier B).
LOW_CONFIDENCE_THRESHOLD = 0.5

NA = "NA"


def _uid(ord_: int) -> str:
    """OSU's u-prefixed uid, zero-padded to 3 digits, widened past u999."""
    return f"u{ord_:04d}" if ord_ > 999 else f"u{ord_:03d}"


def _nonverbal_notes(record: ToneRecord, captions: list[CaptionRecord]) -> str:
    """Deterministic sourcing rule from DATA_MODEL 4.1: this utterance's own
    speech_start caption (matched exactly on the originating timestamp), then
    fixed_interval captions falling in [start, end), joined in time order."""
    picked = []
    for cap in captions:
        if cap.trigger == "speech_start":
            if cap.speech_start is not None and round(cap.speech_start, 3) == round(record.start, 3):
                picked.append(cap)
        elif record.start <= cap.timestamp < record.end:
            picked.append(cap)
    picked.sort(key=lambda c: c.timestamp)
    return " ".join(c.caption for c in picked)


def build_utterance_rows(
    transcript: list[ToneRecord],
    captions: list[CaptionRecord],
    manifest: SessionManifest,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> list[UtteranceRow]:
    """Project tone-enriched transcript records into UtteranceRows, joining in
    caption prose as nonverbal_notes and roles from the manifest."""
    if not transcript:
        raise DeliverableError("cannot build an utterance table from an empty transcript")

    roles = manifest.speaker_roles or {}
    filename = f"{manifest.session_id}.utterances.csv"
    ordered = sorted(transcript, key=lambda r: r.start)

    rows: list[UtteranceRow] = []
    for i, record in enumerate(ordered):
        ord_ = i + 1
        prior = ordered[i - 1] if i > 0 else None
        rows.append(
            UtteranceRow(
                document=manifest.session_id,
                uid=_uid(ord_),
                ord=ord_,
                speaker=record.speaker,
                utterance=record.text,
                time=record.start_hms,
                end_time=record.end_hms,
                filename=filename,
                prior_utterance=prior.text if prior else NA,
                prior_speaker=prior.speaker if prior else NA,
                role=roles.get(record.speaker, "participant"),
                tone=normalize_tone(record.emotion),
                low_confidence=record.avg_word_score < low_confidence_threshold,
                nonverbal_notes=_nonverbal_notes(record, captions),
            )
        )
    return rows


def write_csv(rows: list[UtteranceRow], session_id: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write the delivered CSV: UTF-8, every field quoted so spreadsheet imports
    cannot coerce the time columns (the fate of OSU's own sample, DATA_MODEL 4.1)."""
    if not rows:
        raise DeliverableError("refusing to write an empty utterance table")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{session_id}.utterances.csv"
    fields = list(UtteranceRow.model_fields)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())
    logger.info("Wrote %d utterance rows to %s", len(rows), output_path)
    return output_path


def write_json(rows: list[UtteranceRow], session_id: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write the JSON form: a schema_version envelope around the same rows."""
    if not rows:
        raise DeliverableError("refusing to write an empty utterance table")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{session_id}.utterances.json"
    envelope = {"schema_version": SCHEMA_VERSION, "rows": [r.model_dump() for r in rows]}
    output_path.write_text(json.dumps(envelope, indent=2))
    logger.info("Wrote %d utterance rows to %s", len(rows), output_path)
    return output_path

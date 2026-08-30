"""Merge a video-captions JSON and an audio-transcript JSON into one chronological timeline."""

import logging
from pathlib import Path

from src.persist import write_records

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "screenplays"


def merge_screenplay(captions: list[dict], transcript: list[dict]) -> list[dict]:
    """Combine caption records (point-in-time) and transcript records (start/end spans)
    into a single list of events sorted by timestamp.

    Each event has a "type" of "visual" or "speech" plus the fields from its source
    record (renamed to a common "timestamp"/"timestamp_hms" for captions and
    "start"/"end"/"start_hms"/"end_hms" for transcript utterances). `transcript` is
    expected to be tone-enriched (src/tone/classify_tone.classify_tone output) — each
    speech event carries an "emotion" field (None if the record wasn't tone-classified).
    `captions` (src/captioning/caption_video) combines fixed-interval frames and
    speech-triggered "reaction shot" frames into one chronologically-captioned
    pass — each visual event carries a "trigger" field ("fixed_interval" or
    "speech_start") so consumers can tell the two apart, plus a "mentioned_speakers"
    field (any SPEAKER_NN tags found in the caption text) that write_screenplay's
    Ornith prompt uses to tell a grounded speaker attribution apart from one it would
    have to invent.
    """
    events = []

    for record in captions:
        events.append(
            {
                "type": "visual",
                "timestamp": record["timestamp"],
                "timestamp_hms": record["timestamp_hms"],
                "caption": record["caption"],
                "trigger": record.get("trigger", "fixed_interval"),
                "mentioned_speakers": record.get("mentioned_speakers", []),
            }
        )

    for record in transcript:
        events.append(
            {
                "type": "speech",
                "timestamp": record["start"],
                "start": record["start"],
                "end": record["end"],
                "start_hms": record["start_hms"],
                "end_hms": record["end_hms"],
                "speaker": record["speaker"],
                "text": record["text"],
                "emotion": record.get("emotion"),
            }
        )

    events.sort(key=lambda e: e["timestamp"])
    for i, event in enumerate(events):
        event["index"] = i

    return events


def write_json(events: list[dict], name: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.screenplay.json"
    write_records(events, output_path)
    logger.info("Wrote %d events to %s", len(events), output_path)
    return output_path

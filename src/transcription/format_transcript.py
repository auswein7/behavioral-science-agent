"""Flatten WhisperX diarized segments into a clean, per-utterance record for downstream use."""

import logging
from pathlib import Path
from typing import Any

from src.persist import write_records

logger = logging.getLogger(__name__)


def format_segments(segments: list[dict]) -> list[dict]:
    """Convert raw WhisperX segments into flat utterance records.

    Each record: index, start/end (seconds + hh:mm:ss), speaker, text,
    word_count, avg_word_score (mean per-word confidence, when word-level
    data is present), and low_confidence_words (words scoring below 0.5,
    useful for flagging likely transcription errors).
    """

    def _to_native(value: Any) -> Any:
        """Unwrap numpy scalars (e.g. np.float64) so results are plain-JSON-serializable."""
        if hasattr(value, "item"):
            return value.item()
        return value
        

    def _hhmmss(seconds: float) -> str:
        # Work in whole milliseconds so a fraction that rounds up carries into
        # the seconds place instead of rendering as ".1000".
        total_ms = round(seconds * 1000)
        whole, ms = divmod(total_ms, 1000)
        hh, rem = divmod(whole, 3600)
        mm, ss = divmod(rem, 60)
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


    records = []
    for i, seg in enumerate(segments):
        words = seg.get("words", [])
        scores = [_to_native(w["score"]) for w in words if "score" in w]
        avg_score = round(sum(scores) / len(scores), 3) if scores else None
        low_conf = [w["word"] for w in words if "score" in w and _to_native(w["score"]) < 0.5]

        start = _to_native(seg["start"])
        end = _to_native(seg["end"])

        records.append(
            {
                "index": i,
                "start": round(start, 3),
                "end": round(end, 3),
                "start_hms": _hhmmss(start),
                "end_hms": _hhmmss(end),
                "speaker": seg.get("speaker", "UNKNOWN"),
                "text": seg["text"].strip(),
                "word_count": len(words),
                "avg_word_score": avg_score,
                "low_confidence_words": low_conf,
            }
        )
    return records


def write_json(records: list[dict], name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.formatted.json"
    write_records(records, output_path)
    logger.info("Wrote %d records to %s", len(records), output_path)
    return output_path

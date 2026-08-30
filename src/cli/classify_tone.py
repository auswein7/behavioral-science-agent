"""CLI entry point: add per-utterance emotional tone classification to a diarized transcript.

Takes an existing formatted transcript JSON (src/cli/transcribe.py's output) plus the
source video (to re-decode audio for the per-utterance clips), and writes a copy of
the transcript with "emotion"/"emotion_scores" added to each record.

This module is the only place that reads environment variables or resolves
default paths — everything under src/ is a plain library that takes what it
needs as arguments.

Run from the repo root: python -m src.cli.classify_tone <video_filename_or_path> <transcript_json>
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.persist import read_records
from src.tone import classify_tone, write_json
from src.video_utils import load_audio_numpy_array

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_VIDEOS_DIR = REPO_ROOT / "data" / "raw_videos"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "transcripts"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m src.cli.classify_tone <video_filename_or_path> <transcript_json>")
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        video_path = RAW_VIDEOS_DIR / video_path
    transcript_path = Path(sys.argv[2])
    if not transcript_path.exists():
        raise SystemExit(f"Transcript JSON not found: {transcript_path}")

    records = read_records(transcript_path)

    logger.info("Starting tone classification for: %s", video_path)
    audio = load_audio_numpy_array(video_path)
    enriched = classify_tone(
        records,
        audio,
        model_name=os.environ.get("TONE_MODEL", "iic/emotion2vec_plus_large"),
    )
    output_path = write_json(enriched, video_path.stem, DEFAULT_OUTPUT_DIR)

    logger.info("Tone classification complete: %s", output_path)


if __name__ == "__main__":
    main()

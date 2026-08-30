"""CLI entry point: extract audio from a video and run the diarized transcript pipeline.

This module is the only place that reads environment variables or resolves
default paths — everything under src/ is a plain library that takes what it
needs as arguments. The video file is the only thing passed on the command
line; other settings are tunable via .env (see .env.example).

Run from the repo root: python -m src.cli.transcribe <video_filename_or_path>
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.transcription import diarize_transcript, format_segments, write_json
from src.video_utils import load_audio_numpy_array

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_VIDEOS_DIR = REPO_ROOT / "data" / "raw_videos"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "transcripts"


def _optional_int(value: str | None) -> int | None:
    return int(value) if value else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.cli.transcribe <video_filename_or_path>")
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        video_path = RAW_VIDEOS_DIR / video_path

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set. Add it to .env.")

    logger.info("Starting transcription pipeline for: %s", video_path)
    audio = load_audio_numpy_array(video_path)
    result = diarize_transcript(
        audio,
        hf_token=hf_token,
        device=os.environ.get("WHISPERX_DEVICE", "cpu"),
        compute_type=os.environ.get("WHISPERX_COMPUTE_TYPE", "int8"),
        whisper_model=os.environ.get("WHISPERX_MODEL", "large-v2"),
        batch_size=int(os.environ.get("WHISPERX_BATCH_SIZE", "16")),
        num_speakers=_optional_int(os.environ.get("WHISPERX_NUM_SPEAKERS")),
        min_speakers=_optional_int(os.environ.get("WHISPERX_MIN_SPEAKERS")),
        max_speakers=_optional_int(os.environ.get("WHISPERX_MAX_SPEAKERS")),
        language=os.environ.get("WHISPERX_LANGUAGE", "en"),
    )
    records = format_segments(result["segments"])
    output_path = write_json(records, video_path.stem, DEFAULT_OUTPUT_DIR)

    logger.info("Transcription pipeline complete: %s", output_path)


if __name__ == "__main__":
    main()

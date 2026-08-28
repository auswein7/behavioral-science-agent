"""CLI entry point: load a video and caption its sampled frames.

This module is the only place that reads environment variables or resolves
default paths — everything under src/ is a plain library that takes what it
needs as arguments. The video file is the only thing passed on the command
line; other settings are tunable via .env (see .env.example).

Run from the repo root: python -m src.cli.caption_video <video_filename_or_path>
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.captioning import caption_video, write_json

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_VIDEOS_DIR = REPO_ROOT / "data" / "raw_videos"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "captions"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python -m src.cli.caption_video <video_filename_or_path>")
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        video_path = RAW_VIDEOS_DIR / video_path

    context_raw = os.environ.get("CAPTION_CONTEXT_CAPTIONS", "1").strip()
    context_captions = None if context_raw.lower() == "all" else int(context_raw)
    max_dimension_raw = os.environ.get("CAPTION_MAX_DIMENSION", "1280").strip()
    max_dimension = None if max_dimension_raw.lower() == "none" else int(max_dimension_raw)

    logger.info("Starting captioning pipeline for: %s", video_path)
    records = caption_video(
        video_path,
        fps=float(os.environ.get("CAPTION_FPS", "1.0")),
        model_name=os.environ.get("CAPTION_MODEL", "qwen3-vl:8b"),
        context_captions=context_captions,
        max_dimension=max_dimension,
    )
    output_path = write_json(records, video_path.stem, DEFAULT_OUTPUT_DIR)

    logger.info("Captioning pipeline complete: %s", output_path)


if __name__ == "__main__":
    main()

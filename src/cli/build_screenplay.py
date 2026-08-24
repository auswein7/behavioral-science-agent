"""CLI entry point: merge a captions JSON and a transcript JSON into one screenplay JSON.

This module is the only place that resolves default paths — everything under src/
is a plain library that takes what it needs as arguments.

Run from the repo root: python -m src.cli.build_screenplay <captions_json> <transcript_json>
"""

import json
import logging
import sys
from pathlib import Path

from src.screenplay import merge_screenplay, write_json

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTIONS_DIR = REPO_ROOT / "data" / "captions"
TRANSCRIPTS_DIR = REPO_ROOT / "data" / "transcripts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "screenplays"


def _resolve(path_str: str, default_dir: Path) -> Path:
    path = Path(path_str)
    if not path.exists():
        path = default_dir / path_str
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: python -m src.cli.build_screenplay <captions_json> <transcript_json>")

    captions_path = _resolve(sys.argv[1], CAPTIONS_DIR)
    transcript_path = _resolve(sys.argv[2], TRANSCRIPTS_DIR)

    captions = json.loads(captions_path.read_text())
    transcript = json.loads(transcript_path.read_text())

    logger.info("Merging %s and %s", captions_path, transcript_path)
    events = merge_screenplay(captions, transcript)

    name = captions_path.stem.removesuffix(".captions")
    output_path = write_json(events, name, DEFAULT_OUTPUT_DIR)

    logger.info("Screenplay merge complete: %s", output_path)


if __name__ == "__main__":
    main()

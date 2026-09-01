"""CLI entry point: format a merged screenplay JSON into a screenplay .md via Ornith.

This module is the only place that reads environment variables or resolves default
paths — everything under src/ is a plain library that takes what it needs as arguments.

Run from the repo root: python -m src.cli.write_screenplay <screenplay_json>
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.backends import build_screenplay_model
from src.persist import read_records
from src.screenplay import write_md, write_screenplay
from src.screenplay.write_screenplay import ORNITH_UNAVAILABLE_HINT

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCREENPLAYS_DIR = REPO_ROOT / "data" / "screenplays"
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "screenplay_template.md"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.cli.write_screenplay <screenplay_json>")

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        json_path = SCREENPLAYS_DIR / json_path
    if not json_path.exists():
        raise SystemExit(f"Screenplay JSON not found: {json_path}")

    events = read_records(json_path)
    template = DEFAULT_TEMPLATE_PATH.read_text()

    logger.info("Writing screenplay for: %s", json_path)
    screenplay_md = write_screenplay(
        events,
        template,
        model=build_screenplay_model(
            os.environ.get("SCREENPLAY_BACKEND", "ollama"),
            os.environ.get("ORNITH_MODEL", "ornith-1.5-255k"),
            unavailable_hint=ORNITH_UNAVAILABLE_HINT,
        ),
    )

    name = json_path.stem.removesuffix(".screenplay")
    output_path = write_md(screenplay_md, name, SCREENPLAYS_DIR)

    logger.info("Screenplay writing complete: %s", output_path)


if __name__ == "__main__":
    main()

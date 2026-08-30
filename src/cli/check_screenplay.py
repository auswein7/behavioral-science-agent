"""CLI entry point: verify a screenplay .md is faithful to its merged events JSON.

This module is the only place that resolves default paths - everything under src/
is a plain library that takes what it needs as arguments.

Run from the repo root:
python -m src.cli.check_screenplay <screenplay_md> <screenplay_json>

Exits nonzero when the screenplay drops, duplicates, or misattributes speech
events, invents action attributions, or leaks think text (TODO 3.6).
"""

import argparse
import logging
from pathlib import Path

from src.persist import read_records
from src.screenplay.faithfulness import check_faithfulness

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCREENPLAYS_DIR = REPO_ROOT / "data" / "screenplays"


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        path = SCREENPLAYS_DIR / path_str
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screenplay_md")
    parser.add_argument("screenplay_json")
    args = parser.parse_args()

    md_path = _resolve(args.screenplay_md)
    events_path = _resolve(args.screenplay_json)

    report = check_faithfulness(md_path.read_text(), read_records(events_path))
    for line in report.summary_lines():
        logger.info(line)

    if report.ok:
        logger.info("Faithfulness check passed: %s", md_path.name)
    else:
        raise SystemExit(f"Faithfulness check FAILED for {md_path.name} - see findings above")


if __name__ == "__main__":
    main()

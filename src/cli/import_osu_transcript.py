"""CLI entry point: import an OSU coded-transcript workbook for local coding.

Run from the repo root:
python -m src.cli.import_osu_transcript <workbook.xlsx> --session-id <id>

Writes <id>.utterances.json (the 4.1 envelope the coder reads) under
data/deliverables/, and under data/imports/ (git-ignored, local only) the
generated <id>.manifest.json with source_class osu_study, which is never
cleared for egress, and the <id>.speaker_map.json from OSU's demographic
labels to SPEAKER_NN.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.errors import PipelineError
from src.evaluation import import_osu_rows, read_table
from src.schemas import SCHEMA_VERSION

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rows, manifest, speaker_map = import_osu_rows(read_table(args.workbook), args.session_id)
    deliverables = args.data_dir / "deliverables"
    imports = args.data_dir / "imports"
    for directory in (deliverables, imports):
        directory.mkdir(parents=True, exist_ok=True)
    rows_path = deliverables / f"{args.session_id}.utterances.json"
    rows_path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "rows": [r.model_dump() for r in rows]}, indent=2)
    )
    (imports / f"{args.session_id}.manifest.json").write_text(manifest.model_dump_json(indent=2))
    (imports / f"{args.session_id}.speaker_map.json").write_text(json.dumps(speaker_map, indent=2))
    logger.info(
        "imported %d rows, %d speakers -> %s (manifest and speaker map in %s)",
        len(rows),
        len(speaker_map),
        rows_path,
        imports,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)

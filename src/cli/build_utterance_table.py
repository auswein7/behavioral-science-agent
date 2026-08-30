"""CLI entry point: build the per-utterance deliverable table (DATA_MODEL 4.1).

This module is the only place that resolves default paths - everything under src/
is a plain library that takes what it needs as arguments.

Run from the repo root:
python -m src.cli.build_utterance_table <captions_json> <tone_transcript_json> <manifest_json>

Pass a tone-enriched transcript (.formatted.tone.json), not the plain
.formatted.json - the delivered tone column comes from it. The manifest is a
SessionManifest JSON (DATA_MODEL 2.2); its session_id names the outputs.
"""

import argparse
import json
import logging
from pathlib import Path

from src.deliverables.utterance_table import build_utterance_rows, write_csv, write_json
from src.schemas import CaptionRecord, SessionManifest, ToneRecord
from src.screenplay import scrub_check

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTIONS_DIR = REPO_ROOT / "data" / "captions"
TRANSCRIPTS_DIR = REPO_ROOT / "data" / "transcripts"


def _resolve(path_str: str, default_dir: Path) -> Path:
    path = Path(path_str)
    if not path.exists():
        path = default_dir / path_str
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captions_json")
    parser.add_argument("tone_transcript_json")
    parser.add_argument("manifest_json")
    args = parser.parse_args()

    captions_path = _resolve(args.captions_json, CAPTIONS_DIR)
    transcript_path = _resolve(args.tone_transcript_json, TRANSCRIPTS_DIR)
    manifest_path = Path(args.manifest_json)

    manifest = SessionManifest.model_validate_json(manifest_path.read_text())
    captions = [CaptionRecord.model_validate(r) for r in json.loads(captions_path.read_text())]
    transcript = [ToneRecord.model_validate(r) for r in json.loads(transcript_path.read_text())]

    logger.info("Building utterance table for session %s (%d utterances, %d captions)",
                manifest.session_id, len(transcript), len(captions))
    rows = build_utterance_rows(transcript, captions, manifest)

    csv_path = write_csv(rows, manifest.session_id)
    json_path = write_json(rows, manifest.session_id)

    # Warn-only spot check over the delivered text, same backstop the pipeline
    # runs on captions and the screenplay; the enforcing gate is a separate stage.
    scrub_check("\n".join(f"{r.utterance}\n{r.nonverbal_notes}" for r in rows), "utterance_table")

    logger.info("Utterance table complete: %s and %s", csv_path, json_path)


if __name__ == "__main__":
    main()

"""CLI entry point: build the per-utterance deliverable table (DATA_MODEL 4.1).

This module is the only place that resolves default paths - everything under src/
is a plain library that takes what it needs as arguments.

Run from the repo root:
python -m src.cli.build_utterance_table <captions_json> <tone_transcript_json> <manifest_json>

Pass a tone-enriched transcript (.formatted.tone.json), not the plain
.formatted.json - the delivered tone column comes from it. The manifest is a
SessionManifest JSON (DATA_MODEL 2.2); its session_id names the outputs.

The transcript text goes through the deterministic spoken-name scrub before
the table is built, and the finished CSV must pass the enforcing gate: on
findings the table and its reports are still written under data/ for review,
but the command fails with GateBlockedError so a blocked artifact cannot be
mistaken for a deliverable. --cleared-by <role> records a human reviewer's
clearance and turns the same findings into a cleared_by_review pass.
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.deliverables.utterance_table import (
    DEFAULT_OUTPUT_DIR,
    build_utterance_rows,
    write_csv,
    write_json,
)
from src.persist import read_records
from src.schemas import CaptionRecord, SessionManifest, ToneRecord
from src.scrub import (
    evaluate_gate,
    require_pass,
    scrub_names,
    write_findings,
    write_report,
    write_scrubbed_json,
)

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
    parser.add_argument("--cleared-by", default=None, metavar="ROLE",
                        help="reviewer role clearing existing gate findings (records cleared_by_review)")
    parser.add_argument("--run-id", default=None, help="run id for the gate reports (default: local-<utc stamp>)")
    args = parser.parse_args()

    captions_path = _resolve(args.captions_json, CAPTIONS_DIR)
    transcript_path = _resolve(args.tone_transcript_json, TRANSCRIPTS_DIR)
    manifest_path = Path(args.manifest_json)
    run_id = args.run_id or "local-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    manifest = SessionManifest.model_validate_json(manifest_path.read_text())
    captions = [CaptionRecord.model_validate(r) for r in read_records(captions_path)]
    raw_transcript = read_records(transcript_path)

    scrubbed, replacements = scrub_names(raw_transcript)
    write_scrubbed_json(scrubbed, replacements, manifest.session_id)
    transcript = [ToneRecord.model_validate(r) for r in scrubbed]

    logger.info("Building utterance table for session %s (%d utterances, %d captions, %d names scrubbed)",
                manifest.session_id, len(transcript), len(captions), len(replacements))
    rows = build_utterance_rows(transcript, captions, manifest)

    csv_path = write_csv(rows, manifest.session_id)
    write_json(rows, manifest.session_id)

    gate_result = evaluate_gate(
        csv_path.read_text(encoding="utf-8"),
        artifact=csv_path.name,
        session_id=manifest.session_id,
        run_id=run_id,
        cleared_by=args.cleared_by,
    )
    write_findings(gate_result)
    write_report(gate_result, DEFAULT_OUTPUT_DIR)

    logger.info("Utterance table written: %s (gate: %s)", csv_path, gate_result.report.resolution)
    require_pass(gate_result)


if __name__ == "__main__":
    main()

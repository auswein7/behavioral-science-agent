"""Write the coded utterance table (DATA_MODEL 4.5) in OSU's flattened shape."""

import csv
import json
import logging
from pathlib import Path

from src.errors import DeliverableError
from src.schemas import SCHEMA_VERSION, CodedUtteranceRow, UtteranceRow

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "deliverables"


def flatten_row(row: CodedUtteranceRow, code_names: tuple[str, ...]) -> dict[str, object]:
    """One CSV record: the 4.1 columns, then <CODE>_<coder_id> and
    <CODE>_<coder_id>_rationale per code in codebook order, then the coder
    provenance columns."""
    record: dict[str, object] = {name: getattr(row, name) for name in UtteranceRow.model_fields}
    for code in code_names:
        record[f"{code}_{row.coder_id}"] = row.codes[code]
        record[f"{code}_{row.coder_id}_rationale"] = row.rationales.get(code, "")
    record["coder_model"] = row.coder_model
    record["coder_prompt_version"] = row.coder_prompt_version
    record["codebook_version"] = row.codebook_version
    return record


def write_coded_csv(
    rows: list[CodedUtteranceRow],
    session_id: str,
    code_names: tuple[str, ...],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """UTF-8, every field quoted, like the 4.1 table it extends."""
    if not rows:
        raise DeliverableError("refusing to write an empty coded table")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{session_id}.coded.csv"
    records = [flatten_row(row, code_names) for row in rows]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(records)
    logger.info("Wrote %d coded rows to %s", len(rows), output_path)
    return output_path


def write_coded_json(
    rows: list[CodedUtteranceRow],
    session_id: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """The JSON form, rewritten after every coded row so a long run never
    holds its only copy in memory (design principle 6). An empty list is
    allowed here, unlike the CSV: it is the resumable starting state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{session_id}.coded.json"
    envelope = {"schema_version": SCHEMA_VERSION, "rows": [r.model_dump() for r in rows]}
    output_path.write_text(json.dumps(envelope, indent=2))
    return output_path


def read_coded_json(path: Path) -> list[CodedUtteranceRow]:
    """Load a previous partial or complete coded table for resumption."""
    try:
        data = json.loads(path.read_text())
        return [CodedUtteranceRow.model_validate(r) for r in data["rows"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DeliverableError(f"cannot read coded rows from {path}: {exc}") from exc

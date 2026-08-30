"""Envelope persistence for stage record files (schema_version on every artifact).

Stage outputs are written as {"schema_version": ..., "records": [...]} so any
file can name the spec it was written under (DATA_MODEL conventions).
read_records accepts the pre-envelope bare-list form too, so artifacts from
runs before schema 0.1 (run 1 included) keep loading.
"""

import json
import logging
from pathlib import Path

from src.errors import ArtifactError
from src.schemas import SCHEMA_VERSION

logger = logging.getLogger(__name__)


def write_records(records: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"schema_version": SCHEMA_VERSION, "records": records}
    output_path.write_text(json.dumps(envelope, indent=2))
    return output_path


def read_records(path: Path) -> list[dict]:
    """Load a stage record file: enveloped (schema 0.1+) or legacy bare list."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read stage records from {path}: {exc}") from exc
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    raise ArtifactError(f"{path} is neither a record envelope nor a legacy record list")

"""Read codings from any of the three shapes they arrive in.

A coding is uid -> {code: 0, 1 or "NA"}. It comes from our own coded table
(DATA_MODEL 4.5, the .coded.json envelope or the flattened .coded.csv) or
from OSU's coded-transcript workbook (.xlsx), whose code columns follow the
same <CODE>_<coder_id> naming (DATA_MODEL 6). The coder id picks which
coder's columns to read, so several coders can sit in one file.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from src.errors import DeliverableError
from src.schemas import CodedUtteranceRow

NA = "NA"

Cell = int | str


@dataclass(frozen=True)
class LoadedCodes:
    """One coder's codes for one session, keyed by uid."""

    document: str
    codes: dict[str, dict[str, Cell]]


def normalize_cell(value: object) -> Cell:
    """Map a spreadsheet or CSV cell to 0, 1 or "NA".

    An empty cell reads as NA: OSU's sample leaves uncoded rows NA and
    nothing else blank. Anything else is refused rather than guessed."""
    if value is None:
        return NA
    if isinstance(value, bool):
        raise DeliverableError(f"code cell {value!r} is a boolean, expected 0, 1 or NA")
    if isinstance(value, int | float) and value in (0, 1):
        return int(value)
    text = str(value).strip()
    if text in ("0", "1"):
        return int(text)
    if text.upper() == NA or text == "":
        return NA
    raise DeliverableError(f"code cell {value!r} is not 0, 1 or NA")


def read_table(path: Path) -> list[dict[str, object]]:
    """Rows of a .xlsx (first sheet) or .csv file as header-keyed dicts."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".xlsx":
            import openpyxl

            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            rows = list(workbook.worksheets[0].iter_rows(values_only=True))
            workbook.close()
            if not rows:
                return []
            header = [str(h) if h is not None else "" for h in rows[0]]
            return [dict(zip(header, row, strict=False)) for row in rows[1:]]
        if suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
    except OSError as exc:
        raise DeliverableError(f"cannot read {path}: {exc}") from exc
    raise DeliverableError(f"{path}: expected a .xlsx, .csv or .coded.json file")


def _from_coded_json(path: Path, coder_id: str, code_names: tuple[str, ...]) -> LoadedCodes:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [CodedUtteranceRow.model_validate(r) for r in data["rows"]]
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        raise DeliverableError(f"cannot read coded rows from {path}: {exc}") from exc
    codes: dict[str, dict[str, Cell]] = {}
    documents = set()
    for row in rows:
        if row.coder_id != coder_id:
            raise DeliverableError(
                f"{path}: row {row.uid} was coded by {row.coder_id!r}, not {coder_id!r}"
            )
        missing = [c for c in code_names if c not in row.codes]
        if missing:
            raise DeliverableError(f"{path}: row {row.uid} has no code {missing}")
        if row.uid in codes:
            raise DeliverableError(f"{path}: uid {row.uid} appears twice")
        codes[row.uid] = {c: normalize_cell(row.codes[c]) for c in code_names}
        documents.add(row.document)
    return LoadedCodes(document=_single_document(path, documents), codes=codes)


def _from_table(path: Path, coder_id: str, code_names: tuple[str, ...]) -> LoadedCodes:
    rows = read_table(path)
    if not rows:
        raise DeliverableError(f"{path} holds no rows")
    columns = {f"{code}_{coder_id}": code for code in code_names}
    absent = [col for col in columns if col not in rows[0]]
    if "uid" not in rows[0]:
        absent.insert(0, "uid")
    if absent:
        raise DeliverableError(f"{path} has no column(s) {absent}")
    codes: dict[str, dict[str, Cell]] = {}
    documents = set()
    for row in rows:
        uid = str(row["uid"]).strip()
        if not uid or uid == "None":
            continue
        if uid in codes:
            raise DeliverableError(f"{path}: uid {uid} appears twice")
        codes[uid] = {code: normalize_cell(row[col]) for col, code in columns.items()}
        documents.add(str(row.get("document", "")))
    return LoadedCodes(document=_single_document(path, documents), codes=codes)


def _single_document(path: Path, documents: set[str]) -> str:
    # uids are unique within a document only (DATA_MODEL 4.1), so a file
    # mixing documents cannot be joined on uid.
    if len(documents) != 1:
        raise DeliverableError(f"{path} mixes {len(documents)} documents; split it per session")
    return documents.pop()


def load_codes(path: Path, coder_id: str, code_names: tuple[str, ...]) -> LoadedCodes:
    """One coder's codes from a coded table or a coded-transcript workbook."""
    if path.name.endswith(".json"):
        return _from_coded_json(path, coder_id, code_names)
    return _from_table(path, coder_id, code_names)

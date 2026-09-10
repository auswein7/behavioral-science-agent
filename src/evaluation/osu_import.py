"""Turn an OSU coded-transcript workbook into our utterance rows.

This is what lets the coder run on OSU's own sessions: the text-only
condition of their paper, reproduced on their transcripts with our local
model, then scored against their Gemini codes (or human codes) by the
harness. Their columns map onto DATA_MODEL 4.1 as section 6 describes;
the channels a transcript does not have are filled neutrally (tone
"unknown", no nonverbal notes, confidence not flagged).

OSU labels speakers demographically ("White24"), which our rules forbid in
any artifact, so each label becomes SPEAKER_NN in order of first appearance
and a "Confederate" label becomes the confederate role. The label-to-speaker
map is returned separately; the caller keeps it local and never delivers it.
The imported text is OSU's as sent and has not been through our name scrub,
so these rows are for local coding only.
"""

from collections.abc import Sequence

from src.errors import DeliverableError
from src.schemas import SCHEMA_VERSION, SessionManifest, UtteranceRow

NA = "NA"
REQUIRED_COLUMNS = ("uid", ".ord", "speaker", "utterance")


def _text(value: object) -> str:
    return NA if value is None or str(value).strip() == "" else str(value)


def import_osu_rows(
    records: Sequence[dict[str, object]], session_id: str
) -> tuple[list[UtteranceRow], SessionManifest, dict[str, str]]:
    """Rows, manifest and speaker map for one OSU session.

    records are the workbook's header-keyed rows (reference.read_table).
    Raises DeliverableError when a required column is missing, a row has no
    utterance, or the workbook mixes documents."""
    if not records:
        raise DeliverableError("the workbook holds no rows")
    absent = [c for c in REQUIRED_COLUMNS if c not in records[0]]
    if absent:
        raise DeliverableError(f"the workbook has no column(s) {absent}")
    documents = {str(r.get("document")) for r in records if r.get("uid") is not None}
    if len(documents) > 1:
        raise DeliverableError(f"the workbook mixes {len(documents)} documents")

    speaker_map: dict[str, str] = {}

    def speaker_of(label: object) -> str:
        key = str(label).strip()
        if key not in speaker_map:
            speaker_map[key] = f"SPEAKER_{len(speaker_map):02d}"
        return speaker_map[key]

    rows: list[UtteranceRow] = []
    for record in records:
        if record.get("uid") is None:
            continue
        if record.get("utterance") is None:
            raise DeliverableError(f"row {record['uid']} has no utterance")
        prior_label = record.get("prior_speaker")
        rows.append(
            UtteranceRow(
                document=session_id,
                uid=str(record["uid"]).strip(),
                ord=int(float(str(record[".ord"]))),
                speaker=speaker_of(record["speaker"]),
                utterance=str(record["utterance"]),
                time=_text(record.get("time")),
                end_time=NA,
                filename=f"{session_id}.utterances.csv",
                prior_utterance=_text(record.get("prior_utterance")),
                prior_speaker=(
                    NA if _text(prior_label) == NA else speaker_of(prior_label)
                ),
                role=(
                    "confederate"
                    if str(record["speaker"]).strip().lower() == "confederate"
                    else "participant"
                ),
                tone="unknown",
                low_confidence=False,
                nonverbal_notes="",
            )
        )
    roles = {row.speaker: row.role for row in rows}
    manifest = SessionManifest(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        source_class="osu_study",
        task_name="osu_imported_transcript",
        group_size=len(roles),
        expected_speaker_count=len(roles),
        speaker_roles=roles,
        notes=(
            "Imported from an OSU coded-transcript workbook: text only (no tone "
            "or nonverbal channels), OSU's text as sent, local coding only."
        ),
    )
    return rows, manifest, speaker_map

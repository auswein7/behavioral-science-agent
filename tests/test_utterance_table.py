"""Unit tests for the utterance-table builder against a synthetic fixture."""

import csv

import pytest

from src.deliverables.utterance_table import build_utterance_rows, write_csv
from src.errors import DeliverableError
from src.schemas import CaptionRecord, SessionManifest, ToneRecord


def tone_record(index: int, start: float, end: float, speaker: str, text: str, **overrides) -> ToneRecord:
    base = {
        "index": index,
        "start": start,
        "end": end,
        "start_hms": f"00:00:{start:06.3f}",
        "end_hms": f"00:00:{end:06.3f}",
        "speaker": speaker,
        "text": text,
        "word_count": len(text.split()),
        "avg_word_score": 0.9,
        "low_confidence_words": [],
        "emotion": "neutral",
        "emotion_scores": {"neutral": 0.9},
    }
    base.update(overrides)
    return ToneRecord.model_validate(base)


MANIFEST = SessionManifest(
    session_id="s001",
    source_class="public_domain",
    task_name="mock_jury",
    group_size=3,
    expected_speaker_count=3,
    speaker_roles={"SPEAKER_02": "confederate"},
)

TRANSCRIPT = [
    tone_record(0, 1.0, 4.0, "SPEAKER_00", "I think we should vote.", emotion="<unk>"),
    tone_record(1, 5.0, 8.0, "SPEAKER_01", "Agreed, let us begin.", avg_word_score=0.3),
    tone_record(2, 9.0, 11.0, "SPEAKER_02", "Please rank the items.", emotion="sad"),
]

CAPTIONS = [
    CaptionRecord.model_validate(
        {"index": 0, "timestamp": 1.5, "timestamp_hms": "00:00:01.500",
             "caption": "SPEAKER_00 leans forward.", "trigger": "speech_start", "speech_start": 1.0}
    ),
    CaptionRecord.model_validate(
        {"index": 1, "timestamp": 2.0, "timestamp_hms": "00:00:02.000",
             "caption": "A participant nods.", "trigger": "fixed_interval"}
    ),
    CaptionRecord.model_validate(
        {"index": 2, "timestamp": 4.5, "timestamp_hms": "00:00:04.500",
             "caption": "Everyone sits still.", "trigger": "fixed_interval"}
    ),
]


class TestBuildUtteranceRows:
    def test_empty_transcript_raises_typed_error(self):
        with pytest.raises(DeliverableError):
            build_utterance_rows([], CAPTIONS, MANIFEST)

    def test_osu_column_conventions(self):
        rows = build_utterance_rows(TRANSCRIPT, CAPTIONS, MANIFEST)
        assert [r.uid for r in rows] == ["u001", "u002", "u003"]
        assert [r.ord for r in rows] == [1, 2, 3]
        assert rows[0].prior_utterance == "NA"
        assert rows[0].prior_speaker == "NA"
        assert rows[1].prior_utterance == "I think we should vote."
        assert rows[1].prior_speaker == "SPEAKER_00"
        assert all(r.document == "s001" for r in rows)
        assert all(r.filename == "s001.utterances.csv" for r in rows)

    def test_roles_from_manifest_default_participant(self):
        rows = build_utterance_rows(TRANSCRIPT, CAPTIONS, MANIFEST)
        assert rows[0].role == "participant"
        assert rows[2].role == "confederate"

    def test_tone_normalized(self):
        rows = build_utterance_rows(TRANSCRIPT, CAPTIONS, MANIFEST)
        assert rows[0].tone == "unknown"
        assert rows[1].tone == "neutral"
        assert rows[2].tone == "sad"

    def test_low_confidence_flag(self):
        rows = build_utterance_rows(TRANSCRIPT, CAPTIONS, MANIFEST)
        assert rows[1].low_confidence is True
        assert rows[0].low_confidence is False

    def test_nonverbal_notes_sourcing_rule(self):
        rows = build_utterance_rows(TRANSCRIPT, CAPTIONS, MANIFEST)
        # Utterance 1 gets its own speech_start caption plus the fixed_interval
        # caption inside [1.0, 4.0); the 4.5s caption falls outside every span.
        assert rows[0].nonverbal_notes == "SPEAKER_00 leans forward. A participant nods."
        assert rows[1].nonverbal_notes == ""
        assert "Everyone sits still." not in " ".join(r.nonverbal_notes for r in rows)

    def test_rows_sorted_by_start_even_if_input_is_not(self):
        rows = build_utterance_rows(list(reversed(TRANSCRIPT)), CAPTIONS, MANIFEST)
        assert [r.speaker for r in rows] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]


class TestWriteCsv:
    def test_csv_round_trip_fully_quoted(self, tmp_path):
        rows = build_utterance_rows(TRANSCRIPT, CAPTIONS, MANIFEST)
        path = write_csv(rows, "s001", output_dir=tmp_path)
        raw = path.read_text(encoding="utf-8")
        assert '"time"' in raw.splitlines()[0]
        with path.open(encoding="utf-8", newline="") as handle:
            read_back = list(csv.DictReader(handle))
        assert len(read_back) == 3
        assert read_back[0]["uid"] == "u001"
        assert read_back[0]["time"] == "00:00:01.000"
        assert read_back[0]["prior_utterance"] == "NA"

    def test_empty_rows_raise_typed_error(self, tmp_path):
        with pytest.raises(DeliverableError):
            write_csv([], "s001", output_dir=tmp_path)

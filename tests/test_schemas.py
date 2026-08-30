"""Unit tests for the stage-boundary models in src/schemas.py."""

import pytest
from pydantic import ValidationError

from src.schemas import (
    CaptionRecord,
    SessionManifest,
    ToneRecord,
    UtteranceRow,
    normalize_tone,
)


def make_manifest(**overrides) -> SessionManifest:
    base = {
        "session_id": "s001",
        "source_class": "public_domain",
        "task_name": "mock_jury",
        "group_size": 8,
        "expected_speaker_count": 7,
    }
    base.update(overrides)
    return SessionManifest(**base)


class TestSessionManifest:
    def test_minimal_manifest_validates(self):
        manifest = make_manifest()
        assert manifest.schema_version == "0.1"
        assert manifest.speaker_roles is None

    def test_bad_source_class_rejected(self):
        with pytest.raises(ValidationError):
            make_manifest(source_class="youtube")

    def test_bad_role_rejected(self):
        with pytest.raises(ValidationError):
            make_manifest(speaker_roles={"SPEAKER_00": "moderator"})

    def test_undeclared_field_rejected(self):
        with pytest.raises(ValidationError):
            make_manifest(participant_names=["Alice"])


class TestNormalizeTone:
    def test_unk_maps_to_unknown(self):
        assert normalize_tone("<unk>") == "unknown"

    def test_none_maps_to_unknown(self):
        assert normalize_tone(None) == "unknown"

    def test_known_label_passes_through(self):
        assert normalize_tone("sad") == "sad"


class TestIntermediates:
    def test_tone_record_tolerates_extra_fields(self):
        record = ToneRecord.model_validate(
            {
                "index": 0,
                "start": 0.0,
                "end": 1.0,
                "start_hms": "00:00:00.000",
                "end_hms": "00:00:01.000",
                "speaker": "SPEAKER_00",
                "text": "hello",
                "word_count": 1,
                "avg_word_score": 0.9,
                "low_confidence_words": [],
                "emotion": "<unk>",
                "emotion_scores": {"sad": 0.5},
                "some_future_debug_field": True,
            }
        )
        assert record.emotion == "<unk>"

    def test_caption_record_defaults(self):
        record = CaptionRecord.model_validate(
            {"index": 0, "timestamp": 1.0, "timestamp_hms": "00:00:01.000", "caption": "A person nods."}
        )
        assert record.trigger == "fixed_interval"
        assert record.mentioned_speakers == []


class TestUtteranceRow:
    def make_row(self, **overrides) -> UtteranceRow:
        base = {
            "document": "s001",
            "uid": "u001",
            "ord": 1,
            "speaker": "SPEAKER_00",
            "utterance": "hello",
            "time": "00:00:00.000",
            "end_time": "00:00:01.000",
            "filename": "s001.utterances.csv",
            "prior_utterance": "NA",
            "prior_speaker": "NA",
            "role": "participant",
            "tone": "unknown",
            "low_confidence": False,
            "nonverbal_notes": "",
        }
        base.update(overrides)
        return UtteranceRow(**base)

    def test_row_is_frozen(self):
        row = self.make_row()
        with pytest.raises(ValidationError):
            row.speaker = "SPEAKER_01"

    def test_undeclared_column_rejected(self):
        with pytest.raises(ValidationError):
            self.make_row(shirt_color="blue")

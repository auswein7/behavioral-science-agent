"""Unit tests for format_segments and merge_screenplay (TODO 7.1)."""

from typing import ClassVar

from src.screenplay.merge_screenplay import merge_screenplay
from src.transcription.format_transcript import format_segments


class FakeScore:
    """Mimics a numpy scalar: carries .item(), is not a plain float."""

    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class TestFormatSegments:
    def test_basic_record_shape(self):
        records = format_segments(
            [{"start": 1.5, "end": 3.0, "speaker": "SPEAKER_00", "text": " hello world ",
              "words": [{"word": "hello", "score": 0.9}, {"word": "world", "score": 0.8}]}]
        )
        record = records[0]
        assert record["index"] == 0
        assert record["text"] == "hello world"
        assert record["start_hms"] == "00:00:01.500"
        assert record["avg_word_score"] == 0.85
        assert record["low_confidence_words"] == []

    def test_low_confidence_words_collected(self):
        records = format_segments(
            [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "um yes",
              "words": [{"word": "um", "score": 0.2}, {"word": "yes", "score": 0.9}]}]
        )
        assert records[0]["low_confidence_words"] == ["um"]

    def test_numpy_scalars_unwrapped(self):
        records = format_segments(
            [{"start": FakeScore(0.5), "end": FakeScore(1.0), "speaker": "SPEAKER_01",
              "text": "hi", "words": [{"word": "hi", "score": FakeScore(0.7)}]}]
        )
        assert records[0]["start"] == 0.5
        assert records[0]["avg_word_score"] == 0.7

    def test_missing_speaker_is_unknown(self):
        records = format_segments([{"start": 0.0, "end": 1.0, "text": "hi", "words": []}])
        assert records[0]["speaker"] == "UNKNOWN"
        assert records[0]["avg_word_score"] is None

    def test_hms_millisecond_rounding_carries(self):
        records = format_segments(
            [{"start": 1.9996, "end": 3661.9996, "speaker": "SPEAKER_00", "text": "hi", "words": []}]
        )
        assert records[0]["start_hms"] == "00:00:02.000"
        assert records[0]["end_hms"] == "01:01:02.000"


class TestMergeScreenplay:
    CAPTIONS: ClassVar[list[dict]] = [
        {"timestamp": 2.0, "timestamp_hms": "00:00:02.000", "caption": "A nod.",
         "trigger": "speech_start", "mentioned_speakers": ["SPEAKER_00"]},
        {"timestamp": 0.5, "timestamp_hms": "00:00:00.500", "caption": "Stillness."},
    ]
    TRANSCRIPT: ClassVar[list[dict]] = [
        {"start": 1.0, "end": 3.0, "start_hms": "00:00:01.000", "end_hms": "00:00:03.000",
         "speaker": "SPEAKER_00", "text": "Hello.", "emotion": "happy"},
    ]

    def test_chronological_order_and_reindex(self):
        events = merge_screenplay(self.CAPTIONS, self.TRANSCRIPT)
        assert [e["timestamp"] for e in events] == [0.5, 1.0, 2.0]
        assert [e["index"] for e in events] == [0, 1, 2]
        assert [e["type"] for e in events] == ["visual", "speech", "visual"]

    def test_trigger_defaults_to_fixed_interval(self):
        events = merge_screenplay(self.CAPTIONS, self.TRANSCRIPT)
        assert events[0]["trigger"] == "fixed_interval"
        assert events[2]["trigger"] == "speech_start"

    def test_speech_event_carries_emotion_and_span(self):
        events = merge_screenplay(self.CAPTIONS, self.TRANSCRIPT)
        speech = events[1]
        assert speech["emotion"] == "happy"
        assert speech["start"] == 1.0 and speech["end"] == 3.0

    def test_mentioned_speakers_passed_through(self):
        events = merge_screenplay(self.CAPTIONS, self.TRANSCRIPT)
        assert events[2]["mentioned_speakers"] == ["SPEAKER_00"]
        assert events[0]["mentioned_speakers"] == []

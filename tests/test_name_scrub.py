"""Unit tests for the deterministic spoken-name scrub. Needs the spaCy model;
self-skips when it is not installed (repo rule: skipif, never an ignore list)."""

import importlib.util

import pytest

# Importing the module is safe before the skip guard: spaCy loads lazily
# inside scrub_names, not at import time.
from src.scrub.name_scrub import scrub_names

spacy_missing = importlib.util.find_spec("spacy") is None
if not spacy_missing:
    import spacy

    try:
        spacy.load("en_core_web_sm")
        model_missing = False
    except OSError:
        model_missing = True
else:
    model_missing = True

pytestmark = pytest.mark.skipif(
    spacy_missing or model_missing, reason="spacy en_core_web_sm not installed"
)


def record(index: int, text: str) -> dict:
    return {
        "index": index,
        "start": float(index),
        "end": float(index) + 1.0,
        "start_hms": "00:00:00.000",
        "end_hms": "00:00:01.000",
        "speaker": "SPEAKER_00",
        "text": text,
        "word_count": len(text.split()),
        "avg_word_score": 0.9,
        "low_confidence_words": [],
        "emotion": "neutral",
        "emotion_scores": {"neutral": 0.9},
    }


class TestScrubNames:
    def test_person_name_replaced(self):
        scrubbed, replacements = scrub_names([record(0, "I spoke with Sarah Johnson about the case.")])
        assert "[NAME]" in scrubbed[0]["text"]
        assert "Sarah" not in scrubbed[0]["text"]
        assert replacements[0].entity == "Sarah Johnson"
        assert replacements[0].record_index == 0

    def test_clean_text_untouched(self):
        text = "I think we should take the money."
        scrubbed, replacements = scrub_names([record(0, text)])
        assert scrubbed[0]["text"] == text
        assert replacements == []

    def test_input_not_mutated(self):
        records = [record(0, "Ask Sarah Johnson about it.")]
        scrub_names(records)
        assert "Sarah Johnson" in records[0]["text"]

    def test_multiple_names_in_one_utterance(self):
        scrubbed, replacements = scrub_names(
            [record(0, "Sarah Johnson called, and later Mary Smith agreed with everything.")]
        )
        assert scrubbed[0]["text"].count("[NAME]") == 2
        assert len(replacements) == 2

    def test_only_text_field_changes(self):
        original = record(0, "Sarah Johnson agreed.")
        scrubbed, _ = scrub_names([original])
        for key in original:
            if key != "text":
                assert scrubbed[0][key] == original[key]

    def test_org_and_place_get_class_tokens(self):
        scrubbed, replacements = scrub_names(
            [record(0, "Most policies in Connecticut are cheap, even at Walmart.")]
        )
        assert "[PLACE]" in scrubbed[0]["text"]
        assert "Connecticut" not in scrubbed[0]["text"]
        labels = {r.label for r in replacements}
        assert "GPE" in labels

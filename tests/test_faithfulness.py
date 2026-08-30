"""Unit tests for the screenplay faithfulness checker."""

from src.screenplay.faithfulness import check_faithfulness

EVENTS = [
    {"type": "speech", "start_hms": "00:00:01.000", "speaker": "SPEAKER_00", "text": "Hello there."},
    {"type": "speech", "start_hms": "00:00:05.000", "speaker": "SPEAKER_01", "text": "I agree."},
    {"type": "visual", "timestamp_hms": "00:00:03.000", "caption": "SPEAKER_00 nods.",
     "mentioned_speakers": ["SPEAKER_00"]},
]

FAITHFUL_MD = """## SCENE 1

SPEAKER_00 nods at the group.

**SPEAKER_00** *(00:00:01.000)*
> Hello there.

**SPEAKER_01** *(00:00:05.000)*
> I agree.
"""


class TestFaithful:
    def test_faithful_screenplay_passes(self):
        report = check_faithfulness(FAITHFUL_MD, EVENTS)
        assert report.ok
        assert report.matched == 2
        assert report.total_speech_events == 2


class TestViolations:
    def test_missing_event_detected(self):
        md = FAITHFUL_MD.replace("**SPEAKER_01** *(00:00:05.000)*\n> I agree.\n", "")
        report = check_faithfulness(md, EVENTS)
        assert not report.ok
        assert len(report.missing) == 1
        assert "SPEAKER_01" in report.missing[0]

    def test_extra_dialogue_detected(self):
        md = FAITHFUL_MD + "\n**SPEAKER_00** *(00:00:09.000)*\n> I never said this.\n"
        report = check_faithfulness(md, EVENTS)
        assert report.extra_dialogue == ["00:00:09.000 SPEAKER_00"]

    def test_speaker_mismatch_detected(self):
        md = FAITHFUL_MD.replace("**SPEAKER_01** *(00:00:05.000)*", "**SPEAKER_03** *(00:00:05.000)*")
        report = check_faithfulness(md, EVENTS)
        assert report.speaker_mismatches
        # The timestamp still matches a known event, so it is a mismatch, not extra dialogue.
        assert report.extra_dialogue == []

    def test_invented_action_attribution_detected(self):
        md = FAITHFUL_MD.replace("SPEAKER_00 nods at the group.", "SPEAKER_05 nods at the group.")
        report = check_faithfulness(md, EVENTS)
        assert report.invented_action_tags == ["SPEAKER_05"]

    def test_grounded_action_attribution_allowed(self):
        report = check_faithfulness(FAITHFUL_MD, EVENTS)
        assert report.invented_action_tags == []

    def test_think_leak_detected(self):
        report = check_faithfulness("<think>planning...</think>\n" + FAITHFUL_MD, EVENTS)
        assert report.think_leak
        assert not report.ok

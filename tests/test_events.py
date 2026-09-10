"""Pipeline stage events: the stage context emits started/finished/failed,
subscribers derive timings and the run log from the bus, and dispatch is by
exact type. No model, no network."""

import logging

import pytest

pytest.importorskip("fairlib")

from fairlib.core.event_bus import AgentEventBus

from src.events import (
    GateDecided,
    PipelineEvents,
    RunLog,
    StageFailed,
    StageFinished,
    StageStarted,
    StageTimings,
)


def _recording_bus(*types):
    bus = AgentEventBus()
    seen = []
    for event_type in types:
        bus.subscribe(event_type, seen.append)
    return bus, seen


class TestStageContext:
    def test_a_stage_emits_started_then_finished_with_its_records(self):
        bus, seen = _recording_bus(StageStarted, StageFinished, StageFailed)
        with PipelineEvents(bus, "run1").stage("merge") as stage:
            stage.records = 12
        assert [type(e) for e in seen] == [StageStarted, StageFinished]
        finished = seen[1]
        assert finished.stage == "merge"
        assert finished.run_id == "run1"
        assert finished.records == 12
        assert finished.seconds >= 0

    def test_a_failing_stage_emits_failed_and_the_error_propagates(self):
        bus, seen = _recording_bus(StageStarted, StageFinished, StageFailed)
        with pytest.raises(KeyError), PipelineEvents(bus, "run1").stage("caption"):
            raise KeyError("boom")
        assert [type(e) for e in seen] == [StageStarted, StageFailed]
        assert seen[1].error_type == "KeyError"


class TestSubscribers:
    def test_timings_come_from_finished_events_only(self):
        bus = AgentEventBus()
        timings = StageTimings()
        timings.subscribe(bus)
        pipeline = PipelineEvents(bus, "run1")
        with pipeline.stage("tone"):
            pass
        pipeline.resumed("transcribe", 79, "x.formatted.json")
        pipeline.degraded("transcribe", "speaker count")
        assert set(timings.rounded()) == {"tone"}

    def test_the_run_log_is_a_subscriber(self, caplog):
        bus = AgentEventBus()
        RunLog().subscribe(bus)
        pipeline = PipelineEvents(bus, "run1")
        with caplog.at_level(logging.INFO, logger="pipeline"):
            with pipeline.stage("merge") as stage:
                stage.records = 3
            pipeline.resumed("caption", 379, "a.captions.json")
            pipeline.degraded("transcribe", "found 6 speakers, expected 7")
            pipeline.gate_decided("a.screenplay.md", "blocked", 4)
        text = caplog.text
        assert "[merge] started" in text
        assert "(3 records)" in text
        assert "resumed: loaded 379 records" in text
        assert "DEGRADED: found 6 speakers" in text
        assert "a.screenplay.md: blocked (4 findings)" in text
        blocked = [r for r in caplog.records if "[gate]" in r.getMessage()]
        assert blocked[0].levelno == logging.WARNING

    def test_dispatch_is_by_exact_type(self):
        # fairlib's bus does not deliver a leaf event to a base-class
        # subscriber; every consumer names the events it wants.
        from fairlib.core.events import FairlibEvent

        bus, seen = _recording_bus(FairlibEvent)
        PipelineEvents(bus, "run1").gate_decided("a", "clean", 0)
        assert seen == []
        bus.subscribe(GateDecided, seen.append)
        PipelineEvents(bus, "run1").gate_decided("a", "clean", 0)
        assert len(seen) == 1

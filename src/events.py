"""Typed pipeline stage events on fairlib's AgentEventBus.

Design principle 4 (observability by design; fairlib adoption map item c):
every stage announces itself as typed events - started, finished, resumed,
degraded, failed - and the gate announces its decision. Consumers subscribe;
nothing infers a stage's state from log lines or file existence. The run log
and the provenance timings are the first two subscribers.

fairlib dispatches by exact type, so each event is a leaf class and a
subscriber names the events it wants. The bus logs and swallows a
subscriber's exception, so subscribers here stay trivial.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from fairlib.core.events import FairlibEvent

logger = logging.getLogger("pipeline")


@dataclass(frozen=True)
class StageStarted(FairlibEvent):
    run_id: str
    stage: str


@dataclass(frozen=True)
class StageFinished(FairlibEvent):
    run_id: str
    stage: str
    seconds: float
    records: int | None


@dataclass(frozen=True)
class StageResumed(FairlibEvent):
    """A stage skipped by --from: its persisted output was loaded instead."""

    run_id: str
    stage: str
    records: int
    source: str


@dataclass(frozen=True)
class StageDegraded(FairlibEvent):
    """A stage finished but its result needs a human look before delivery."""

    run_id: str
    stage: str
    reason: str


@dataclass(frozen=True)
class StageFailed(FairlibEvent):
    run_id: str
    stage: str
    error_type: str


@dataclass(frozen=True)
class GateDecided(FairlibEvent):
    """The scrub gate's decision on one artifact (counts only, no text)."""

    run_id: str
    artifact: str
    resolution: str
    findings: int


class StageHandle:
    """What a running stage reports back before it finishes."""

    def __init__(self) -> None:
        self.records: int | None = None


class PipelineEvents:
    """Emits one run's stage events on a bus."""

    def __init__(self, bus: object, run_id: str) -> None:
        self._bus = bus
        self._run_id = run_id

    def _emit(self, event: FairlibEvent) -> None:
        self._bus.emit(event)  # type: ignore[attr-defined]

    @contextmanager
    def stage(self, name: str) -> Iterator[StageHandle]:
        """Started on entry, then finished with its duration and record count,
        or failed with the error type; the error itself propagates."""
        handle = StageHandle()
        self._emit(StageStarted(run_id=self._run_id, stage=name))
        start = time.monotonic()
        try:
            yield handle
        except Exception as exc:
            self._emit(StageFailed(run_id=self._run_id, stage=name, error_type=type(exc).__name__))
            raise
        self._emit(
            StageFinished(
                run_id=self._run_id,
                stage=name,
                seconds=time.monotonic() - start,
                records=handle.records,
            )
        )

    def resumed(self, stage: str, records: int, source: str) -> None:
        self._emit(StageResumed(run_id=self._run_id, stage=stage, records=records, source=source))

    def degraded(self, stage: str, reason: str) -> None:
        self._emit(StageDegraded(run_id=self._run_id, stage=stage, reason=reason))

    def gate_decided(self, artifact: str, resolution: str, findings: int) -> None:
        self._emit(
            GateDecided(
                run_id=self._run_id, artifact=artifact, resolution=resolution, findings=findings
            )
        )


class StageTimings:
    """The provenance's stage_timings, from StageFinished events."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def subscribe(self, bus: object) -> None:
        bus.subscribe(StageFinished, self._on_finished)  # type: ignore[attr-defined]

    def _on_finished(self, event: StageFinished) -> None:
        self.seconds[event.stage] = event.seconds

    def rounded(self) -> dict[str, float]:
        return {stage: round(seconds, 2) for stage, seconds in self.seconds.items()}


class RunLog:
    """The run log as a subscriber: one line per event."""

    def subscribe(self, bus: object) -> None:
        subscribe = bus.subscribe  # type: ignore[attr-defined]
        subscribe(StageStarted, lambda e: logger.info("[%s] started", e.stage))
        subscribe(
            StageFinished,
            lambda e: logger.info(
                "[%s] finished in %.1f s%s",
                e.stage,
                e.seconds,
                "" if e.records is None else f" ({e.records} records)",
            ),
        )
        subscribe(
            StageResumed,
            lambda e: logger.info("[%s] resumed: loaded %d records from %s", e.stage, e.records, e.source),
        )
        subscribe(StageDegraded, lambda e: logger.warning("[%s] DEGRADED: %s", e.stage, e.reason))
        subscribe(StageFailed, lambda e: logger.error("[%s] failed: %s", e.stage, e.error_type))
        subscribe(GateDecided, self._on_gate)

    @staticmethod
    def _on_gate(event: GateDecided) -> None:
        level = logging.WARNING if event.resolution == "blocked" else logging.INFO
        logger.log(
            level,
            "[gate] %s: %s (%d findings)",
            event.artifact,
            event.resolution,
            event.findings,
        )

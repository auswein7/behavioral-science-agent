"""Typed error hierarchy for the pipeline. Stages raise these, never bare built-ins."""


class PipelineError(Exception):
    """Base class for every error this project raises as a signal."""


class DeliverableError(PipelineError):
    """A deliverable could not be built from the given stage outputs."""


class ManifestError(PipelineError):
    """A session manifest is missing, unreadable, or inconsistent with the data."""


class ConfigurationError(PipelineError):
    """A required model, token, or setting is missing or invalid at startup."""


class GateBlockedError(PipelineError):
    """A deliverable failed the scrub gate and must not leave the box."""


class ArtifactError(PipelineError):
    """A persisted stage artifact is missing, unreadable, or malformed."""


class AdapterError(PipelineError):
    """A model-backend call failed after exhausting its retries."""


class CoderError(PipelineError):
    """The coder stage could not produce a valid coding for one utterance:
    the model exhausted its validated retries or its step budget. Carries
    the uid so a partial run says exactly where it stopped."""

    def __init__(self, message: str, *, uid: str, attempts: int, last_feedback: str = "") -> None:
        super().__init__(message)
        self.uid = uid
        self.attempts = attempts
        self.last_feedback = last_feedback


class CoderBudgetError(CoderError):
    """A coding failed after a model call behind it stopped on the output
    budget (done_reason LENGTH). A subclass so a caller tells a budget
    failure from a prompt or codebook one by isinstance. max_tokens is the
    budget in force, None when the backend default applied."""

    def __init__(self, message: str, *, uid: str, attempts: int, max_tokens: int | None) -> None:
        super().__init__(message, uid=uid, attempts=attempts)
        self.max_tokens = max_tokens

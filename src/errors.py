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

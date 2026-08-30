"""Typed error hierarchy for the pipeline. Stages raise these, never bare built-ins."""


class PipelineError(Exception):
    """Base class for every error this project raises as a signal."""


class DeliverableError(PipelineError):
    """A deliverable could not be built from the given stage outputs."""


class ManifestError(PipelineError):
    """A session manifest is missing, unreadable, or inconsistent with the data."""

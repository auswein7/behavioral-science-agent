"""The codebook: the pluggable definition of what the coder stage codes.

DATA_MODEL 4.5 makes the code set a document, not a schema: the set can
change without a schema-version bump to the anonymizer deliverables. This
module is that document's typed form. The real definitions are OSU's
(McLeer et al., pending from Dave); until they arrive the placeholder below
carries only the code names and the one-line meanings from
PROJECT_FRAMING so the scaffold, its tests and its prompt assembly are
exercised end to end. A run that used the placeholder says so in its
provenance through the version string, never silently.
"""

import hashlib
import json

from pydantic import Field

from src.schemas import _Deliverable

PLACEHOLDER_VERSION = "placeholder-2026-09-02"


class CodeDefinition(_Deliverable):
    """One code: its column name, its long name and the definition the
    model is given verbatim."""

    code: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9]*$")
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class Codebook(_Deliverable):
    """The ordered code set plus the version recorded in provenance."""

    version: str = Field(min_length=1)
    codes: tuple[CodeDefinition, ...] = Field(min_length=1)

    @property
    def code_names(self) -> tuple[str, ...]:
        return tuple(c.code for c in self.codes)

    @property
    def is_placeholder(self) -> bool:
        return self.version.startswith("placeholder-")

    def digest(self) -> str:
        """Content digest, so provenance pins the exact definitions used."""
        payload = json.dumps(self.model_dump(), sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()[:16]


def placeholder_codebook() -> Codebook:
    """The scaffold codebook: OSU's six code names with the one-line meanings
    from PROJECT_FRAMING. Not the real definitions; see the module docstring."""
    return Codebook(
        version=PLACEHOLDER_VERSION,
        codes=(
            CodeDefinition(
                code="AO",
                name="Action Opportunity",
                definition="The speaker invites another person to act or speak, or gives them the floor.",
            ),
            CodeDefinition(
                code="PO",
                name="Performance Output",
                definition="The speaker contributes a task-relevant idea, proposal or piece of work.",
            ),
            CodeDefinition(
                code="PE",
                name="Positive Evaluation",
                definition="The speaker positively evaluates another person's contribution.",
            ),
            CodeDefinition(
                code="NE",
                name="Negative Evaluation",
                definition="The speaker negatively evaluates another person's contribution.",
            ),
            CodeDefinition(
                code="I",
                name="Influence",
                definition="The speaker changes position or accepts another person's view.",
            ),
            CodeDefinition(
                code="Apology",
                name="Apology",
                definition="The speaker apologizes.",
            ),
        ),
    )

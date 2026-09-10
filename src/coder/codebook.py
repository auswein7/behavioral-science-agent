"""The codebook: the pluggable definition of what the coder stage codes.

DATA_MODEL 4.5 makes the code set a document, not a schema: the set can
change without a schema-version bump to the anonymizer deliverables. This
module is that document's typed form. The real definitions are OSU's
(McLeer et al., pending from Dave); they arrive as a JSON file named by the
CODER_CODEBOOK lever (docs/codebook.example.json is the template). Until
then the placeholder below carries only the code names and the one-line
meanings from PROJECT_FRAMING so the scaffold, its tests and its prompt
assembly are exercised end to end. A run that used a placeholder says so in
its provenance through the version string, never silently.
"""

import hashlib
import json
from pathlib import Path

from pydantic import Field, ValidationError, model_validator

from src.errors import ConfigurationError
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

    @model_validator(mode="after")
    def _unique_codes(self) -> "Codebook":
        # A repeated code name would make two CSV columns collide and let the
        # validator accept a judgment set that is short one definition.
        names = [c.code for c in self.codes]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate code names {duplicates}")
        return self

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


def load_codebook(path: Path) -> Codebook:
    """Load a codebook document (JSON, the Codebook shape).

    Raises ConfigurationError naming the file on a missing or unreadable
    file, invalid JSON, or a document that does not validate: the codebook
    is a run lever, so a bad one stops the run before the first row."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read codebook {path}: {exc}") from exc
    except ValueError as exc:
        raise ConfigurationError(f"codebook {path} is not valid JSON: {exc}") from exc
    try:
        return Codebook.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"codebook {path} is invalid: {exc}") from exc


def resolve_codebook(path: str | None) -> Codebook:
    """The codebook a run uses: the file the CODER_CODEBOOK lever names, or
    the placeholder when the lever is unset."""
    return placeholder_codebook() if path is None else load_codebook(Path(path))


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

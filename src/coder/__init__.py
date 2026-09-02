"""The coder stage: de-identified utterance rows -> OSU behavior codes.

Layer 3 (orchestration) in the design principles table: the stage consumes
only Tier C deliverables (DATA_MODEL 4.1 rows) and produces Tier C coded
rows (4.5). The model behind it is a fairlib chat adapter handed in by the
backend factory; nothing here knows which provider answers.
"""

from src.coder.agent import (
    AbstractUtteranceCoder,
    FairlibAgentCoder,
    coding_validator,
    iter_coded_rows,
    render_coder_role,
)
from src.coder.codebook import Codebook, CodeDefinition, placeholder_codebook
from src.coder.deliverable import write_coded_csv, write_coded_json

__all__ = [
    "AbstractUtteranceCoder",
    "CodeDefinition",
    "Codebook",
    "FairlibAgentCoder",
    "coding_validator",
    "iter_coded_rows",
    "placeholder_codebook",
    "render_coder_role",
    "write_coded_csv",
    "write_coded_json",
]

"""The evaluation harness: per-code agreement between two codings.

Layer 4 (application) in the design principles table. It reads delivered
coded tables (DATA_MODEL 4.5) and reference codings (OSU's coded-transcript
workbooks, human codes when they arrive) and reports precision, recall, F1
and Cohen's kappa per code, never pooled accuracy. No model is called.
"""

from src.evaluation.metrics import (
    CodeAgreement,
    EvaluationReport,
    code_agreement,
    evaluate,
)
from src.evaluation.osu_import import import_osu_rows
from src.evaluation.reference import LoadedCodes, load_codes, normalize_cell, read_table
from src.evaluation.split import SPLIT_RULE, split_of

__all__ = [
    "SPLIT_RULE",
    "CodeAgreement",
    "EvaluationReport",
    "LoadedCodes",
    "code_agreement",
    "evaluate",
    "import_osu_rows",
    "load_codes",
    "normalize_cell",
    "read_table",
    "split_of",
]

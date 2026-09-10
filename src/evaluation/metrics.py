"""Per-code agreement between a candidate coding and a reference coding.

The codes are rare (NE 6%, I 9%, Apology 2% in OSU's sample), so a pooled
accuracy would be dominated by the easy zeros; every figure here is per
code. The reference is whatever the comparison treats as truth: human codes
when OSU sends them, OSU's Gemini codes for the baseline-reproduction check.

NA cells (roles OSU leaves uncoded) are excluded from a code's counts. A row
where exactly one side is NA is counted as one_side_na: it means the two
codings disagree about which rows get coded at all, which a precision figure
would otherwise hide. A ratio with a zero denominator is None, never 0.
"""

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import Field

from src.errors import DeliverableError
from src.evaluation.reference import NA, Cell
from src.evaluation.split import SPLIT_RULE, split_of
from src.schemas import SCHEMA_VERSION, _Deliverable


class CodeAgreement(_Deliverable):
    """Confusion counts and derived agreement for one code."""

    code: str
    n: int = Field(ge=0)
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    tn: int = Field(ge=0)
    one_side_na: int = Field(ge=0)
    precision: float | None
    recall: float | None
    f1: float | None
    agreement: float | None
    cohen_kappa: float | None


class EvaluationReport(_Deliverable):
    """One candidate-versus-reference comparison. Counts only, no text, so
    the report can travel wherever the codes themselves may."""

    schema_version: str = SCHEMA_VERSION
    candidate: str
    reference: str
    codebook_version: str
    split: Literal["all", "tune", "test"]
    split_rule: str
    rows_compared: int = Field(ge=0)
    missing_in_candidate: tuple[str, ...]
    missing_in_reference: tuple[str, ...]
    per_code: dict[str, CodeAgreement]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def code_agreement(code: str, pairs: Iterable[tuple[Cell, Cell]]) -> CodeAgreement:
    """Agreement for one code over (candidate, reference) cell pairs."""
    tp = fp = fn = tn = one_side_na = 0
    for candidate, reference in pairs:
        for cell in (candidate, reference):
            if cell not in (0, 1, NA):
                raise DeliverableError(f"code {code}: cell {cell!r} is not 0, 1 or NA")
        if candidate == NA and reference == NA:
            continue
        if candidate == NA or reference == NA:
            one_side_na += 1
        elif candidate == 1 and reference == 1:
            tp += 1
        elif candidate == 1:
            fp += 1
        elif reference == 1:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    kappa = None
    if n:
        observed = (tp + tn) / n
        expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        # Both codings constant and identical: chance agreement is total and
        # kappa is undefined, not perfect.
        kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return CodeAgreement(
        code=code,
        n=n,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        one_side_na=one_side_na,
        precision=precision,
        recall=recall,
        f1=f1,
        agreement=_ratio(tp + tn, n),
        cohen_kappa=kappa,
    )


def evaluate(
    candidate: Mapping[str, Mapping[str, Cell]],
    reference: Mapping[str, Mapping[str, Cell]],
    code_names: tuple[str, ...],
    *,
    candidate_label: str,
    reference_label: str,
    codebook_version: str,
    session: str,
    split: Literal["all", "tune", "test"] = "all",
) -> EvaluationReport:
    """Compare two codings of one session over the rows both carry.

    session keys the declared split (split_of); split="all" scores every
    row. Rows present on one side only are listed, not scored."""

    def in_split(uid: str) -> bool:
        return split == "all" or split_of(session, uid) == split

    shared = sorted(uid for uid in set(candidate) & set(reference) if in_split(uid))
    per_code = {
        code: code_agreement(code, ((candidate[u][code], reference[u][code]) for u in shared))
        for code in code_names
    }
    return EvaluationReport(
        candidate=candidate_label,
        reference=reference_label,
        codebook_version=codebook_version,
        split=split,
        split_rule=SPLIT_RULE,
        rows_compared=len(shared),
        missing_in_candidate=tuple(sorted(u for u in set(reference) - set(candidate) if in_split(u))),
        missing_in_reference=tuple(sorted(u for u in set(candidate) - set(reference) if in_split(u))),
        per_code=per_code,
    )

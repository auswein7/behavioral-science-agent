"""CLI entry point: score a coding against a reference, per code.

Run from the repo root:
python -m src.cli.evaluate_coding <candidate> <reference> \
    --candidate-id fairlib_local --reference-id gemini35flash [--split all|tune|test]

Either file may be a .coded.json, a .coded.csv or an OSU coded-transcript
.xlsx. Prints precision, recall, F1, agreement and Cohen's kappa per code and
writes the report (counts only) as JSON next to the candidate, or to
--output. The codebook comes from the CODER_CODEBOOK lever like the coder's.
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.coder.codebook import resolve_codebook
from src.config import load_coder_config
from src.errors import PipelineError
from src.evaluation import EvaluationReport, evaluate, load_codes

logger = logging.getLogger(__name__)


def _fmt(value: float | None) -> str:
    return "   -" if value is None else f"{value:.2f}"


def render(report: EvaluationReport) -> str:
    """The per-code table as plain text."""
    lines = [
        f"candidate {report.candidate} vs reference {report.reference}",
        f"split {report.split}, rows compared {report.rows_compared}, codebook {report.codebook_version}",
        f"{'code':<8}{'n':>5}{'ref+':>6}{'cand+':>6}{'prec':>7}{'rec':>7}{'f1':>7}{'agree':>7}{'kappa':>7}{'1-NA':>6}",
    ]
    for code, a in report.per_code.items():
        lines.append(
            f"{code:<8}{a.n:>5}{a.tp + a.fn:>6}{a.tp + a.fp:>6}{_fmt(a.precision):>7}"
            f"{_fmt(a.recall):>7}{_fmt(a.f1):>7}{_fmt(a.agreement):>7}{_fmt(a.cohen_kappa):>7}"
            f"{a.one_side_na:>6}"
        )
    if report.missing_in_candidate:
        lines.append(f"missing in candidate: {len(report.missing_in_candidate)} rows")
    if report.missing_in_reference:
        lines.append(f"missing in reference: {len(report.missing_in_reference)} rows")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--split", choices=("all", "tune", "test"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    codebook = resolve_codebook(load_coder_config().codebook_path)

    candidate = load_codes(args.candidate, args.candidate_id, codebook.code_names)
    reference = load_codes(args.reference, args.reference_id, codebook.code_names)
    report = evaluate(
        candidate.codes,
        reference.codes,
        codebook.code_names,
        candidate_label=f"{args.candidate.name}:{args.candidate_id}",
        reference_label=f"{args.reference.name}:{args.reference_id}",
        codebook_version=codebook.version,
        session=candidate.document,
        split=args.split,
    )
    print(render(report))
    output = args.output or args.candidate.with_name(
        f"{candidate.document}.eval_{args.reference_id}_{args.split}.json"
    )
    output.write_text(report.model_dump_json(indent=2))
    logger.info("wrote %s", output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)

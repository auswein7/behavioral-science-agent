"""The enforcing scrub gate (DATA_MODEL section 1 gate rule, 4.4 report shapes).

Where scrub_check warns and moves on, the gate decides. evaluate_gate scans a
deliverable's text and produces both forms of the record: the internal finding
list (Tier B - carries excerpts, stays under data/) and the delivered
ScrubReport (Tier C - counts and the decision only, because a report quoting a
leak would itself leak). require_pass turns a blocked result into a typed
error so no caller can ship a blocked artifact by ignoring a warning.
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.errors import ArtifactError, GateBlockedError
from src.schemas import GateFinding, ScrubReport
from src.scrub.patterns import iter_findings

logger = logging.getLogger(__name__)

DEFAULT_FINDINGS_DIR = Path(__file__).resolve().parents[2] / "data" / "gate"

CATEGORIES = ("gendered", "appearance", "likely_name")
EXCERPT_WINDOW = 40


@dataclass(frozen=True)
class GateResult:
    findings: list[GateFinding]
    report: ScrubReport

    @property
    def blocked(self) -> bool:
        return self.report.resolution == "blocked"


def _excerpt(text: str, position: int, term: str) -> str:
    start = max(0, position - EXCERPT_WINDOW)
    end = min(len(text), position + len(term) + EXCERPT_WINDOW)
    return " ".join(text[start:end].split())


def evaluate_gate(
    text: str,
    artifact: str,
    session_id: str,
    run_id: str,
    cleared_by: str | None = None,
) -> GateResult:
    """Scan `text` and decide. Resolution is "clean" with no findings; with
    findings it is "blocked" unless `cleared_by` names the human reviewer role
    that examined them (then "cleared_by_review"). Redaction is an upstream
    fix, not something the gate performs."""
    if not text.strip():
        # An empty artifact trivially has zero findings; without this check it
        # would be ruled "clean" (it happened: a 0-char screenplay from a
        # context-overflowed model call). Malformed input, not a gate decision.
        raise ArtifactError(
            f"{artifact} is empty ({len(text)} chars); refusing to gate a "
            f"blank artifact - the producing stage failed upstream"
        )
    findings = [
        GateFinding(category=category, term=term, position=position,
                    excerpt=_excerpt(text, position, term))
        for category, term, position in iter_findings(text)
    ]
    counts = dict.fromkeys(CATEGORIES, 0)
    for finding in findings:
        counts[finding.category] += 1

    if not findings:
        resolution = "clean"
    elif cleared_by:
        resolution = "cleared_by_review"
    else:
        resolution = "blocked"

    report = ScrubReport(
        session_id=session_id,
        run_id=run_id,
        artifact=artifact,
        findings_by_category=counts,
        resolution=resolution,
        reviewer=cleared_by,
        decided=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    logger.info("Gate decision for %s: %s (%s)", artifact, resolution,
                ", ".join(f"{k}={v}" for k, v in counts.items()))
    return GateResult(findings=findings, report=report)


def require_pass(result: GateResult) -> None:
    """Raise GateBlockedError for a blocked result. The message carries counts
    only - excerpts stay in the Tier B findings file."""
    if result.blocked:
        counts = result.report.findings_by_category
        raise GateBlockedError(
            f"{result.report.artifact} is blocked by the scrub gate "
            f"({', '.join(f'{k}={v}' for k, v in counts.items() if v)}); "
            f"review the findings file under data/gate/ and fix upstream or clear by review"
        )


def write_findings(result: GateResult, output_dir: Path = DEFAULT_FINDINGS_DIR) -> Path:
    """Persist the internal (Tier B) findings beside the decision for human review.
    Named by artifact, which carries the session prefix (one file per gated artifact)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.report.artifact}.findings.json"
    payload = {
        "schema_version": result.report.schema_version,
        "report": result.report.model_dump(),
        "findings": [f.model_dump() for f in result.findings],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %d gate findings to %s", len(result.findings), output_path)
    return output_path


def write_report(result: GateResult, output_dir: Path) -> Path:
    """Persist the delivered (Tier C) ScrubReport next to the artifact it covers.
    One report per gated artifact: <artifact>.scrub_report.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.report.artifact}.scrub_report.json"
    output_path.write_text(result.report.model_dump_json(indent=2))
    logger.info("Wrote scrub report to %s", output_path)
    return output_path

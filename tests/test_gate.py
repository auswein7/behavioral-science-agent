"""Unit tests for the enforcing scrub gate."""

import pytest

from src.errors import ArtifactError, GateBlockedError
from src.schemas import ScrubReport
from src.scrub.gate import evaluate_gate, require_pass, write_findings

CLEAN_TEXT = 'SPEAKER_00 leans forward and mentions [NAME] from [PLACE]. "NA" is fine too.'
LEAKY_TEXT = "The person in the blue shirt nods at Walmart while she speaks."


def gate(text: str, cleared_by: str | None = None):
    return evaluate_gate(text, artifact="t.csv", session_id="s001", run_id="r1", cleared_by=cleared_by)


class TestEvaluateGate:
    def test_clean_text_passes(self):
        result = gate(CLEAN_TEXT)
        assert result.report.resolution == "clean"
        assert not result.blocked
        assert result.findings == []
        require_pass(result)

    def test_empty_artifact_is_typed_error_not_clean(self):
        # A 0-char artifact trivially has zero findings; it must be refused as
        # malformed, never ruled "clean" (a context-overflowed model call once
        # produced exactly this).
        with pytest.raises(ArtifactError):
            gate("")
        with pytest.raises(ArtifactError):
            gate("   \n\t ")

    def test_leaky_text_blocks(self):
        result = gate(LEAKY_TEXT)
        assert result.blocked
        categories = {f.category for f in result.findings}
        assert categories == {"gendered", "appearance", "likely_name"}
        with pytest.raises(GateBlockedError):
            require_pass(result)

    def test_counts_match_findings(self):
        result = gate(LEAKY_TEXT)
        counts = result.report.findings_by_category
        assert sum(counts.values()) == len(result.findings)
        assert counts["appearance"] == 1  # "shirt"

    def test_clearance_turns_block_into_review_pass(self):
        result = gate(LEAKY_TEXT, cleared_by="fair_lab_reviewer")
        assert result.report.resolution == "cleared_by_review"
        assert result.report.reviewer == "fair_lab_reviewer"
        require_pass(result)

    def test_blocked_message_carries_no_excerpts(self):
        result = gate(LEAKY_TEXT)
        with pytest.raises(GateBlockedError) as excinfo:
            require_pass(result)
        assert "shirt" not in str(excinfo.value)
        assert "Walmart" not in str(excinfo.value)

    def test_report_is_valid_deliverable_model(self):
        result = gate(CLEAN_TEXT)
        assert isinstance(result.report, ScrubReport)
        assert result.report.schema_version == "0.1"


class TestWriteFindings:
    def test_findings_file_carries_excerpts(self, tmp_path):
        result = gate(LEAKY_TEXT)
        path = write_findings(result, output_dir=tmp_path)
        payload = path.read_text()
        assert "shirt" in payload
        assert '"resolution": "blocked"' in payload

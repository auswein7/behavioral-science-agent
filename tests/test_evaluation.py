"""The evaluation harness: per-code metrics, the declared split, the three
coding file shapes, and the OSU workbook import. Fixtures are synthetic;
no model, no network, no OSU data."""

import csv
import json
from pathlib import Path

import pytest

from src.errors import DeliverableError
from src.evaluation import (
    code_agreement,
    evaluate,
    import_osu_rows,
    load_codes,
    normalize_cell,
    read_table,
    split_of,
)
from src.evaluation.split import SPLIT_RULE
from src.schemas import SCHEMA_VERSION

openpyxl = pytest.importorskip("openpyxl")

CODES = ("AO", "NE")
OSU_HEADER = (
    "document", "uid", ".ord", "speaker", "utterance", "time", "filename",
    "prior_utterance", "prior_speaker", "AO_gemini35flash", "NE_gemini35flash",
)


def _osu_workbook(path: Path, rows: list[tuple]) -> Path:
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(OSU_HEADER)
    for row in rows:
        sheet.append(row)
    book.save(path)
    return path


OSU_ROWS = [
    ("Doc A", "u001", 1, "White24", "first line", 0.1, "f", "NA", "NA", "1", "0"),
    ("Doc A", "u002", 2, "Confederate", "second line", 0.2, "f", "first line", "White24", "NA", "NA"),
    ("Doc A", "u003", 3, "Latina23", "third line", 0.3, "f", "second line", "Confederate", 0, 1),
]


class TestCodeAgreement:
    def test_counts_and_ratios(self):
        pairs = [(1, 1), (1, 0), (0, 1), (0, 0), (0, 0)]
        a = code_agreement("AO", pairs)
        assert (a.tp, a.fp, a.fn, a.tn, a.n) == (1, 1, 1, 2, 5)
        assert a.precision == 0.5
        assert a.recall == 0.5
        assert a.f1 == 0.5
        assert a.agreement == 0.6
        # observed 0.6, expected (2*2 + 3*3)/25 = 0.52
        assert a.cohen_kappa == pytest.approx((0.6 - 0.52) / 0.48)

    def test_na_rows_are_excluded_and_one_sided_na_is_counted(self):
        a = code_agreement("AO", [("NA", "NA"), ("NA", 1), (0, "NA"), (1, 1)])
        assert a.n == 1
        assert a.one_side_na == 2

    def test_zero_denominators_are_none_not_zero(self):
        a = code_agreement("Apology", [(0, 0), (0, 0)])
        assert a.precision is None
        assert a.recall is None
        assert a.f1 is None
        # Both codings constant: chance agreement is total, kappa undefined.
        assert a.cohen_kappa is None
        assert a.agreement == 1.0

    def test_no_true_positive_is_an_f1_of_zero(self):
        a = code_agreement("NE", [(1, 0), (0, 1)])
        assert a.precision == 0.0
        assert a.recall == 0.0
        assert a.f1 == 0.0

    def test_perfect_agreement_on_a_varied_code_is_kappa_one(self):
        assert code_agreement("PO", [(1, 1), (0, 0)]).cohen_kappa == 1.0

    def test_a_bad_cell_is_refused(self):
        with pytest.raises(DeliverableError, match="not 0, 1 or NA"):
            code_agreement("AO", [(2, 1)])


class TestSplit:
    def test_split_is_deterministic_and_keyed_on_session_and_uid(self):
        assert split_of("s1", "u001") == split_of("s1", "u001")
        halves = {split_of("s1", f"u{i:03d}") for i in range(1, 200)}
        assert halves == {"tune", "test"}

    def test_about_a_fifth_is_held_out(self):
        test = sum(split_of("s1", f"u{i:04d}") == "test" for i in range(1, 2001))
        assert 300 < test < 500

    def test_the_rule_is_stated(self):
        assert "bsa-split-v1" in SPLIT_RULE


class TestEvaluate:
    def test_scores_shared_rows_and_lists_the_rest(self):
        cand = {"u001": {"AO": 1, "NE": 0}, "u002": {"AO": 0, "NE": 0}}
        ref = {"u001": {"AO": 1, "NE": 1}, "u003": {"AO": 0, "NE": 0}}
        report = evaluate(
            cand, ref, CODES, candidate_label="c", reference_label="r",
            codebook_version="v", session="s1",
        )
        assert report.rows_compared == 1
        assert report.missing_in_candidate == ("u003",)
        assert report.missing_in_reference == ("u002",)
        assert report.per_code["AO"].tp == 1
        assert report.per_code["NE"].fn == 1
        assert report.split_rule == SPLIT_RULE

    def test_split_filters_rows(self):
        uids = [f"u{i:03d}" for i in range(1, 101)]
        cand = {u: {"AO": 0, "NE": 0} for u in uids}
        test = evaluate(cand, cand, CODES, candidate_label="c", reference_label="r",
                        codebook_version="v", session="s1", split="test")
        tune = evaluate(cand, cand, CODES, candidate_label="c", reference_label="r",
                        codebook_version="v", session="s1", split="tune")
        assert test.rows_compared + tune.rows_compared == 100
        assert test.rows_compared == sum(split_of("s1", u) == "test" for u in uids)


class TestLoadCodes:
    def test_reads_an_osu_workbook(self, tmp_path):
        path = _osu_workbook(tmp_path / "osu.xlsx", OSU_ROWS)
        loaded = load_codes(path, "gemini35flash", CODES)
        assert loaded.document == "Doc A"
        assert loaded.codes["u001"] == {"AO": 1, "NE": 0}
        assert loaded.codes["u002"] == {"AO": "NA", "NE": "NA"}
        assert loaded.codes["u003"] == {"AO": 0, "NE": 1}

    def test_reads_a_coded_csv(self, tmp_path):
        path = tmp_path / "s1.coded.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["document", "uid", "AO_local", "NE_local"])
            writer.writerow(["s1", "u001", "1", "NA"])
        assert load_codes(path, "local", CODES).codes["u001"] == {"AO": 1, "NE": "NA"}

    def test_reads_a_coded_json_and_checks_the_coder(self, tmp_path):
        from test_coder import make_row

        base = make_row(1).model_dump()
        row = dict(base, codes={"AO": 1, "NE": 0}, rationales={}, coder_id="local",
                   coder_model="m", coder_prompt_version="p", codebook_version="v")
        path = tmp_path / "s1.coded.json"
        path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "rows": [row]}))
        assert load_codes(path, "local", CODES).codes["u001"] == {"AO": 1, "NE": 0}
        with pytest.raises(DeliverableError, match="coded by"):
            load_codes(path, "other", CODES)

    def test_missing_coder_columns_are_named(self, tmp_path):
        path = _osu_workbook(tmp_path / "osu.xlsx", OSU_ROWS)
        with pytest.raises(DeliverableError, match="AO_human"):
            load_codes(path, "human", CODES)

    def test_mixed_documents_are_refused(self, tmp_path):
        rows = [OSU_ROWS[0], ("Doc B", *OSU_ROWS[2][1:])]
        path = _osu_workbook(tmp_path / "osu.xlsx", rows)
        with pytest.raises(DeliverableError, match="mixes"):
            load_codes(path, "gemini35flash", CODES)

    def test_normalize_cell(self):
        assert normalize_cell("1") == 1
        assert normalize_cell(0.0) == 0
        assert normalize_cell(None) == "NA"
        assert normalize_cell(" na ") == "NA"
        with pytest.raises(DeliverableError):
            normalize_cell("yes")
        with pytest.raises(DeliverableError):
            normalize_cell(True)


class TestOsuImport:
    def test_rows_manifest_and_speaker_map(self, tmp_path):
        records = read_table(_osu_workbook(tmp_path / "osu.xlsx", OSU_ROWS))
        rows, manifest, speaker_map = import_osu_rows(records, "osu_s1")
        assert [r.uid for r in rows] == ["u001", "u002", "u003"]
        assert speaker_map == {"White24": "SPEAKER_00", "Confederate": "SPEAKER_01", "Latina23": "SPEAKER_02"}
        # No demographic label survives into the rows.
        text = json.dumps([r.model_dump() for r in rows])
        assert "White24" not in text and "Latina23" not in text and "Confederate" not in text
        assert rows[1].role == "confederate"
        assert rows[0].prior_speaker == "NA"
        assert rows[2].prior_speaker == "SPEAKER_01"
        assert rows[0].document == "osu_s1"
        assert rows[0].tone == "unknown"
        assert manifest.source_class == "osu_study"
        assert manifest.speaker_roles == {
            "SPEAKER_00": "participant", "SPEAKER_01": "confederate", "SPEAKER_02": "participant",
        }

    def test_missing_column_is_named(self):
        with pytest.raises(DeliverableError, match="utterance"):
            import_osu_rows([{"uid": "u001", ".ord": 1, "speaker": "x"}], "s")

    def test_empty_workbook_is_refused(self):
        with pytest.raises(DeliverableError, match="no rows"):
            import_osu_rows([], "s")

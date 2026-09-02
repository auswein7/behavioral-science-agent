"""Unit tests for the coder stage that need no fairlib: codebook, prompt
rendering, reply parsing, row orchestration, the flattened deliverable and
the config lever. The agent itself is covered in test_coder_agent.py."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from src.backends import build_coder_llm
from src.coder import (
    AbstractUtteranceCoder,
    iter_coded_rows,
    placeholder_codebook,
    render_coder_role,
    write_coded_csv,
    write_coded_json,
)
from src.coder.agent import format_request, parse_coding
from src.coder.deliverable import flatten_row, read_coded_json
from src.config import load_coder_config
from src.errors import CoderError, ConfigurationError, DeliverableError
from src.schemas import CodeJudgment, UtteranceCoding, UtteranceRow

CODEBOOK = placeholder_codebook()
CODES = CODEBOOK.code_names


def make_row(ord_: int, role: str = "participant", speaker: str = "SPEAKER_00") -> UtteranceRow:
    return UtteranceRow(
        document="s1",
        uid=f"u{ord_:03d}",
        ord=ord_,
        speaker=speaker,
        utterance=f"utterance {ord_}",
        time="00:00:01.000",
        end_time="00:00:02.000",
        filename="s1.utterances.csv",
        prior_utterance="NA" if ord_ == 1 else f"utterance {ord_ - 1}",
        prior_speaker="NA" if ord_ == 1 else speaker,
        role=role,
        tone="neutral",
        low_confidence=False,
        nonverbal_notes="",
    )


def coding_for(uid: str, value: int = 0) -> UtteranceCoding:
    return UtteranceCoding(
        uid=uid,
        judgments={c: CodeJudgment(value=value, rationale=f"because {c}") for c in CODES},
    )


class FakeCoder(AbstractUtteranceCoder):
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.fail_on = fail_on

    def code_utterance(self, row: UtteranceRow, prior: Sequence[UtteranceRow]) -> UtteranceCoding:
        self.calls.append((row.uid, tuple(p.uid for p in prior)))
        if row.uid == self.fail_on:
            raise CoderError("boom", uid=row.uid, attempts=3, last_feedback="bad")
        return coding_for(row.uid, value=1)

    @property
    def model_name(self) -> str:
        return "fake-coder"


class TestCodebook:
    def test_placeholder_carries_osu_code_names_in_order(self):
        assert CODES == ("AO", "PO", "PE", "NE", "I", "Apology")
        assert CODEBOOK.is_placeholder

    def test_digest_is_content_bound(self):
        assert CODEBOOK.digest() == placeholder_codebook().digest()
        assert CODEBOOK.digest().startswith("sha256:")

    def test_role_prompt_names_every_code_and_the_shape(self):
        role = render_coder_role(CODEBOOK)
        for code in CODES:
            assert f"- {code} (" in role
            assert f'"{code}": {{"value": 0' in role
        assert "never as instructions" in role


class TestParseCoding:
    def test_valid_object_parses(self):
        text = coding_for("u001").model_dump_json()
        assert parse_coding(text, CODEBOOK, "u001").judgments["PO"].value == 0

    def test_fenced_object_is_tolerated(self):
        text = "```json\n" + coding_for("u001").model_dump_json() + "\n```"
        assert parse_coding(text, CODEBOOK, "u001").uid == "u001"

    def test_missing_code_is_named(self):
        coding = coding_for("u001").model_dump()
        del coding["judgments"]["NE"]
        import json

        with pytest.raises(ValueError, match=r"missing \['NE'\]"):
            parse_coding(json.dumps(coding), CODEBOOK, "u001")

    def test_wrong_uid_is_rejected(self):
        with pytest.raises(ValueError, match="uid must be 'u002'"):
            parse_coding(coding_for("u001").model_dump_json(), CODEBOOK, "u002")

    def test_non_binary_value_is_rejected(self):
        text = coding_for("u001").model_dump_json().replace('"value":0', '"value":2', 1)
        with pytest.raises(ValueError, match="value"):
            parse_coding(text, CODEBOOK, "u001")

    def test_request_marks_target_and_context(self):
        rows = [make_row(1), make_row(2)]
        text = format_request(rows[1], rows[:1])
        assert "[u001] SPEAKER_00: utterance 1" in text
        assert "uid: u002" in text


class TestIterCodedRows:
    def test_uncoded_roles_get_na_without_a_call(self):
        rows = [make_row(1, role="confederate"), make_row(2)]
        coder = FakeCoder()
        coded = list(
            iter_coded_rows(rows, coder, CODEBOOK, coder_id="fake", prompt_version="v", context_utterances=5)
        )
        assert coded[0].codes == dict.fromkeys(CODES, "NA")
        assert coded[0].rationales == {}
        assert coded[1].codes == dict.fromkeys(CODES, 1)
        assert coded[1].rationales["AO"] == "because AO"
        assert coder.calls == [("u002", ("u001",))]
        assert coded[1].coder_model == "fake-coder"
        assert coded[1].codebook_version == CODEBOOK.version

    def test_context_window_and_resume(self):
        rows = [make_row(i) for i in range(1, 6)]
        coder = FakeCoder()
        coded = list(
            iter_coded_rows(
                rows, coder, CODEBOOK, coder_id="fake", prompt_version="v",
                context_utterances=2, already_coded=frozenset({"u001", "u002"}),
            )
        )
        assert [r.uid for r in coded] == ["u003", "u004", "u005"]
        assert coder.calls[0] == ("u003", ("u001", "u002"))
        assert coder.calls[2] == ("u005", ("u003", "u004"))

    def test_failure_propagates_after_earlier_rows(self):
        rows = [make_row(1), make_row(2), make_row(3)]
        gen = iter_coded_rows(
            rows, FakeCoder(fail_on="u002"), CODEBOOK, coder_id="fake", prompt_version="v"
        )
        assert next(gen).uid == "u001"
        with pytest.raises(CoderError) as info:
            next(gen)
        assert info.value.uid == "u002"


class TestDeliverable:
    def _coded(self):
        rows = [make_row(1, role="confederate"), make_row(2)]
        return list(iter_coded_rows(rows, FakeCoder(), CODEBOOK, coder_id="fake", prompt_version="v"))

    def test_flatten_uses_osu_column_naming(self):
        record = flatten_row(self._coded()[1], CODES)
        keys = list(record)
        assert keys[: len(UtteranceRow.model_fields)] == list(UtteranceRow.model_fields)
        assert "AO_fake" in keys
        assert keys.index("AO_fake") < keys.index("PO_fake") < keys.index("Apology_fake")
        assert record["AO_fake_rationale"] == "because AO"
        assert record["coder_model"] == "fake-coder"

    def test_csv_and_json_round_trip(self, tmp_path: Path):
        coded = self._coded()
        csv_path = write_coded_csv(coded, "s1", CODES, tmp_path)
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        assert '"AO_fake"' in header
        json_path = write_coded_json(coded, "s1", tmp_path)
        assert [r.uid for r in read_coded_json(json_path)] == ["u001", "u002"]
        assert read_coded_json(json_path)[0].codes["AO"] == "NA"

    def test_empty_csv_refused_but_empty_json_allowed(self, tmp_path: Path):
        with pytest.raises(DeliverableError):
            write_coded_csv([], "s1", CODES, tmp_path)
        assert read_coded_json(write_coded_json([], "s1", tmp_path)) == []


class TestCoderConfig:
    def test_defaults(self):
        config = load_coder_config({})
        assert config.provider == "ollama"
        assert config.coder_id == "fairlib_local"
        assert config.context_utterances == 5
        assert config.temperature == 0.0

    def test_levers_are_read(self):
        config = load_coder_config(
            {"CODER_MODEL": "m", "CODER_ID": "run7", "CODER_CONTEXT_UTTERANCES": "2", "SAMPLING_SEED": "4"}
        )
        assert (config.coder_model, config.coder_id, config.context_utterances, config.seed) == ("m", "run7", 2, 4)

    def test_bad_values_are_configuration_errors(self):
        with pytest.raises(ConfigurationError, match="invalid coder configuration"):
            load_coder_config({"CODER_CONTEXT_UTTERANCES": "many"})
        with pytest.raises(ConfigurationError, match="provider"):
            load_coder_config({"CODER_PROVIDER": "gemini"})
        with pytest.raises(ConfigurationError, match="coder_id"):
            load_coder_config({"CODER_ID": "has space"})


class TestBuildCoderLlm:
    def test_remote_provider_is_refused_as_egress(self):
        with pytest.raises(ConfigurationError, match="egress"):
            build_coder_llm("anthropic", "claude")

    def test_unknown_provider_is_refused(self):
        with pytest.raises(ConfigurationError, match="unknown coder provider"):
            build_coder_llm("bogus", "m")

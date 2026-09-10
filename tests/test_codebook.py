"""The codebook file lever: OSU's definitions arrive as a JSON document that
loads into the typed Codebook, or the run fails at startup naming the file.
No model, no network."""

import copy
import json
from pathlib import Path

import pytest

from src.coder.codebook import (
    PLACEHOLDER_VERSION,
    load_codebook,
    placeholder_codebook,
    resolve_codebook,
)
from src.config import load_coder_config
from src.errors import ConfigurationError

GOOD = {
    "version": "osu-2026-10",
    "codes": [
        {"code": "AO", "name": "Action Opportunity", "definition": "first definition"},
        {"code": "NE", "name": "Negative Evaluation", "definition": "second definition"},
    ],
}


def _write(tmp_path: Path, payload: object, name: str = "codebook.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


class TestLoadCodebook:
    def test_loads_a_valid_document(self, tmp_path):
        book = load_codebook(_write(tmp_path, GOOD))
        assert book.version == "osu-2026-10"
        assert book.code_names == ("AO", "NE")
        assert not book.is_placeholder

    def test_digest_tracks_the_definitions(self, tmp_path):
        changed = copy.deepcopy(GOOD)
        changed["codes"][0]["definition"] = "a different definition"
        a = load_codebook(_write(tmp_path, GOOD, "a.json"))
        b = load_codebook(_write(tmp_path, changed, "b.json"))
        assert a.digest() != b.digest()

    def test_missing_file_is_a_configuration_error(self, tmp_path):
        with pytest.raises(ConfigurationError, match=r"nope\.json"):
            load_codebook(tmp_path / "nope.json")

    def test_invalid_json_is_a_configuration_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ConfigurationError, match=r"bad\.json"):
            load_codebook(path)

    def test_schema_violation_is_a_configuration_error(self, tmp_path):
        bad = {"version": "v", "codes": [{"code": "A O", "name": "x", "definition": "y"}]}
        with pytest.raises(ConfigurationError, match="code"):
            load_codebook(_write(tmp_path, bad))

    def test_unknown_field_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError):
            load_codebook(_write(tmp_path, dict(GOOD, extra="x")))

    def test_duplicate_code_names_are_refused(self, tmp_path):
        dup = {"version": "v", "codes": [GOOD["codes"][0], GOOD["codes"][0]]}
        with pytest.raises(ConfigurationError, match="duplicate"):
            load_codebook(_write(tmp_path, dup))

    def test_the_example_document_loads_and_is_flagged_placeholder(self):
        # docs/codebook.example.json is the template OSU fills in; until its
        # version is changed a run using it is still marked a placeholder.
        path = Path(__file__).resolve().parents[1] / "docs" / "codebook.example.json"
        book = load_codebook(path)
        assert book.code_names == placeholder_codebook().code_names
        assert book.is_placeholder


class TestResolveCodebook:
    def test_unset_lever_is_the_placeholder(self):
        assert resolve_codebook(None).version == PLACEHOLDER_VERSION

    def test_set_lever_loads_the_file(self, tmp_path):
        assert resolve_codebook(str(_write(tmp_path, GOOD))).version == "osu-2026-10"

    def test_the_lever_is_read_from_the_environment(self):
        assert load_coder_config({"CODER_CODEBOOK": "book.json"}).codebook_path == "book.json"
        assert load_coder_config({}).codebook_path is None

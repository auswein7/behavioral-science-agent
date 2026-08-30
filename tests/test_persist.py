"""Unit tests for envelope persistence of stage record files."""

import json

import pytest

from src.errors import ArtifactError
from src.persist import read_records, write_records

RECORDS = [{"index": 0, "text": "hello"}, {"index": 1, "text": "world"}]


class TestRoundTrip:
    def test_envelope_written_and_read(self, tmp_path):
        path = tmp_path / "stage.json"
        write_records(RECORDS, path)
        payload = json.loads(path.read_text())
        assert payload["schema_version"] == "0.1"
        assert read_records(path) == RECORDS

    def test_legacy_bare_list_still_loads(self, tmp_path):
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(RECORDS))
        assert read_records(path) == RECORDS


class TestErrors:
    def test_missing_file_is_typed_error(self, tmp_path):
        with pytest.raises(ArtifactError):
            read_records(tmp_path / "absent.json")

    def test_malformed_payload_is_typed_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"schema_version": "0.1"}')
        with pytest.raises(ArtifactError):
            read_records(path)

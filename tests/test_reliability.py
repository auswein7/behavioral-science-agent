"""Unit tests for the adapter-layer retry primitive."""

import ollama
import pytest

from src.errors import AdapterError
from src.reliability import call_with_retry


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("src.reliability.time.sleep", lambda _s: None)


class TestCallWithRetry:
    def test_success_passes_through(self):
        assert call_with_retry(lambda: 42, description="answer") == 42

    def test_transient_failure_retried(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("blip")
            return "ok"

        assert call_with_retry(flaky, description="flaky call") == "ok"
        assert calls["n"] == 3

    def test_exhaustion_raises_adapter_error(self):
        def always_down():
            raise TimeoutError("still down")

        with pytest.raises(AdapterError, match="after 3 attempts"):
            call_with_retry(always_down, description="doomed call")

    def test_definitive_rejection_fails_immediately(self):
        calls = {"n": 0}

        def not_found():
            calls["n"] += 1
            raise ollama.ResponseError("model not found", status_code=404)

        with pytest.raises(AdapterError, match="rejected"):
            call_with_retry(not_found, description="bad model")
        assert calls["n"] == 1

    def test_server_error_is_retried(self):
        calls = {"n": 0}

        def flaky_server():
            calls["n"] += 1
            if calls["n"] == 1:
                raise ollama.ResponseError("boom", status_code=500)
            return "recovered"

        assert call_with_retry(flaky_server, description="server call") == "recovered"

    def test_unrelated_exception_propagates_untouched(self):
        with pytest.raises(KeyError):
            call_with_retry(lambda: {}["missing"], description="bug, not transport")

"""Unit tests for the orchestrator stage's typed-error contract, model mocked
at the adapter boundary (the stage itself never touches a provider SDK)."""

import importlib
from types import SimpleNamespace

import pytest

from src import adapters
from src.errors import AdapterError

# The package re-exports the function under the same name, so a plain
# "import src.screenplay.write_screenplay as ws" binds the function.
ws = importlib.import_module("src.screenplay.write_screenplay")


def fake_ollama(content: str, done_reason: str | None = "length"):
    return SimpleNamespace(
        list=lambda: SimpleNamespace(models=[SimpleNamespace(model="ornith-1.5-255k:latest")]),
        chat=lambda **kwargs: SimpleNamespace(
            message=SimpleNamespace(content=content),
            done_reason=done_reason,
            prompt_eval_count=110936,
            eval_count=20136,
        ),
    )


@pytest.fixture
def with_model_output(monkeypatch):
    def patch(content: str, done_reason: str | None = "length"):
        monkeypatch.setattr(adapters, "ollama", fake_ollama(content, done_reason))

    return patch


class TestWriteScreenplayEmptyContent:
    def test_empty_content_raises_typed_error(self, with_model_output):
        # A reasoning model that overflows its context mid-think returns 200 OK
        # with empty content; that must surface as a typed AdapterError, never
        # a silent empty screenplay (principle 6).
        with_model_output("")
        with pytest.raises(AdapterError, match="empty screenplay"):
            ws.write_screenplay([{"type": "speech"}], template="# T")

    def test_whitespace_content_raises_typed_error(self, with_model_output):
        with_model_output("  \n ")
        with pytest.raises(AdapterError):
            ws.write_screenplay([{"type": "speech"}], template="# T")

    def test_error_message_names_the_context_lever(self, with_model_output):
        with_model_output("")
        with pytest.raises(AdapterError, match="ORNITH_NUM_CTX"):
            ws.write_screenplay([{"type": "speech"}], template="# T")

    def test_overflow_remedy_comes_from_normalized_signal(self, with_model_output):
        # The remediation branch reads ChatResponse.truncated, not the raw
        # provider string; a truncated stop names the num_ctx lever.
        with_model_output("", done_reason="length")
        with pytest.raises(AdapterError, match="overflowed num_ctx"):
            ws.write_screenplay([{"type": "speech"}], template="# T")

    def test_non_overflow_stop_gets_no_overflow_remedy(self, with_model_output):
        with_model_output("", done_reason="stop")
        with pytest.raises(AdapterError, match="without a context overflow"):
            ws.write_screenplay([{"type": "speech"}], template="# T")

    def test_unknown_stop_reason_says_backend_does_not_report(self, with_model_output):
        # A backend with no stop reasons (fairlib until fair_llm #147) must say
        # so, not claim or deny overflow it cannot see (principle 6).
        with_model_output("", done_reason=None)
        with pytest.raises(AdapterError, match="does not report stop reasons"):
            ws.write_screenplay([{"type": "speech"}], template="# T")

    def test_real_content_returned_stripped(self, with_model_output):
        with_model_output("# Screenplay\n")
        assert ws.write_screenplay([{"type": "speech"}], template="# T") == "# Screenplay"

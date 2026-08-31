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


def fake_ollama(content: str):
    return SimpleNamespace(
        list=lambda: SimpleNamespace(models=[SimpleNamespace(model="ornith-1.5-255k:latest")]),
        chat=lambda **kwargs: SimpleNamespace(
            message=SimpleNamespace(content=content),
            done_reason="length",
            prompt_eval_count=110936,
            eval_count=20136,
        ),
    )


@pytest.fixture
def with_model_output(monkeypatch):
    def patch(content: str):
        monkeypatch.setattr(adapters, "ollama", fake_ollama(content))

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

    def test_real_content_returned_stripped(self, with_model_output):
        with_model_output("# Screenplay\n")
        assert ws.write_screenplay([{"type": "speech"}], template="# T") == "# Screenplay"

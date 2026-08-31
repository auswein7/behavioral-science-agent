"""Unit tests for the orchestrator stage's typed-error contract, model mocked."""

import importlib
from types import SimpleNamespace

import pytest

from src.errors import AdapterError

# The package re-exports the function under the same name, so a plain
# "import src.screenplay.write_screenplay as ws" binds the function.
ws = importlib.import_module("src.screenplay.write_screenplay")


def fake_response(content: str):
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        done_reason="length",
        prompt_eval_count=110936,
        eval_count=20136,
    )


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(ws, "_ensure_model_available", lambda name: None)
    return monkeypatch


class TestWriteScreenplayEmptyContent:
    def test_empty_content_raises_typed_error(self, patched):
        # A reasoning model that overflows its context mid-think returns 200 OK
        # with empty content; that must surface as a typed AdapterError, never
        # a silent empty screenplay (principle 6).
        patched.setattr(ws.ollama, "chat", lambda **kwargs: fake_response(""))
        with pytest.raises(AdapterError, match="empty screenplay"):
            ws.write_screenplay([{"type": "speech"}], template="# T")

    def test_whitespace_content_raises_typed_error(self, patched):
        patched.setattr(ws.ollama, "chat", lambda **kwargs: fake_response("  \n "))
        with pytest.raises(AdapterError):
            ws.write_screenplay([{"type": "speech"}], template="# T")

    def test_error_message_names_the_context_lever(self, patched):
        patched.setattr(ws.ollama, "chat", lambda **kwargs: fake_response(""))
        with pytest.raises(AdapterError, match="ORNITH_NUM_CTX"):
            ws.write_screenplay([{"type": "speech"}], template="# T")

    def test_real_content_returned_stripped(self, patched):
        patched.setattr(ws.ollama, "chat", lambda **kwargs: fake_response("# Screenplay\n"))
        assert ws.write_screenplay([{"type": "speech"}], template="# T") == "# Screenplay"

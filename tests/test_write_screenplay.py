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


class TestEnforceTitle:
    """The H1 line is fixed in code, not left to the model. Run 4's gate was
    blocked by four likely_name findings that were all mid-title capitals in
    a generated title ("# A Responsibility for Our Blue Planet"), with no real
    leak anywhere in the artifact."""

    def test_generated_title_is_replaced(self):
        md = "# A Responsibility for Our Blue Planet\n\n*A screenplay.*\n\n## SCENE 1"
        out = ws.enforce_title(md)
        assert out.startswith("# Screenplay\n")
        assert "Responsibility" not in out

    def test_canonical_title_is_left_alone(self):
        md = "# Screenplay\n\n## SCENE 1"
        assert ws.enforce_title(md) == md

    def test_scene_headings_are_untouched(self):
        # Only the first H1 is normalized; "## SCENE" lines are not H1 and a
        # later "# " line belongs to the body, not the title.
        md = "# Invented Title\n\n## SCENE 1 - [00:00:00.000 - 00:00:12.000]\n\n## SCENE 2"
        out = ws.enforce_title(md)
        assert "## SCENE 1 - [00:00:00.000 - 00:00:12.000]" in out
        assert "## SCENE 2" in out

    def test_missing_title_is_inserted(self):
        md = "## SCENE 1\n\n**SPEAKER_00** *(00:00:03.500)*"
        out = ws.enforce_title(md)
        assert out.startswith("# Screenplay\n\n")
        assert "## SCENE 1" in out

    def test_leading_blank_lines_do_not_hide_the_title(self):
        md = "\n\n# Some Invented Name\n\n## SCENE 1"
        out = ws.enforce_title(md)
        assert "# Screenplay" in out
        assert "Invented" not in out

    def test_enforced_title_clears_the_likely_name_gate(self):
        # The end-to-end point of the fix: the exact run-4 title produced four
        # likely_name findings; after enforcement the artifact produces none.
        from src.scrub.patterns import iter_findings

        before = "# A Responsibility for Our Blue Planet\n\n*A screenplay.*\n"
        after = ws.enforce_title(before)
        names_before = [t for category, t, _ in iter_findings(before) if category == "likely_name"]
        names_after = [t for category, t, _ in iter_findings(after) if category == "likely_name"]
        assert names_before, "expected the generated title to trip likely_name"
        assert names_after == []

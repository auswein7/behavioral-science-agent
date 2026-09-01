"""Unit tests for the caption backend factory (CAPTION_BACKEND lever)."""

import dataclasses

import pytest

from src.adapters import OllamaChatModel
from src.backends import build_caption_model, build_screenplay_model
from src.errors import ConfigurationError
from src.fairlib_adapter import FairlibChatModel


class TestBuildCaptionModel:
    def test_ollama_backend_builds_the_builtin_adapter(self):
        model = build_caption_model("ollama", "qwen3-vl:8b")
        assert isinstance(model, OllamaChatModel)
        assert model.model_name == "qwen3-vl:8b"
        assert model.auto_pull is True

    def test_unknown_backend_is_typed_error_not_a_fallback(self):
        # CLI entry points read the lever straight from the environment, so
        # the factory itself must refuse an unknown value (principle 6).
        with pytest.raises(ConfigurationError, match="langchain"):
            build_caption_model("langchain", "qwen3-vl:8b")

    def test_fairlib_backend_builds_the_fairlib_model_with_vision(self):
        message_mod = pytest.importorskip("fairlib.core.message")
        fields = {f.name for f in dataclasses.fields(message_mod.Message)}
        if "images" not in fields:
            pytest.skip("installed fairlib predates issue #146 vision support")
        model = build_caption_model("fairlib", "qwen3-vl:8b")
        assert isinstance(model, FairlibChatModel)
        assert model.model_name == "qwen3-vl:8b"
        # The captioner sends frames; the factory must hand back a
        # vision-declaring adapter or the stage would be refused at payload
        # build. No network: capabilities are declared, not probed.
        assert model.capabilities() == {"vision": True}


class TestBuildScreenplayModel:
    def test_ollama_backend_builds_ornith_without_auto_pull(self):
        model = build_screenplay_model("ollama", "ornith-1.5-255k", unavailable_hint="hint")
        assert isinstance(model, OllamaChatModel)
        assert model.model_name == "ornith-1.5-255k"
        # Ornith is not on the public registry; a missing model must be a
        # typed error carrying the hint, never a doomed pull attempt.
        assert model.auto_pull is False
        assert model.unavailable_hint == "hint"

    def test_fairlib_backend_capability_gated_on_147(self):
        # The gate probes the installed fairlib: without #147 usage telemetry
        # the lever is refused with the blocker named (a constructed model
        # would silently degrade the overflow diagnostic, principle 6); with
        # it, construction succeeds and Ornith declares no vision.
        from src.fairlib_adapter import fairlib_reports_usage

        if fairlib_reports_usage():
            model = build_screenplay_model(
                "fairlib", "ornith-1.5-255k", unavailable_hint="hint"
            )
            assert isinstance(model, FairlibChatModel)
            assert model.capabilities() == {"vision": False}
        else:
            with pytest.raises(ConfigurationError, match="#147"):
                build_screenplay_model("fairlib", "ornith-1.5-255k")

    def test_unknown_backend_is_typed_error(self):
        with pytest.raises(ConfigurationError, match="langchain"):
            build_screenplay_model("langchain", "ornith-1.5-255k")

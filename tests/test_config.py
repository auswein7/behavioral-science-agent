"""Unit tests for the typed config loader and preflight checks."""

import pytest

from src.config import load_run_config, preflight, prompt_versions
from src.errors import ConfigurationError


class TestLoadRunConfig:
    def test_defaults_from_empty_env(self):
        config = load_run_config(env={})
        assert config.transcription.whisper_model == "large-v2"
        assert config.captioning.caption_model == "qwen3-vl:8b"
        assert config.captioning.context_captions == 1
        assert config.screenplay.ornith_model == "ornith-1.5-255k"
        assert config.sampling.temperature == 0.0

    def test_env_overrides(self):
        config = load_run_config(env={"CAPTION_MODEL": "qwen3-vl-16k", "WHISPERX_NUM_SPEAKERS": "7"})
        assert config.captioning.caption_model == "qwen3-vl-16k"
        assert config.transcription.num_speakers == 7

    def test_all_and_none_sentinels(self):
        config = load_run_config(env={"CAPTION_CONTEXT_CAPTIONS": "all", "CAPTION_MAX_DIMENSION": "none"})
        assert config.captioning.context_captions is None
        assert config.captioning.max_dimension is None

    def test_malformed_value_is_typed_error(self):
        with pytest.raises(ConfigurationError):
            load_run_config(env={"CAPTION_FPS": "one"})

    def test_prompt_versions_recorded(self):
        config = load_run_config(env={})
        assert "ANONYMIZATION_RULES" in config.prompt_versions
        assert config.prompt_versions["ANONYMIZATION_RULES"].startswith("sha256:")

    def test_secrets_never_in_config(self):
        config = load_run_config(env={"HF_TOKEN": "hf_secret123"})
        assert "hf_secret123" not in config.model_dump_json()


class TestPromptVersions:
    def test_digests_are_stable(self):
        assert prompt_versions() == prompt_versions()


class TestPreflight:
    @pytest.fixture(autouse=True)
    def fake_ollama(self, monkeypatch):
        monkeypatch.setattr(
            "src.config._ollama_model_names",
            lambda: {"ornith-1.5-255k:latest", "qwen3-vl:8b"},
        )

    def make_config(self, **env):
        return load_run_config(env={"WHISPERX_DEVICE": "cpu", **env})

    def test_passes_when_models_present(self):
        preflight(self.make_config(), hf_token="token")

    def test_missing_hf_token_blocks_transcription(self):
        with pytest.raises(ConfigurationError):
            preflight(self.make_config(), hf_token=None, need_transcription=True)

    def test_missing_token_ok_when_resuming_past_transcription(self):
        preflight(self.make_config(), hf_token=None, need_transcription=False)

    def test_missing_ornith_model_is_fatal(self):
        with pytest.raises(ConfigurationError):
            preflight(self.make_config(ORNITH_MODEL="ornith-9000"), hf_token="token")

    def test_missing_caption_model_is_not_fatal(self):
        preflight(self.make_config(CAPTION_MODEL="qwen3-vl:30b-a3b"), hf_token="token")

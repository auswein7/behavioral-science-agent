"""The run record's reproducibility claim comes from measurement, never
from a capability declaration, and the local context window is pinned
because it changes codes. No model, no network."""

from src import reproducibility
from src.config import load_coder_config
from src.reproducibility import MEASUREMENTS, Measurement, reproducibility_of


class TestReproducibilityOf:
    def test_the_local_coder_model_is_recorded_as_not_reproducible(self):
        # Same code, prompt, seed and temperature moved code cells across
        # server states (2026-09-10), so the record must not claim otherwise.
        record = reproducibility_of("ollama", "qwen2.5:14b", seed=1234, temperature=0.0)
        assert record.status == "measured_not_reproducible"
        assert "num_ctx" in record.evidence
        assert record.seed == 1234

    def test_the_osu_parity_model_is_recorded_as_not_reproducible(self):
        # gemini-3.5-flash declares seed, yet varied at temperature 0.
        record = reproducibility_of("gemini", "gemini-3.5-flash", seed=1234, temperature=0.0)
        assert record.status == "measured_not_reproducible"

    def test_an_unmeasured_backend_is_never_assumed_reproducible(self):
        record = reproducibility_of("gemini", "gemini-3.6-flash", seed=1234, temperature=0.0)
        assert record.status == "unmeasured"
        assert "declared seed" in record.evidence

    def test_back_to_back_stability_alone_is_not_a_measurement(self):
        assert reproducibility_of("ollama", "llama3.1:8b", seed=1234, temperature=0.0).status == (
            "unmeasured"
        )

    def test_a_positive_measurement_does_not_extend_to_other_temperatures(self, monkeypatch):
        monkeypatch.setattr(
            reproducibility,
            "MEASUREMENTS",
            {("ollama", "m"): Measurement(status="measured_reproducible", evidence="2026 x")},
        )
        assert reproducibility.reproducibility_of("ollama", "m", seed=1, temperature=0.0).status == (
            "measured_reproducible"
        )
        record = reproducibility.reproducibility_of("ollama", "m", seed=1, temperature=0.7)
        assert record.status == "unmeasured"
        assert "0.7" in record.evidence

    def test_a_negative_measurement_holds_at_any_temperature(self):
        record = reproducibility_of("gemini", "gemini-3.5-flash", seed=None, temperature=0.7)
        assert record.status == "measured_not_reproducible"

    def test_every_entry_cites_dated_evidence(self):
        for measurement in MEASUREMENTS.values():
            assert measurement.evidence[:4] == "2026"


class TestPinnedContext:
    def test_the_local_context_window_is_pinned_and_recorded(self):
        # Unset, Ollama picks the window itself (32768 on 0.33.1) and it
        # changes codes; the pinned value is what the provenance records.
        assert load_coder_config({}).num_ctx == 8192

    def test_an_explicit_value_wins(self):
        assert load_coder_config({"CODER_NUM_CTX": "4096"}).num_ctx == 4096

    def test_a_hosted_model_gets_no_context_option(self):
        assert load_coder_config({"CODER_PROVIDER": "gemini"}).num_ctx is None

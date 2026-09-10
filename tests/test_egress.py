"""The egress gate (ADR 0001): our decision first, fairlib's grant as the
backstop. No network: the Gemini adapter constructs offline, and the
backstop test is refused by fairlib before any transport call."""

import pytest

from src.backends import build_coder_llm
from src.egress import (
    REMOTE_DESTINATIONS,
    capability_bag_for,
    decide_egress,
    security_manager_for,
)
from src.errors import ConfigurationError, GateBlockedError
from src.schemas import ScrubReport, SessionManifest

GEMINI_HOST = "generativelanguage.googleapis.com"


def manifest(source_class: str = "public_domain", session_id: str = "pd1") -> SessionManifest:
    return SessionManifest(
        session_id=session_id,
        source_class=source_class,
        task_name="t",
        group_size=2,
        expected_speaker_count=2,
    )


def report(
    resolution: str = "clean", session_id: str = "pd1", artifact: str | None = None
) -> ScrubReport:
    return ScrubReport(
        session_id=session_id,
        run_id="r",
        artifact=artifact or f"{session_id}.utterances.csv",
        resolution=resolution,
        decided="2026-09-10T00:00:00+00:00",
    )


def authorized(model: str = "gemini-3.5-flash"):
    return decide_egress(
        "gemini", model, session_id="pd1", manifest=manifest(), scrub_report=report()
    )


class TestDecideEgress:
    def test_local_needs_nothing(self):
        decision = decide_egress("ollama", "qwen2.5:14b", session_id="x")
        assert not decision.remote
        assert decision.destination is None

    def test_local_coding_of_human_subjects_data_is_allowed(self):
        decision = decide_egress(
            "ollama", "qwen2.5:14b", session_id="pd1", manifest=manifest("cadet_pii")
        )
        assert not decision.remote
        assert decision.source_class == "cadet_pii"

    def test_public_domain_with_a_clean_report_is_authorized(self):
        decision = authorized()
        assert decision.remote
        assert decision.destination == f"{GEMINI_HOST}:443"
        assert decision.scrub_resolution == "clean"
        assert decision.source_class == "public_domain"

    def test_the_json_form_of_the_table_counts_as_gated(self):
        decision = decide_egress(
            "gemini",
            "gemini-3.5-flash",
            session_id="pd1",
            manifest=manifest(),
            scrub_report=report(artifact="pd1.utterances.json"),
        )
        assert decision.remote

    @pytest.mark.parametrize("source_class", ["cadet_pii", "osu_study"])
    def test_human_subjects_data_never_leaves(self, source_class):
        with pytest.raises(GateBlockedError, match="public_domain"):
            decide_egress(
                "gemini",
                "gemini-3.5-flash",
                session_id="pd1",
                manifest=manifest(source_class),
                scrub_report=report(),
            )

    @pytest.mark.parametrize("resolution", ["cleared_by_review", "blocked", "redacted"])
    def test_only_a_clean_report_authorizes_egress(self, resolution):
        with pytest.raises(GateBlockedError, match="clean"):
            decide_egress(
                "gemini",
                "gemini-3.5-flash",
                session_id="pd1",
                manifest=manifest(),
                scrub_report=report(resolution),
            )

    def test_missing_manifest_or_report_is_blocked(self):
        with pytest.raises(GateBlockedError, match="manifest"):
            decide_egress("gemini", "m", session_id="pd1", scrub_report=report())
        with pytest.raises(GateBlockedError, match="scrub report"):
            decide_egress("gemini", "m", session_id="pd1", manifest=manifest())

    def test_evidence_for_another_session_or_artifact_is_blocked(self):
        with pytest.raises(GateBlockedError, match="manifest is for"):
            decide_egress(
                "gemini", "m", session_id="pd1",
                manifest=manifest(session_id="pd2"), scrub_report=report(),
            )
        with pytest.raises(GateBlockedError, match="covers"):
            decide_egress(
                "gemini", "m", session_id="pd1",
                manifest=manifest(), scrub_report=report(session_id="pd2"),
            )
        with pytest.raises(GateBlockedError, match="covers"):
            decide_egress(
                "gemini", "m", session_id="pd1",
                manifest=manifest(), scrub_report=report(artifact="pd1.screenplay.md"),
            )

    def test_a_provider_off_the_list_is_blocked(self):
        assert set(REMOTE_DESTINATIONS) == {"gemini"}
        with pytest.raises(GateBlockedError, match="ruling 2"):
            decide_egress(
                "anthropic", "claude", session_id="pd1",
                manifest=manifest(), scrub_report=report(),
            )


class TestCapabilityBag:
    def test_a_remote_decision_grants_the_model_and_its_host_only(self):
        pytest.importorskip("fairlib")
        bag = capability_bag_for(authorized())
        assert bag.allows_model("gemini-3.5-flash")
        assert not bag.allows_model("gemini-3.6-flash")
        assert bag.allows_host(GEMINI_HOST, 443)
        assert not bag.allows_host("api.anthropic.com", 443)

    def test_a_local_decision_grants_no_host(self):
        pytest.importorskip("fairlib")
        bag = capability_bag_for(decide_egress("ollama", "qwen2.5:14b", session_id="x"))
        assert bag.allows_model("qwen2.5:14b")
        assert not bag.allows_model("gemini-3.5-flash")
        assert not bag.allows_host(GEMINI_HOST, 443)


class TestBuildCoderLlm:
    def test_gemini_without_a_decision_is_blocked(self):
        with pytest.raises(GateBlockedError, match="no egress decision"):
            build_coder_llm("gemini", "gemini-3.5-flash")

    def test_a_decision_for_something_else_does_not_authorize(self):
        local = decide_egress("ollama", "gemini-3.5-flash", session_id="pd1")
        with pytest.raises(GateBlockedError):
            build_coder_llm("gemini", "gemini-3.5-flash", egress=local)
        with pytest.raises(GateBlockedError):
            build_coder_llm("gemini", "gemini-3.6-flash", egress=authorized())

    def test_providers_off_the_list_and_unknown_ones(self):
        with pytest.raises(GateBlockedError, match="egress"):
            build_coder_llm("anthropic", "claude")
        with pytest.raises(ConfigurationError, match="unknown coder provider"):
            build_coder_llm("bogus", "m")

    def test_num_ctx_is_refused_for_a_hosted_model(self):
        with pytest.raises(ConfigurationError, match="CODER_NUM_CTX"):
            build_coder_llm("gemini", "gemini-3.5-flash", num_ctx=8192, egress=authorized())

    def test_an_authorized_gemini_model_constructs_offline(self, monkeypatch):
        pytest.importorskip("google.genai")
        gemini = pytest.importorskip("fairlib.modules.mal.gemini_adapter")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
        llm = build_coder_llm("gemini", "gemini-3.5-flash", egress=authorized())
        assert isinstance(llm, gemini.GeminiAdapter)
        assert llm.describe_config().model_name == "gemini-3.5-flash"
        # The coder's default options (seed included) are carried by Gemini.
        from src.coder import (
            FairlibAgentCoder,
            coder_generation_options,
            placeholder_codebook,
        )

        FairlibAgentCoder(
            llm,
            placeholder_codebook(),
            generation_options=coder_generation_options(temperature=0.0, seed=1234, max_tokens=512),
        )

    def test_a_missing_key_is_a_configuration_error(self, monkeypatch):
        pytest.importorskip("google.genai")
        pytest.importorskip("fairlib.modules.mal.gemini_adapter")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
            build_coder_llm("gemini", "gemini-3.5-flash", egress=authorized())


class TestFairlibBackstop:
    """If our code ever handed the coder a model its decision did not cover,
    fairlib's grant still refuses the call, before any transport."""

    def test_a_model_outside_the_grant_is_refused_before_the_wire(self):
        pytest.importorskip("fairlib.modules.mal.local_llama_adapter")
        from test_coder import make_row

        from src.coder import FairlibAgentCoder, placeholder_codebook

        # The decision grants a different model than the one the coder holds.
        other = decide_egress("ollama", "some-other-model", session_id="x")
        llm = build_coder_llm("ollama", "qwen2.5:14b")
        coder = FairlibAgentCoder(
            llm, placeholder_codebook(), security_manager=security_manager_for(other)
        )
        with pytest.raises(GateBlockedError, match="denied"):
            coder.code_utterance(make_row(1), [])

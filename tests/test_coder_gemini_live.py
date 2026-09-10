"""One live coding through public Gemini, behind the full egress gate.

Self-skips without GEMINI_API_KEY (workspace test rule). Spends two to four
requests of the key's quota. The row is synthetic, so nothing real leaves
the box; the gate is exercised exactly as a real public_domain run would be.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="needs GEMINI_API_KEY or GOOGLE_API_KEY",
)


def test_live_gemini_coding_behind_the_gate():
    pytest.importorskip("google.genai")
    from test_coder import make_row
    from test_egress import authorized

    from src.adapters import UsageTally
    from src.backends import build_coder_llm
    from src.coder import (
        FairlibAgentCoder,
        coder_generation_options,
        placeholder_codebook,
    )
    from src.egress import security_manager_for

    model = os.environ.get("FAIR_LLM_LIVE_GEMINI_MODEL", "gemini-3.5-flash")
    decision = authorized(model)
    tally = UsageTally()
    coder = FairlibAgentCoder(
        build_coder_llm("gemini", model, egress=decision),
        placeholder_codebook(),
        usage_tally=tally,
        generation_options=coder_generation_options(temperature=0.0, seed=1234, max_tokens=2048),
        security_manager=security_manager_for(decision),
    )
    coding = coder.code_utterance(make_row(1), [])
    assert coding.uid == "u001"
    assert set(coding.judgments) == set(placeholder_codebook().code_names)
    assert tally.calls >= 1
    assert tally.calls_reporting >= 1

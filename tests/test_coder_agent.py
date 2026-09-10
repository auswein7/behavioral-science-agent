"""The coder agent against real fairlib: a scripted fairlib chat model
drives SimpleAgent through the ReAct wire format, the validator, the
framework's rewrite retry and the typed exhaustion. Self-skips without
fair-llm (workspace test rule). No network."""

import asyncio
import json
import logging
from typing import Any

import pytest

fairlib = pytest.importorskip("fairlib", reason="fair-llm not installed")

from fairlib.core.interfaces.llm import (  # noqa: E402
    AbstractChatModel,
    ModelDescription,
)
from fairlib.core.message import Message  # noqa: E402
from test_coder import coding_for, make_row  # noqa: E402

from src.adapters import UsageTally  # noqa: E402
from src.backends import build_coder_llm  # noqa: E402
from src.coder import (  # noqa: E402
    FairlibAgentCoder,
    coder_generation_options,
    coding_validator,
    placeholder_codebook,
)
from src.errors import CoderBudgetError, CoderError, ConfigurationError  # noqa: E402
from src.schemas import UtteranceRow  # noqa: E402

CODEBOOK = placeholder_codebook()
OPTIONS = coder_generation_options(temperature=0.0, seed=1234, max_tokens=64)


def react_final(payload: str) -> str:
    return f"Thought: coded.\nAction:\ntool_name: final_answer\ntool_input: {payload}"


class Cut(str):
    """A scripted reply the model stopped on the output budget."""


class ScriptedChatModel(AbstractChatModel):
    """Replies in order; records every message list and every call's options.
    A Cut reply carries done_reason LENGTH on its Message and its event."""

    def __init__(self, replies: list[str], capabilities: dict[str, Any] | None = None) -> None:
        self.replies = list(replies)
        self.requests: list[list[Message]] = []
        self.call_options: list[dict[str, Any]] = []
        self.capabilities = capabilities or {"supports_streaming": False}

    def _next(self, messages: list, kwargs: dict[str, Any]) -> Message:
        from fairlib.core.message import DoneReason, Usage

        self.requests.append(list(messages))
        self.call_options.append(dict(kwargs))
        if not self.replies:
            raise AssertionError("script exhausted")
        reply = self.replies.pop(0)
        cut = isinstance(reply, Cut)
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=2,
            done_reason=DoneReason.LENGTH if cut else DoneReason.STOP,
            raw_done_reason="length" if cut else "stop",
        )
        self._emit(len(messages), usage)
        return Message(role="assistant", content=str(reply), usage=usage)

    def _emit(self, message_count: int, usage: Any) -> None:
        # A real adapter emits one ModelInvocationEvent per call on the bus
        # the agent bound; the script does the same so accounting is tested.
        bus = getattr(self, "_event_bus", None)
        if bus is None:
            return
        from fairlib.core.events import ModelInvocationEvent, ModelInvocationOutcome

        bus.emit(
            ModelInvocationEvent(
                model_name="scripted",
                provider="mal",
                duration_ms=1.0,
                outcome=ModelInvocationOutcome.COMPLETED,
                request_digest="d",
                message_count=message_count,
                usage=usage,
            )
        )

    def invoke(self, messages, **kwargs):
        return self._next(messages, kwargs)

    async def ainvoke(self, messages, **kwargs):
        return self._next(messages, kwargs)

    def stream(self, messages, **kwargs):
        yield self._next(messages, kwargs)

    async def astream(self, messages, **kwargs):
        yield self._next(messages, kwargs)

    def get_model_capabilities(self) -> dict[str, Any]:
        return dict(self.capabilities)

    def describe_config(self) -> ModelDescription:
        return ModelDescription(
            adapter="ScriptedChatModel", model_name="scripted", provider="mal", adapter_kwargs={}
        )


def _rows() -> list[UtteranceRow]:
    return [make_row(1), make_row(2, speaker="SPEAKER_01")]


def _bad_draft(uid: str = "u001") -> str:
    bad = coding_for(uid).model_dump()
    del bad["judgments"]["I"]
    return json.dumps(bad)


class TestValidator:
    def test_approves_and_rejects(self):
        validate = coding_validator(CODEBOOK, "u001")
        assert asyncio.run(validate(coding_for("u001").model_dump_json())).approved
        verdict = asyncio.run(validate('{"uid": "u001", "judgments": {}}'))
        assert not verdict.approved
        assert "missing" in verdict.feedback


class TestFairlibAgentCoder:
    def test_first_answer_is_validated_and_returned(self):
        good = coding_for("u002", value=1).model_dump_json()
        llm = ScriptedChatModel([react_final(good)])
        coder = FairlibAgentCoder(llm, CODEBOOK)
        rows = _rows()
        coding = coder.code_utterance(rows[1], rows[:1])
        assert coding.uid == "u002"
        assert coding.judgments["PO"].value == 1
        assert coder.model_name == "scripted"
        [request] = llm.requests
        text = "\n".join(m.content for m in request)
        assert "TARGET UTTERANCE" in text
        assert "[u001] SPEAKER_00" in text
        assert "CODEBOOK" in text

    def test_rejected_draft_is_rewritten_with_feedback(self):
        good = coding_for("u001").model_dump_json()
        llm = ScriptedChatModel([react_final(_bad_draft()), good])
        coding = FairlibAgentCoder(llm, CODEBOOK, max_retries=1).code_utterance(_rows()[0], [])
        assert set(coding.judgments) == set(CODEBOOK.code_names)
        assert len(llm.requests) == 2
        rewrite = "\n".join(m.content for m in llm.requests[1])
        assert "rejected" in rewrite
        assert "missing ['I']" in rewrite

    def test_exhaustion_is_a_coder_error_with_the_uid(self):
        llm = ScriptedChatModel([react_final("{}"), "{}", "{}"])
        with pytest.raises(CoderError) as info:
            FairlibAgentCoder(llm, CODEBOOK, max_retries=2).code_utterance(_rows()[0], [])
        assert info.value.uid == "u001"
        assert info.value.attempts == 3
        assert info.value.last_feedback
        # No call was cut off, so this is a prompt-shaped failure, not a budget one.
        assert not isinstance(info.value, CoderBudgetError)

    def test_usage_is_counted_from_the_agents_bus(self):
        llm = ScriptedChatModel([react_final(_bad_draft()), coding_for("u001").model_dump_json()])
        tally = UsageTally()
        FairlibAgentCoder(llm, CODEBOOK, max_retries=1, usage_tally=tally).code_utterance(
            _rows()[0], []
        )
        assert tally.source == "fairlib_events"
        assert tally.calls == 2
        assert tally.calls_reporting == 2
        assert tally.prompt_tokens == 20

    def test_sync_entry_refuses_a_running_loop(self):
        coder = FairlibAgentCoder(ScriptedChatModel([]), CODEBOOK)

        async def inside():
            coder.code_utterance(_rows()[0], [])

        with pytest.raises(ConfigurationError, match="acode_utterance"):
            asyncio.run(inside())


class TestGenerationOptions:
    """Sampling and the output budget are run options on fairlib's neutral
    channel (fair_llm #186): they reach every model call, and a name the
    bound model does not declare is refused before the first row."""

    def test_unset_levers_are_left_out(self):
        assert coder_generation_options(temperature=0.0, seed=None, max_tokens=None) == {
            "temperature": 0.0
        }
        assert OPTIONS == {"temperature": 0.0, "seed": 1234, "max_tokens": 64}

    def test_options_reach_the_planner_turn_and_the_rewrite(self):
        llm = ScriptedChatModel([react_final(_bad_draft()), coding_for("u001").model_dump_json()])
        FairlibAgentCoder(
            llm, CODEBOOK, max_retries=1, generation_options=OPTIONS
        ).code_utterance(_rows()[0], [])
        assert len(llm.call_options) == 2
        for sent in llm.call_options:
            assert {name: sent.get(name) for name in OPTIONS} == OPTIONS

    def test_no_options_sends_none(self):
        llm = ScriptedChatModel([react_final(coding_for("u001").model_dump_json())])
        FairlibAgentCoder(llm, CODEBOOK).code_utterance(_rows()[0], [])
        assert not set(llm.call_options[0]) & set(OPTIONS)

    def test_undeclared_option_is_refused_at_construction(self):
        # A backend that does not carry seed must not run with the seed
        # silently dropped: the run record would claim a determinism the
        # backend never offered.
        llm = ScriptedChatModel(
            [], capabilities={"generation_options": frozenset({"temperature", "max_tokens"})}
        )
        with pytest.raises(ConfigurationError, match="seed"):
            FairlibAgentCoder(llm, CODEBOOK, generation_options=OPTIONS)
        assert llm.requests == []

    def test_a_model_declaring_nothing_is_left_alone(self):
        FairlibAgentCoder(ScriptedChatModel([]), CODEBOOK, generation_options=OPTIONS)


class TestTruncation:
    """A coding that fails after a call stopped on the output budget is a
    CoderBudgetError naming the budget, never a validator or step-limit
    failure that points at the prompt (TODO item b2)."""

    def test_cut_rewrites_are_a_budget_error(self):
        llm = ScriptedChatModel([react_final(_bad_draft()), Cut('{"uid": "u0'), Cut('{"ui')])
        coder = FairlibAgentCoder(llm, CODEBOOK, max_retries=2, generation_options=OPTIONS)
        with pytest.raises(CoderBudgetError) as info:
            coder.code_utterance(_rows()[0], [])
        assert isinstance(info.value, CoderError)
        assert info.value.uid == "u001"
        assert info.value.attempts == 3
        assert info.value.max_tokens == 64
        assert "CODER_MAX_TOKENS" in str(info.value)

    def test_cut_planner_turns_are_a_budget_error(self):
        # A thinking model given too small a budget returns no text at all.
        llm = ScriptedChatModel([Cut("")] * 12)
        with pytest.raises(CoderBudgetError) as info:
            FairlibAgentCoder(llm, CODEBOOK).code_utterance(_rows()[0], [])
        assert info.value.max_tokens is None
        assert "backend default" in str(info.value)

    def test_a_recovered_cut_returns_the_coding_and_warns(self, caplog):
        llm = ScriptedChatModel(
            [react_final(_bad_draft()), Cut("{"), coding_for("u001").model_dump_json()]
        )
        with caplog.at_level(logging.WARNING, logger="src.coder.agent"):
            coding = FairlibAgentCoder(llm, CODEBOOK, max_retries=2).code_utterance(
                _rows()[0], []
            )
        assert coding.uid == "u001"
        assert "cut off by the output budget" in caplog.text

    def test_the_watch_resets_between_codings(self):
        llm = ScriptedChatModel(
            [
                react_final(_bad_draft()),
                Cut("{"),
                coding_for("u001").model_dump_json(),
                react_final("{}"),
                "{}",
                "{}",
            ]
        )
        coder = FairlibAgentCoder(llm, CODEBOOK, max_retries=2)
        rows = _rows()
        coder.code_utterance(rows[0], [])
        with pytest.raises(CoderError) as info:
            coder.code_utterance(rows[1], rows[:1])
        assert not isinstance(info.value, CoderBudgetError)


class TestBuildCoderLlm:
    def test_ollama_provider_builds_a_fairlib_adapter(self):
        mal = pytest.importorskip("fairlib.modules.mal.local_llama_adapter")
        llm = build_coder_llm("ollama", "qwen3:8b", num_ctx=8192)
        assert isinstance(llm, mal.OllamaAdapter)
        assert llm.describe_config().model_name == "qwen3:8b"
        assert llm.get_model_capabilities()["vision"] is False

    def test_ollama_carries_the_coders_default_options(self):
        # The live coder's configuration must pass its own construction check.
        pytest.importorskip("fairlib.modules.mal.local_llama_adapter")
        llm = build_coder_llm("ollama", "qwen3:8b")
        FairlibAgentCoder(llm, CODEBOOK, generation_options=OPTIONS)


class TestFrameworkProvenance:
    """The coder's system prompt is fairlib's, not ours, so the run record has
    to carry a digest of what the model actually saw. Two trees reporting the
    same version rendered different planner prompts and changed 3 of 5 codes
    (2026-09-08); the version did not discriminate and the digest did."""

    def test_reports_version_digest_and_length(self):
        coder = FairlibAgentCoder(llm=ScriptedChatModel([]), codebook=placeholder_codebook())
        prov = coder.framework_provenance
        assert set(prov) == {"fairlib_version", "planner_prompt_digest", "planner_prompt_chars"}
        assert len(prov["planner_prompt_digest"]) == 32
        assert int(prov["planner_prompt_chars"]) > 0

    def test_digest_is_stable_for_the_same_codebook(self):
        book = placeholder_codebook()
        first = FairlibAgentCoder(llm=ScriptedChatModel([]), codebook=book).framework_provenance
        second = FairlibAgentCoder(llm=ScriptedChatModel([]), codebook=book).framework_provenance
        assert first == second

    def test_digest_tracks_the_rendered_prompt(self):
        # The digest must cover the role text the codebook produces, or a
        # codebook swap would leave the run record unchanged.
        from src.coder.codebook import Codebook, CodeDefinition

        other = Codebook(
            version="test-codebook",
            codes=(CodeDefinition(code="XX", name="Example", definition="a different code"),),
        )
        a = FairlibAgentCoder(llm=ScriptedChatModel([]), codebook=placeholder_codebook())
        b = FairlibAgentCoder(llm=ScriptedChatModel([]), codebook=other)
        assert a.framework_provenance["planner_prompt_digest"] != b.framework_provenance[
            "planner_prompt_digest"
        ]

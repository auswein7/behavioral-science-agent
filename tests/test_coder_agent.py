"""The coder agent against real fairlib: a scripted fairlib chat model
drives SimpleAgent through the ReAct wire format, the validator, the
framework's rewrite retry and the typed exhaustion. Self-skips without
fair-llm (workspace test rule). No network."""

import asyncio
import json
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
    coding_validator,
    placeholder_codebook,
)
from src.errors import CoderError, ConfigurationError  # noqa: E402
from src.schemas import UtteranceRow  # noqa: E402

CODEBOOK = placeholder_codebook()


def react_final(payload: str) -> str:
    return f"Thought: coded.\nAction:\ntool_name: final_answer\ntool_input: {payload}"


class ScriptedChatModel(AbstractChatModel):
    """Replies in order; records every message list it was sent."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[list[Message]] = []

    def _next(self, messages: list) -> Message:
        self.requests.append(list(messages))
        if not self.replies:
            raise AssertionError("script exhausted")
        self._emit(len(messages))
        return Message(role="assistant", content=self.replies.pop(0))

    def _emit(self, message_count: int) -> None:
        # A real adapter emits one ModelInvocationEvent per call on the bus
        # the agent bound; the script does the same so accounting is tested.
        bus = getattr(self, "_event_bus", None)
        if bus is None:
            return
        from fairlib.core.events import ModelInvocationEvent, ModelInvocationOutcome
        from fairlib.core.message import DoneReason, Usage

        bus.emit(
            ModelInvocationEvent(
                model_name="scripted",
                provider="mal",
                duration_ms=1.0,
                outcome=ModelInvocationOutcome.COMPLETED,
                request_digest="d",
                message_count=message_count,
                usage=Usage(
                    prompt_tokens=10,
                    completion_tokens=2,
                    done_reason=DoneReason.STOP,
                    raw_done_reason="stop",
                ),
            )
        )

    def invoke(self, messages, **kwargs):
        return self._next(messages)

    async def ainvoke(self, messages, **kwargs):
        return self._next(messages)

    def stream(self, messages, **kwargs):
        yield self._next(messages)

    async def astream(self, messages, **kwargs):
        yield self._next(messages)

    def get_model_capabilities(self) -> dict[str, Any]:
        return {"supports_streaming": False}

    def describe_config(self) -> ModelDescription:
        return ModelDescription(
            adapter="ScriptedChatModel", model_name="scripted", provider="mal", adapter_kwargs={}
        )


def _rows() -> list[UtteranceRow]:
    return [make_row(1), make_row(2, speaker="SPEAKER_01")]


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
        bad = coding_for("u001").model_dump()
        del bad["judgments"]["I"]
        good = coding_for("u001").model_dump_json()
        llm = ScriptedChatModel([react_final(json.dumps(bad)), good])
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

    def test_usage_is_counted_from_the_agents_bus(self):
        bad = coding_for("u001").model_dump()
        del bad["judgments"]["I"]
        llm = ScriptedChatModel([react_final(json.dumps(bad)), coding_for("u001").model_dump_json()])
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


class TestBuildCoderLlm:
    def test_ollama_provider_builds_a_fairlib_adapter(self):
        mal = pytest.importorskip("fairlib.modules.mal.local_llama_adapter")
        llm = build_coder_llm("ollama", "qwen3:8b", seed=7, num_ctx=8192)
        assert isinstance(llm, mal.OllamaAdapter)
        assert llm.describe_config().model_name == "qwen3:8b"
        assert llm.get_model_capabilities()["vision"] is False


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

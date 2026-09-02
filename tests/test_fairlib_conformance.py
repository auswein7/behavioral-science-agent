"""Fairlib chat-model conformance over the adapters this fork constructs.

Adoption map item (d): downstream CI runs the upstream conformance suite
(fairlib.conformance, fifteen checks at PR #175) against exactly the fairlib
adapters src.backends builds for the captioning and screenplay stages, so a
fair_llm change that breaks the contract the fork relies on - image carriage,
the non-vision refusal, usage on replies, one ModelInvocationEvent per call -
fails here before a run does. Self-skips without fair-llm installed
(workspace test rule: skipif, never an ignore list); the CI job installs it
only when the FAIR_LLM_TOKEN secret is set.

No network: the fake transport is the same shape fairlib's own Ollama
conformance fixture uses (a recorded httpx client on the adapter), so a
report failure is an adapter-contract failure, never a daemon state.
"""

import asyncio
import contextlib
import json
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

conformance = pytest.importorskip(
    "fairlib.conformance", reason="fair-llm not installed; conformance guard skipped"
)
httpx = pytest.importorskip("httpx", reason="httpx (a fair-llm dependency) missing")

from fairlib.core.message import Message  # noqa: E402

from src.backends import (  # noqa: E402
    fairlib_caption_adapter,
    fairlib_screenplay_adapter,
)

_REPLY = "ready."
_CHUNK_TEXTS = (_REPLY[:3], _REPLY[3:])
_MODEL = "conformance-fixture"


class _FakeStreamResponse:
    """The streaming half of the fake Ollama transport."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for text in _CHUNK_TEXTS:
            yield json.dumps({"message": {"content": text}})
        yield json.dumps(
            {
                "message": {"content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 9,
                "eval_count": 1,
                "total_duration": 3_000_000,
            }
        )


def _fake_transport(adapter: object) -> MagicMock:
    """Install a recording fake httpx client on a fairlib OllamaAdapter."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "model": _MODEL,
        "message": {"role": "assistant", "content": _REPLY},
        "prompt_eval_count": 9,
        "eval_count": 1,
        "total_duration": 3_000_000,
        "done_reason": "stop",
    }
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.stream = MagicMock(side_effect=lambda *args, **kwargs: _FakeStreamResponse())
    adapter.client = client  # type: ignore[attr-defined]
    return client


def _case(build: Callable[[str], object], label: str):
    adapter = build(_MODEL)
    client = _fake_transport(adapter)

    def recorded_request() -> object:
        call = client.post.call_args
        return call.kwargs.get("json") if call else None

    @contextlib.contextmanager
    def failure_mode():
        healthy = client.post
        client.post = AsyncMock(side_effect=httpx.ConnectError("transport down"))
        try:
            yield
        finally:
            client.post = healthy

    return conformance.ChatModelConformanceCase(
        model=adapter,
        recorded_request=recorded_request,
        failure_mode=failure_mode,
        # The same sample for both stages: the vision check must see the
        # image carried by the caption adapter and refused by the
        # screenplay adapter, which is the fork's upstream ruling.
        vision_sample=Message(
            role="user",
            content="what is in this image?",
            images=(b"\xff\xd8\xffconformance-frame",),
        ),
        timeout_seconds=10,
        label=label,
    )


def _run(case) -> object:
    return asyncio.run(
        asyncio.wait_for(conformance.acheck_chat_model_conformance(case), 60)
    )


@pytest.mark.parametrize(
    ("build", "label"),
    [
        (fairlib_caption_adapter, "caption stage adapter (vision)"),
        (fairlib_screenplay_adapter, "screenplay stage adapter (text-only)"),
    ],
    ids=["caption", "screenplay"],
)
def test_stage_adapter_conforms(build, label):
    report = _run(_case(build, label))
    assert report.ok, report.render()


def test_guard_exercises_the_checks_the_fork_depends_on():
    # The fork's seam relies on these named checks; a suite that renamed or
    # dropped one would leave the guard green for the wrong reason.
    relied_on = {
        "vision_image_handling",
        "usage_reporting",
        "model_invocation_event",
        "typed_failure",
        "model_capability_gating",
    }
    missing = relied_on - set(conformance.CHAT_MODEL_CHECKS)
    assert not missing, f"installed fairlib suite lacks {sorted(missing)}"
    report = _run(_case(fairlib_caption_adapter, "caption"))
    observed = {f.check for f in report.findings if f.status.name == "PASSED"}
    assert relied_on <= observed, report.render()

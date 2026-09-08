"""The coder agent: one fairlib SimpleAgent call per utterance, validated.

Adoption map item (b), the flagship fairlib consumer in this repo. The
agent is fairlib's SimpleAgent with a SimpleReActPlanner over an empty
ToolRegistry: the loop, the validator retries and the typed exhaustion
error are the framework's (design principle 7: no hand-rolled retry here).
The validator is the contract the final answer must meet - a JSON object
that parses as UtteranceCoding with exactly the codebook's code set - and
a rejected draft is rewritten by the framework with the validator's
feedback. When fair_llm #174 ships response_model on arun, this validator
becomes the framework's; until then it is the documented fallback path.

The model is a fairlib chat adapter handed in by src.backends; this module
never learns the provider. Input rows are Tier C (already de-identified),
so the stage sends nothing new off-box that the gate has not passed, and
the row text is treated as untrusted data inside the prompt.
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence

from pydantic import ValidationError

from src.adapters import UsageTally
from src.coder.codebook import Codebook
from src.errors import AdapterError, CoderError, ConfigurationError
from src.fairlib_adapter import FairlibUsageSubscriber
from src.prompts import CODER_ROLE_PROMPT
from src.schemas import CodedUtteranceRow, UtteranceCoding, UtteranceRow

logger = logging.getLogger(__name__)

NA = "NA"

# Roles OSU's sample leaves uncoded; these rows never reach a model.
UNCODED_ROLES = frozenset({"confederate", "experimenter"})


def render_coder_role(codebook: Codebook) -> str:
    """Fill CODER_ROLE_PROMPT with the codebook and the exact output shape."""
    lines = [f"- {c.code} ({c.name}): {c.definition}" for c in codebook.codes]
    template = {
        "uid": "<the TARGET utterance's uid, copied exactly>",
        "judgments": {
            c.code: {"value": 0, "rationale": "one sentence"} for c in codebook.codes
        },
    }
    return CODER_ROLE_PROMPT.format(
        codebook="\n".join(lines), output_template=json.dumps(template)
    )


def _json_object(text: str) -> str:
    """The first {...} block in a reply, so a fence or a stray sentence
    around the object is tolerated; the object itself is validated fully."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text.strip()


def parse_coding(text: str, codebook: Codebook, uid: str) -> UtteranceCoding:
    """Parse and validate one reply against the schema and the codebook.

    Raises ValueError with an actionable message on any mismatch; the
    validator turns that message into the framework's rewrite feedback.
    ValueError is the parse layer's local signal here, caught two frames up
    and never crossing a stage boundary."""
    try:
        coding = UtteranceCoding.model_validate_json(_json_object(text))
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "object"
        raise ValueError(f"{first['msg']} at {loc}") from exc
    expected = set(codebook.code_names)
    got = set(coding.judgments)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            f"judgments must have exactly the codebook's codes; missing {missing}, "
            f"unexpected {extra}"
        )
    if coding.uid != uid:
        raise ValueError(f"uid must be {uid!r}, got {coding.uid!r}")
    return coding


def coding_validator(codebook: Codebook, uid: str) -> Callable:
    """The fairlib validator for one utterance: approve a reply that parses
    as this uid's UtteranceCoding over the codebook, else coach the rewrite."""
    from fairlib import Verdict

    async def conforms(response: str):
        try:
            parse_coding(response, codebook, uid)
        except ValueError as exc:
            return Verdict.reject(
                "Your reply must be ONLY a one-line JSON object of the required "
                f"shape. Problem: {exc}."
            )
        return Verdict.approve()

    return conforms


def format_request(row: UtteranceRow, prior: Sequence[UtteranceRow]) -> str:
    """The user turn: prior context then the target, as data, not prose."""
    context = "\n".join(f"[{p.uid}] {p.speaker}: {p.utterance}" for p in prior) or "(none)"
    return (
        f"PRIOR UTTERANCES (context only, do not code):\n{context}\n\n"
        f"TARGET UTTERANCE (code this one):\n"
        f"uid: {row.uid}\nspeaker: {row.speaker}\ntone: {row.tone}\n"
        f"nonverbal_notes: {row.nonverbal_notes or '(none)'}\n"
        f"utterance: {row.utterance}"
    )


class AbstractUtteranceCoder(ABC):
    """The coder stage's contract: one validated coding per utterance."""

    @abstractmethod
    def code_utterance(self, row: UtteranceRow, prior: Sequence[UtteranceRow]) -> UtteranceCoding:
        """Return the coding for row, or raise CoderError / AdapterError."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier recorded on every coded row."""


class FairlibAgentCoder(AbstractUtteranceCoder):
    """AbstractUtteranceCoder over a fairlib SimpleAgent.

    llm is any fairlib AbstractChatModel; the fork's factory builds it with
    its event bus already bound for usage accounting. One fresh agent per
    utterance keeps codings independent (no memory carries across rows)."""

    def __init__(
        self,
        llm: object,
        codebook: Codebook,
        *,
        max_retries: int = 2,
        max_steps: int = 3,
        usage_tally: UsageTally | None = None,
    ) -> None:
        self._llm = llm
        self._codebook = codebook
        self._max_retries = max_retries
        self._max_steps = max_steps
        self._role = render_coder_role(codebook)
        # SimpleAgent binds its model to the agent's bus at construction, so
        # the stage's accounting subscribes to a bus this coder owns and
        # hands to every agent; a bus bound to the adapter beforehand would
        # be displaced and count nothing (found live, 2026-09-02).
        self._events = None
        if usage_tally is not None:
            from fairlib.core.event_bus import AgentEventBus

            self._events = AgentEventBus()
            FairlibUsageSubscriber(usage_tally).subscribe(self._events)

    @property
    def model_name(self) -> str:
        try:
            return str(self._llm.describe_config().model_name)  # type: ignore[attr-defined]
        except Exception as exc:
            raise AdapterError(f"coder model cannot describe itself: {exc}") from exc

    @property
    def framework_provenance(self) -> dict[str, str]:
        """What the framework contributed to this run, for the run record.

        The coder's system prompt is assembled by fairlib's planner, not by
        this repo, so digesting our own CODER_ROLE_PROMPT does not describe
        what the model actually saw. Measured 2026-09-08: two fairlib trees
        both reporting version 0.6.2 rendered different planner prompts and
        changed the codes on 3 of 5 utterances, and nothing in the run record
        distinguished them. The version is kept because it becomes meaningful
        once releases are cut; the digest is the field that discriminates.
        """
        import importlib.metadata as metadata

        try:
            version = metadata.version("fair-llm")
        except metadata.PackageNotFoundError:
            version = "unknown"
        planner, _ = self._planner()
        prompt = planner.render_system_prompt()
        digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=16).hexdigest()
        return {
            "fairlib_version": version,
            "planner_prompt_digest": digest,
            "planner_prompt_chars": str(len(prompt)),
        }

    def _planner(self):
        """The planner exactly as the agent gets it, so a digest taken here
        describes the prompt the coding calls actually used."""
        from fairlib import RoleDefinition, SimpleReActPlanner, ToolRegistry

        registry = ToolRegistry()
        planner = SimpleReActPlanner(self._llm, registry)
        planner.prompt_builder.role_definition = RoleDefinition(self._role)
        return planner, registry

    def _agent(self):
        from fairlib import SimpleAgent, ToolExecutor, WorkingMemory

        planner, registry = self._planner()
        return SimpleAgent(
            llm=self._llm,
            planner=planner,
            tool_executor=ToolExecutor(registry),
            memory=WorkingMemory(),
            max_steps=self._max_steps,
            role_description=self._role,
            events=self._events,
        )

    async def acode_utterance(
        self, row: UtteranceRow, prior: Sequence[UtteranceRow]
    ) -> UtteranceCoding:
        from fairlib import FairlibError, MaxStepsExceeded, ValidatorRejectedError

        try:
            answer = await self._agent().arun(
                format_request(row, prior),
                validator=coding_validator(self._codebook, row.uid),
                max_retries=self._max_retries,
            )
        except ValidatorRejectedError as exc:
            raise CoderError(
                f"no valid coding for {row.uid} after {exc.attempt_count} attempt(s)",
                uid=row.uid,
                attempts=exc.attempt_count,
                last_feedback=exc.last_feedback,
            ) from exc
        except MaxStepsExceeded as exc:
            raise CoderError(
                f"coder agent ran out of steps ({exc.max_steps}) on {row.uid}",
                uid=row.uid,
                attempts=exc.max_steps,
            ) from exc
        except FairlibError as exc:
            raise AdapterError(f"coder model failed on {row.uid}: {exc}") from exc
        # The validator approved this text, so the parse cannot fail here.
        return parse_coding(answer, self._codebook, row.uid)

    def code_utterance(self, row: UtteranceRow, prior: Sequence[UtteranceRow]) -> UtteranceCoding:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acode_utterance(row, prior))
        raise ConfigurationError(
            "code_utterance is the synchronous entry; await acode_utterance "
            "from inside an event loop"
        )


def iter_coded_rows(
    rows: Sequence[UtteranceRow],
    coder: AbstractUtteranceCoder,
    codebook: Codebook,
    *,
    coder_id: str,
    prompt_version: str,
    context_utterances: int = 5,
    already_coded: frozenset[str] = frozenset(),
) -> Iterator[CodedUtteranceRow]:
    """Code rows in order, yielding each finished row so the caller can
    persist incrementally. Uncoded roles get NA cells without a model call;
    uids in already_coded are skipped (resumption). A CoderError propagates
    with the uid on it; rows before it have already been yielded."""
    code_names = codebook.code_names
    model_name = coder.model_name
    for index, row in enumerate(rows):
        if row.uid in already_coded:
            continue
        base = row.model_dump()
        if row.role in UNCODED_ROLES:
            yield CodedUtteranceRow(
                **base,
                codes=dict.fromkeys(code_names, NA),
                rationales={},
                coder_id=coder_id,
                coder_model=model_name,
                coder_prompt_version=prompt_version,
                codebook_version=codebook.version,
            )
            continue
        prior = rows[max(0, index - context_utterances) : index]
        coding = coder.code_utterance(row, prior)
        yield CodedUtteranceRow(
            **base,
            codes={c: coding.judgments[c].value for c in code_names},
            rationales={c: coding.judgments[c].rationale for c in code_names},
            coder_id=coder_id,
            coder_model=model_name,
            coder_prompt_version=prompt_version,
            codebook_version=codebook.version,
        )

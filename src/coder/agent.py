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
never learns the provider. Sampling and the output budget travel with every
call of a run through SimpleAgent.arun(generation_options=) in fairlib's
provider-neutral vocabulary (fair_llm #186), so one CoderConfig reaches any
backend the same way. The run's fairlib security manager (src.egress, ADR
0001) rides on the agent's ToolExecutor, so fairlib checks the model grant
on every call. Input rows are Tier C (already de-identified), and the row
text is treated as untrusted data inside the prompt.
"""

import asyncio
import hashlib
import inspect
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence

from pydantic import ValidationError

from src.adapters import UsageTally
from src.coder.codebook import Codebook
from src.errors import (
    AdapterError,
    CoderBudgetError,
    CoderError,
    ConfigurationError,
    GateBlockedError,
)
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


def coder_generation_options(
    *, temperature: float, seed: int | None, max_tokens: int | None
) -> dict[str, object]:
    """The coder's sampling and output budget in fairlib's provider-neutral
    vocabulary. An unset lever is left out, so the backend's own default
    applies rather than an explicit None."""
    options: dict[str, object] = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    return options


def _require_carried_options(llm: object, options: Mapping[str, object]) -> None:
    """Refuse, before the first row, a run option the bound model does not
    carry. fairlib refuses the same name at call time; checking here turns a
    failure on row one into a wiring error at startup that names the lever.
    A model that declares nothing is left alone, the rule fairlib applies.

    Seed is the case that bites: some backends declare it and some do not,
    so determinism is a property of the backend. Refusing is the current
    rule - a silently dropped seed would let the run record claim a
    determinism the backend never offered."""
    if not options:
        return
    from fairlib import SimpleAgent

    if "generation_options" not in inspect.signature(SimpleAgent.arun).parameters:
        raise ConfigurationError(
            "the installed fair-llm predates SimpleAgent.arun(generation_options=) "
            "(fair_llm #186); install the version pinned in requirements.txt"
        )
    declared = llm.get_model_capabilities().get("generation_options")  # type: ignore[attr-defined]
    if declared is None:
        return
    undeclared = sorted(set(options) - set(declared))
    if undeclared:
        raise ConfigurationError(
            f"the coder model does not carry generation option(s) {undeclared} "
            f"(it declares {sorted(declared)}); unset the matching lever "
            "(SAMPLING_SEED, SAMPLING_TEMPERATURE, CODER_MAX_TOKENS) or bind a "
            "backend that carries it"
        )


class _TruncationWatch:
    """Records whether any model call behind the current coding stopped on
    the output budget. It reads the typed done_reason off fairlib's
    ModelInvocationEvent, so truncation is observed on the bus, never
    inferred from reply text (design principle 4). Planner turns and
    validator rewrites both emit the event, so both are covered."""

    def __init__(self) -> None:
        self.truncated = False
        self._length = None

    def subscribe(self, bus: object) -> None:
        from fairlib.core.events import ModelInvocationEvent
        from fairlib.core.message import DoneReason

        self._length = DoneReason.LENGTH
        bus.subscribe(ModelInvocationEvent, self._on_invocation)  # type: ignore[attr-defined]

    def _on_invocation(self, event: object) -> None:
        usage = event.usage  # type: ignore[attr-defined]
        if usage is not None and usage.done_reason is self._length:
            self.truncated = True


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

    llm is any fairlib AbstractChatModel. generation_options is the run's
    sampling and output budget (coder_generation_options), sent with every
    model call; a name the model does not declare is refused here.
    security_manager is the run's fairlib security manager
    (src.egress.security_manager_for); SimpleAgent binds its executor's
    manager for each run, so passing it here is what puts fairlib's model
    grant on every call. One fresh agent per utterance keeps codings
    independent (no memory carries across rows), and codings run one at a
    time because the truncation watch is reset per coding."""

    def __init__(
        self,
        llm: object,
        codebook: Codebook,
        *,
        max_retries: int = 2,
        max_steps: int = 3,
        usage_tally: UsageTally | None = None,
        generation_options: Mapping[str, object] | None = None,
        security_manager: object | None = None,
    ) -> None:
        from fairlib.core.event_bus import AgentEventBus

        self._llm = llm
        self._codebook = codebook
        self._max_retries = max_retries
        self._max_steps = max_steps
        self._role = render_coder_role(codebook)
        self._options = dict(generation_options or {})
        self._security = security_manager
        _require_carried_options(llm, self._options)
        # SimpleAgent binds its model to the agent's bus at construction, so
        # the stage's subscribers attach to a bus this coder owns and hands
        # to every agent; a bus bound to the adapter beforehand would be
        # displaced and see nothing (found live, 2026-09-02). The bus exists
        # even without a tally because the truncation watch needs it.
        self._events = AgentEventBus()
        self._watch = _TruncationWatch()
        self._watch.subscribe(self._events)
        if usage_tally is not None:
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
            tool_executor=ToolExecutor(registry, security_manager=self._security),
            memory=WorkingMemory(),
            max_steps=self._max_steps,
            role_description=self._role,
            events=self._events,
        )

    def _raise_if_truncated(self, uid: str, attempts: int, cause: Exception) -> None:
        """When a call behind this failure stopped on the output budget,
        report the budget. The validator's or the step limit's message would
        point at the prompt or the codebook and send the operator to debug
        the wrong thing (design principle 6: a degraded result is reported
        as its own kind)."""
        if not self._watch.truncated:
            return
        budget = self._options.get("max_tokens")
        max_tokens = budget if isinstance(budget, int) else None
        raise CoderBudgetError(
            f"coder output for {uid} was cut off by the output budget "
            f"(max_tokens={max_tokens if max_tokens is not None else 'backend default'}) "
            "and the coding failed after it; the cause is the budget, not the "
            "prompt or codebook. Raise CODER_MAX_TOKENS (and CODER_NUM_CTX on a "
            "backend whose context window also bounds the reply).",
            uid=uid,
            attempts=attempts,
            max_tokens=max_tokens,
        ) from cause

    async def acode_utterance(
        self, row: UtteranceRow, prior: Sequence[UtteranceRow]
    ) -> UtteranceCoding:
        from fairlib import (
            FairlibError,
            MaxStepsExceeded,
            PlannerParseError,
            ValidatorRejectedError,
        )
        from fairlib.core.errors import CapabilityDeniedError

        self._watch.truncated = False
        run_options = {"generation_options": self._options} if self._options else {}
        try:
            answer = await self._agent().arun(
                format_request(row, prior),
                validator=coding_validator(self._codebook, row.uid),
                max_retries=self._max_retries,
                **run_options,
            )
        except CapabilityDeniedError as exc:
            # fairlib's grant is the backstop of the egress gate (ADR 0001):
            # a denial is terminal, never retried or downgraded.
            raise GateBlockedError(
                f"fairlib denied the coder model on {row.uid}: {exc}"
            ) from exc
        except ValidatorRejectedError as exc:
            self._raise_if_truncated(row.uid, exc.attempt_count, exc)
            raise CoderError(
                f"no valid coding for {row.uid} after {exc.attempt_count} attempt(s)",
                uid=row.uid,
                attempts=exc.attempt_count,
                last_feedback=exc.last_feedback,
            ) from exc
        except MaxStepsExceeded as exc:
            self._raise_if_truncated(row.uid, exc.max_steps, exc)
            raise CoderError(
                f"coder agent ran out of steps ({exc.max_steps}) on {row.uid}",
                uid=row.uid,
                attempts=exc.max_steps,
            ) from exc
        except PlannerParseError as exc:
            self._raise_if_truncated(row.uid, 1, exc)
            raise AdapterError(f"coder model failed on {row.uid}: {exc}") from exc
        except FairlibError as exc:
            raise AdapterError(f"coder model failed on {row.uid}: {exc}") from exc
        if self._watch.truncated:
            logger.warning(
                "%s: a model call was cut off by the output budget; a later "
                "attempt produced a valid coding",
                row.uid,
            )
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

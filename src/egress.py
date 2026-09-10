"""The egress gate for model calls (ADR 0001).

Two checks compose and a remote model call must clear both. Ours runs
first, where the coder's input is assembled: decide_egress answers "may
these rows leave the box, to this destination" from the session's source
class and its scrub report, and refuses with GateBlockedError otherwise.
fairlib's runs on every call: the capability bag built from the decision
grants exactly the model it authorized, so a model the decision did not
cover is denied by the framework even if our code had a bug.

The destination list lives here, in code, and changes only by a PR that
amends ADR 0001 (ruling 2); no environment lever widens it. Rulings 1 and 3
are the two conditions below: only a public_domain session, and only with a
scrub report whose resolution is "clean" for that same session.
"""

from datetime import UTC, datetime
from types import MappingProxyType

from src.errors import GateBlockedError
from src.schemas import (
    ScrubReport,
    ScrubResolution,
    SessionManifest,
    SourceClass,
    _Deliverable,
)

# Providers served on this machine: no egress, no gate.
LOCAL_PROVIDERS = frozenset({"ollama"})

# Remote providers allowed as destinations, with the one host each may
# reach. Public Gemini's endpoint is fixed in fairlib's adapter and cannot be
# moved at construction or per call (fair_llm #186), which is what makes this
# entry a statement about where bytes go.
REMOTE_DESTINATIONS = MappingProxyType(
    {"gemini": ("generativelanguage.googleapis.com", 443)}
)

EGRESS_SOURCE_CLASSES = frozenset({"public_domain"})


class EgressDecision(_Deliverable):
    """What the gate decided for one coder run; recorded in its provenance."""

    provider: str
    model: str
    remote: bool
    destination: str | None = None
    session_id: str
    source_class: SourceClass | None = None
    scrub_artifact: str | None = None
    scrub_resolution: ScrubResolution | None = None
    decided: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def decide_egress(
    provider: str,
    model: str,
    *,
    session_id: str,
    manifest: SessionManifest | None = None,
    scrub_report: ScrubReport | None = None,
) -> EgressDecision:
    """Authorize a coder run's destination, or raise GateBlockedError.

    A local provider needs nothing. A remote one needs every condition of
    ADR 0001: the provider on the destination list, a manifest for this
    session whose source class may leave the box, and a scrub report for
    this session's utterance table with resolution "clean"."""
    if provider in LOCAL_PROVIDERS:
        return EgressDecision(
            provider=provider,
            model=model,
            remote=False,
            session_id=session_id,
            source_class=manifest.source_class if manifest else None,
            decided=_now(),
        )
    if provider not in REMOTE_DESTINATIONS:
        raise GateBlockedError(
            f"provider {provider!r} is not an allowed egress destination "
            f"(ADR 0001 ruling 2 allows {sorted(REMOTE_DESTINATIONS)})"
        )
    if manifest is None:
        raise GateBlockedError(
            f"remote coding of {session_id} needs its session manifest, to read the source class"
        )
    if manifest.session_id != session_id:
        raise GateBlockedError(
            f"the manifest is for session {manifest.session_id!r}, not {session_id!r}"
        )
    if manifest.source_class not in EGRESS_SOURCE_CLASSES:
        raise GateBlockedError(
            f"session {session_id} is {manifest.source_class}; only public_domain "
            "sessions may be sent to a remote model (ADR 0001 ruling 3)"
        )
    if scrub_report is None:
        raise GateBlockedError(
            f"remote coding of {session_id} needs the scrub report for its utterance table"
        )
    gated = {f"{session_id}.utterances.csv", f"{session_id}.utterances.json"}
    if scrub_report.session_id != session_id or scrub_report.artifact not in gated:
        raise GateBlockedError(
            f"the scrub report covers {scrub_report.session_id}/{scrub_report.artifact}, "
            f"not {session_id}'s utterance table"
        )
    if scrub_report.resolution != "clean":
        raise GateBlockedError(
            f"the scrub report for {session_id} is {scrub_report.resolution!r}; a remote "
            "destination requires 'clean' (ADR 0001 ruling 1: review clearance does "
            "not authorize egress)"
        )
    host, port = REMOTE_DESTINATIONS[provider]
    return EgressDecision(
        provider=provider,
        model=model,
        remote=True,
        destination=f"{host}:{port}",
        session_id=session_id,
        source_class=manifest.source_class,
        scrub_artifact=scrub_report.artifact,
        scrub_resolution=scrub_report.resolution,
        decided=_now(),
    )


def capability_bag_for(decision: EgressDecision) -> object:
    """fairlib's grant for a decided run: the one model, and the one host
    when the destination is remote. Everything else is denied by default."""
    from fairlib.core.capability_bag import CapabilityBag

    hosts = [decision.destination] if decision.destination else []
    return CapabilityBag.from_grants(models=[decision.model], hosts=hosts)


def security_manager_for(decision: EgressDecision) -> object:
    """The fairlib security manager carrying capability_bag_for(decision).
    The coder hands it to the ToolExecutor its agent is built with, because
    SimpleAgent binds that executor's manager for every run."""
    from fairlib.modules.security.basic_security_manager import BasicSecurityManager

    return BasicSecurityManager(capability_bag=capability_bag_for(decision))

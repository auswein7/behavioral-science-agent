# ADR 0001: Where the PII/egress boundary is enforced

Status: ACCEPTED 2026-09-10 (Austin delegated the open questions to Claude's
judgment; rulings below).
Date: 2026-09-08
Context: fairlib adoption map item (e); unblocks the coder's remote providers.
Related: fair_llm #167 (CapabilityBag / BasicSecurityManager), #173
(content-classification egress gate), #148 (Gemini chat adapter, public
Gemini API), #189 (GenAI.mil adapter, deferred and blocked).

## Context

The coder stage refuses every remote provider today. `build_coder_llm` raises
`ConfigurationError` for anthropic / openai / gemini, naming this ADR as the
gate. That refusal was deliberate scaffolding: it is cheaper to refuse than to
ship an egress path nobody has designed. fair_llm #148 removes the upstream
half of the block, so the refusal is now the only thing standing between a
screenplay and a frontier model, and it has to be replaced by a design rather
than deleted.

Two mechanisms exist and they are easy to conflate.

Ours (`src/scrub/gate.py`) knows what a leak IS. `evaluate_gate` scans a
deliverable, produces Tier B findings with excerpts and a Tier C `ScrubReport`
with counts, and `require_pass` raises `GateBlockedError`. Only this side holds
the domain knowledge: the leak taxonomy, the tier rules, the human-clearance
path (`--cleared-by`). fairlib cannot judge whether "light blue shirt" is a
re-identification risk in a cadet study.

Fairlib's knows where bytes may GO. `CapabilityBag` / `BasicSecurityManager`
grant or deny a capability; denial raises `CapabilityDeniedError`. Construction
is offline - the adapter builds, describes itself and reports capabilities
without touching the wire, raising a typed error at construction when the
credential is missing - so we can gate on capability and `describe_config`
before any call.

Scope correction (Austin's ruling, 2026-09-08): the Gemini adapter is
PUBLIC-GEMINI ONLY. An earlier draft of this ADR described a declared egress
host derived from each adapter's `base_url`; that hook, and the base-URL egress
branch, are being REMOVED from the #148 branch. GenAI.mil moves to #189, which
is blocked until the lab supplies the endpoint, auth mechanism, API dialect and
a test credential - and its eventual shape may not be a Gemini adapter at all
(a gateway option on GeminiAdapter only if the gateway speaks the Gemini API, a
base URL on OpenAIAdapter if it is OpenAI-compatible, a new adapter for a
dialect of its own). So this ADR must NOT be designed against `base_url`. For
now the only Gemini destination fairlib declares is the fixed public endpoint
`generativelanguage.googleapis.com:443`, and the destination side of the gate
is correspondingly simpler: one known host, not a configurable one.

Second leg of the same guarantee (fairlib PR #186 round 1, 2026-09-08): a call
may pass GENERATION OPTIONS ONLY. Every other request-config field, the SDK's
`http_options` among them, is refused with a typed error before any call. This
matters to this ADR more than it looks: a fixed declared destination is only
worth having if a caller cannot move the destination at call time, and
`http_options` could have redirected the request, API key included, to another
host. Without that refusal the capability grant would have named a host the
call was free to ignore. The destination is now unmovable from both directions
- not configurable at construction, not overridable per call - which is what
makes "the grant names where bytes may go" a true statement rather than a
convention.

The failure mode this ADR exists to prevent is the one design principle 3
names: "there is no path where a local-only stage can be pointed at a remote
endpoint by changing one string."

## Decision

The two mechanisms COMPOSE. They are not alternatives and neither is a fallback
for the other. A remote call must clear both, and each answers a different
question.

1. Capability grant (fairlib, coarse, default deny) answers MAY BYTES MOVE AT
   ALL, FROM THIS STAGE, TO THIS HOST. Every stage runs under a capability bag.
   The captioning and transcription stages hold no network-egress grant for any
   remote host, permanently, so pointing `CAPTION_BACKEND` at a remote adapter
   fails at construction no matter what the config string says. That is the
   architectural form of the local-only rule: it is a missing grant, not a
   validation branch someone can edit.

2. Content classification (ours, fine, content-aware) answers MAY THIS
   PARTICULAR TEXT MOVE. No artifact may be placed in a message bound for a
   remotely-granted adapter unless it carries a Tier C classification from our
   own gate - `resolution` of "clean" or "cleared_by_review", recorded with the
   run. (Narrowed by ruling 1 below: egress requires "clean".)

Ordering and authority: our classification runs FIRST, because it happens where
the coder's input is assembled, before any adapter is constructed. Fairlib's
grant is the BACKSTOP - the check that still holds if our classifier has a bug,
which is exactly why it must not be reachable from our code path as something
to catch and continue. A denial from either side is terminal. Neither is
downgraded to a warning, and neither is caught and retried.

On #173 specifically: labels on `Message` enforced at the adapter boundary are
the right CARRIER for our classification, and we would adopt them. The
CLASSIFIER stays ours. fairlib should enforce a label it is handed; it should
not attempt to decide what a leak is. If #173 is built so the framework assigns
labels, we do not adopt it and we keep our own pre-adapter check.

## Consequences

- `build_coder_llm` stops refusing remote providers by name. It refuses on a
  missing grant instead, which is the check that generalizes to providers
  nobody has written yet.
- The coder's input assembly gains a typed precondition: a `CodedUtteranceRow`
  batch bound for a remote model must reference a passing `ScrubReport`.
  Violation raises `GateBlockedError`, the error that already means this.
- `RunProvenance` records BOTH decisions, to the extent each is observable. A
  remote coder run whose provenance carries no record of the egress side is a
  defect - the double gate is only worth having if a silently no-opped half is
  visible. Caveat confirmed 2026-09-08: fairlib does not today record "this
  adapter was constructed under grant X" anywhere; the bag is bound per dispatch
  or per run through a context variable and is not attached to the adapter. So
  our provenance records our own classification decision plus any denial we
  caught, and a framework-side constructed-under-grant record is a second
  #173 ask.
- A denial reaches us as a RAISED TYPED ERROR ONLY. Confirmed with
  fairlib-agent 2026-09-08: fairlib does not emit `CapabilityDeniedEvent` or
  `NetworkEgressDeniedEvent` on the LLM deny path today. The security manager
  raises `CapabilityDeniedError` from `check_model_capability` and
  `check_network_egress`; the typed events come only from the tool executor and
  from SimpleAgent around its own denials. So we record the denial ourselves
  where we catch it. We do NOT write a subscription that would silently never
  fire - a listener for an event nobody emits is precisely the shadow state
  principle 6 forbids, and it would make the double gate look instrumented when
  it is not. Framework-emitted events at the adapter seam are a REQUIREMENT we
  ask fair_llm #173 to carry, not an assumption this ADR builds on.
  (Update 2026-09-10: inside a SimpleAgent run, SimpleAgent does emit
  `CapabilityDeniedEvent` around a denied planner call and validator rewrite
  before re-raising; only a direct `llm.invoke()` outside SimpleAgent is
  unaudited. The coder always runs through `SimpleAgent.arun`.)
- The captioner's permanent local-only status becomes expressible and testable:
  a test asserts the captioning stage's bag denies every remote host.

## What this ADR asks of fair_llm

Neither is a blocker for the decision above; both are needed before the egress
half of the gate is genuinely observable rather than merely enforced.

1. Emit the typed denial event at the adapter seam before raising, so a denied
   model call or host reaches a subscriber and not only an exception handler.
2. Record what grant an adapter was constructed under, so provenance can state
   the egress authorization rather than infer it from the absence of an error.

## Alternatives rejected

Ours alone. Rejected: it leaves the "one config string" hole open. Our gate
checks content, not destination; nothing in it prevents a remote adapter being
constructed for the captioning stage.

Fairlib's alone. Rejected: fairlib cannot classify our content, and a grant that
says "the coder may reach Gemini" says nothing about whether the text in hand is
de-identified. It would authorize sending Tier A material to an allowed host.

One merged gate. Rejected: it couples our leak taxonomy to fairlib's release
cycle and puts domain knowledge in a framework that should not carry it.

## Open questions (as posed 2026-09-08)

1. Does the human-clearance path (`--cleared-by`) authorize EGRESS, or only
   local delivery? A screenplay cleared by a reviewer is Tier C by our rules,
   but "a human said it is fine" is a different assurance from "the gate found
   nothing" when the destination is off-box. Recommend: egress requires
   resolution "clean"; "cleared_by_review" requires a named second approver.
2. NARROWED by the 2026-09-08 ruling. GenAI.mil is out of scope until #189, so
   the near-term question is only whether public Gemini (fixed endpoint) and
   Anthropic are allowed destinations for PUBLIC-DOMAIN-CORPUS work, and who
   owns that list. Note this decides nothing about cadet data: neither service
   is cleared for PII, and per the framing chain GenAI.mil is not either, so no
   frontier destination ever receives Tier A or B material regardless of how
   this is answered.
3. NOW THE LOAD-BEARING ONE, given (2). Is the public-domain corpus exempt (it
   carries no cadet PII), and is the exemption a different capability bag or a
   manifest field? Recommend: a different bag, so the exemption is a grant and
   not a boolean someone flips. With GenAI.mil deferred, public-domain corpus
   work is the ONLY near-term egress case, so this ruling is what actually
   unblocks the coder's remote providers.

## Rulings (2026-09-10)

Austin delegated these to Claude's best judgment. Each chooses the more
conservative option where the two differ.

1. Human clearance does NOT authorize egress. A remote destination requires a
   `ScrubReport` with resolution "clean" for the exact session whose rows are
   sent. "cleared_by_review" keeps its meaning for local delivery. The
   second-approver path is not built; if egress of a reviewer-cleared artifact
   is ever needed it comes back here as an amendment naming who approves.

2. The remote destination list is exactly one entry: public Gemini, fixed
   endpoint `generativelanguage.googleapis.com:443`. It is the model OSU's
   baseline used (gemini-3.5-flash), and the only frontier adapter verified on
   the wire. Anthropic and OpenAI stay refused until a use case and a live
   verification exist. The list lives in code (`src/egress.py`), never in
   `.env` or any config file, and it changes only by a PR that amends this ADR
   - so no lever widens it.

3. The exemption follows the source class, and only `public_domain` may leave
   the box. `cadet_pii` and `osu_study` never do, whatever the gate says,
   because neither frontier service is cleared for human-subjects data. The
   fairlib side is a capability bag built in code from the egress decision: a
   remote model is granted only when our decision authorized it, so a manifest
   edit alone cannot open egress - it would also need a clean gate report for
   that session, and the destination would still have to be on the list.

Implementation notes. SimpleAgent binds its tool executor's security manager
for each run and so replaces any binding made outside it; the coder therefore
passes a `BasicSecurityManager` carrying the bag to the `ToolExecutor` it
builds the agent with, which puts fairlib's model check on every call of the
run. The coder provenance gains an `egress` block recording the decision
(destination, source class, the scrub report's session and resolution). The
captioning-stage bag (Decision 1's permanent local-only test) is a follow-up;
the captioner has no remote backend to point at today.

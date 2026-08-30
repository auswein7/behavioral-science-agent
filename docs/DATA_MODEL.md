# Data model

Schema version: 0.1 (draft, 2026-08-30; cross-checked against OSU's
`example_coded.xlsx` same day). Status: agreed shapes for the anonymizer
deliverables; the coded-row schema (section 4.5) is a placeholder pending the
OSU codebook. This file is the canonical spec; the Pydantic models
under `src/` must match it, and every persisted artifact carries a
`schema_version` field naming the version it was written under.

Conventions used throughout:

- Timestamps are seconds from video start as floats, paired where useful with a
  `HH:MM:SS.mmm` string.
- All JSON artifacts are UTF-8, one file per session per stage.
- Field tables list `name : type` with `?` marking optional/nullable fields.

## 1. Classification tiers

Every artifact in the pipeline belongs to exactly one tier. The tier decides
where it may live and whether it may leave the machine.

| Tier | Meaning | Examples | May leave the box |
|---|---|---|---|
| A | Raw identifiable recordings | video, decoded audio | never |
| B | PII-derived intermediates | transcripts, tone records, captions, merged timeline, internal scrub findings | never |
| C | De-identified deliverables | utterance table, screenplay markdown, run provenance, delivered scrub report, coded rows | yes, after the gate |

Gate rule: only Tier C artifacts cross the boundary, and only after the scrub
gate has recorded a pass (or a logged human override) for that specific
artifact. Tier A and B artifacts stay under `data/` on the processing host for
their whole life; their retention and deletion schedule is the data-management
plan's concern, not this spec's.

Public-domain footage is exempt from the boundary but flows through the same
tiers and gate so that the pipeline exercised in development is the pipeline
used on cadet data.

## 2. Inputs

### 2.1 Video file (Tier A)

Constraints, not a schema:

- Decodable by PyAV (H.264/H.265 in MP4/MKV are the tested paths).
- Exactly one continuous session per file; no editing or scene cuts.
- An audio track must be present; it is downmixed to mono 16 kHz internally.
- Resolution floor 720p; gaze and gesture reading degrades below that.
- The camera should hold a fixed wide shot in which every participant is
  visible; speaker-to-body mapping assumes nobody enters or leaves mid-session.
- The raw filename is treated as potentially identifying (dates, names). It
  never appears in a Tier C artifact; deliverable names derive from
  `session_id` only.

### 2.2 SessionManifest (Tier C metadata, no PII permitted in it)

One JSON per session, written by a human before the run. This object is new;
run 1 had no equivalent and its absence is why diarization guessed the speaker
count and why confederates cannot currently be marked.

| field | type | notes |
|---|---|---|
| schema_version | str | |
| session_id | str | opaque id, e.g. `s017`; no dates or names |
| source_class | str | `cadet_pii` or `public_domain`; drives gate behavior |
| task_name | str | e.g. `lost_on_the_moon` |
| group_size | int | people on camera, including confederates |
| expected_speaker_count | int | feeds diarization (`WHISPERX_NUM_SPEAKERS`) |
| speaker_roles | dict[str, str]? | `SPEAKER_NN` -> `participant` / `confederate` / `experimenter` |
| protocol_ref | str? | IRB protocol or consent reference, cadet sessions only |
| notes | str? | free text; must itself be PII-free |

`speaker_roles` is two-phase by nature: the manifest is created with it empty,
diarization runs, a human listens to enough audio to identify which
`SPEAKER_NN` is the confederate/experimenter, fills the map in, and only then
can the utterance table be finalized (confederate rows are coded `NA`
downstream, matching OSU practice).

### 2.3 RunConfig (Tier C, embedded in provenance)

Typed configuration for one run; the source of truth the `.env` file feeds.
Nothing outside the config loader reads the environment. Every run writes its
resolved config next to its outputs (inside RunProvenance, section 4.3).

Groups and fields:

- transcription: `whisper_model`, `device`, `compute_type`, `batch_size`,
  `language`, `num_speakers` (from the manifest when set)
- tone: `tone_model`
- captioning: `caption_model`, `fps`, `context_captions`, `max_dimension`,
  `speech_frame_offset`, `burst_frames`, `burst_spacing`
- screenplay: `ornith_model`, `num_ctx`
- sampling: `temperature` (default 0), `seed`
- prompts: `prompt_versions` - the version id of each prompt asset used

## 3. Intermediates (Tier B, internal only)

These are the shapes the pipeline already writes, kept as-is and frozen by
this spec plus a `schema_version`. They exist so a run can resume and be
audited; they are never deliverables.

### 3.1 TranscriptRecord (`data/transcripts/<id>.formatted.json`)

| field | type | PII note |
|---|---|---|
| index | int | |
| start, end | float | |
| start_hms, end_hms | str | |
| speaker | str | `SPEAKER_NN` |
| text | str | verbatim speech: names, mishearings-as-names |
| word_count | int | |
| avg_word_score | float | |
| low_confidence_words | list[str] | verbatim fragments |

### 3.2 ToneRecord (`data/transcripts/<id>.formatted.tone.json`)

TranscriptRecord plus:

| field | type | notes |
|---|---|---|
| emotion | str? | one of section 5.3's labels, or null |
| emotion_scores | dict[str, float]? | per-label confidence; never delivered |

### 3.3 CaptionRecord (`data/captions/<id>.captions.json`)

| field | type | PII note |
|---|---|---|
| index | int | |
| timestamp | float | |
| timestamp_hms | str | |
| caption | str | model prose; appearance leakage occurs here |
| trigger | str | `fixed_interval` or `speech_start` |
| mentioned_speakers | list[str] | `SPEAKER_NN` tags found in the caption |
| speech_start | float? | speech-triggered records only |
| burst_captions | list? | debug trail, speech-triggered only |

### 3.4 MergedEvent (`data/screenplays/<id>.screenplay.json`)

Chronological union of 3.2 and 3.3 with `type: "speech" | "visual"`; speech
events carry `timestamp, start, end, start_hms, end_hms, speaker, text,
emotion`, visual events carry the CaptionRecord fields. This is Ornith's input
and stays internal.

## 4. Deliverables (Tier C)

### 4.1 UtteranceRow - the primary deliverable

One row per diarized utterance, emitted as CSV (and JSON). Column-compatible
with the OSU coded-transcript schema; OSU's own columns come first, our added
channels after. See section 6 for the mapping.

| field | type | notes |
|---|---|---|
| document | str | = `session_id` |
| uid | str | `u<ord>` zero-padded to 3 digits, e.g. `u042` (OSU convention; widen to 4 digits past `u999`); unique within `document` only, joins use (`document`, `uid`) |
| ord | int | 1-based utterance order within the session |
| speaker | str | `SPEAKER_NN`; never demographic labels |
| utterance | str | scrubbed dialogue text (spoken names replaced) |
| time | str | utterance start, `HH:MM:SS.mmm`, delivered as text (see note) |
| end_time | str | utterance end, `HH:MM:SS.mmm`, delivered as text |
| filename | str | the artifact's own filename, `<session_id>.utterances.csv` |
| prior_utterance | str | previous row's `utterance`; literal `NA` for row 1 (OSU convention) |
| prior_speaker | str | previous row's `speaker`; literal `NA` for row 1 |
| role | str | from the manifest's `speaker_roles`; default `participant` |
| tone | str | emotion label only; `unknown` for null/`<unk>` |
| low_confidence | bool | true when transcription confidence was poor |
| nonverbal_notes | str | scrubbed visual description, sourcing rule below |

Time-as-text note: the `time` values in OSU's sample were damaged by
spreadsheet auto-coercion (stored as Excel day fractions with implausible
magnitudes for the session length, plus `NA` holes). To avoid the same fate,
the CSV always quotes `time`/`end_time` and the delivery note tells recipients
to import them as text.

`nonverbal_notes` sourcing rule (deterministic, no model involved): the
caption from this utterance's own `speech_start` burst, if any, followed by
any `fixed_interval` captions whose timestamp falls in `[start, end)`, joined
in time order. Notes are scrubbed text and pass the same gate as everything
else in the row.

### 4.2 Screenplay markdown (`<session_id>.screenplay.md`)

The human-readable rendering: scene headings with time ranges, action prose,
tone parentheticals, timestamped dialogue per `screenplay_template.md`. It is
derived presentation, not a second source of truth - any analysis consumes
4.1. Delivered only after its own gate pass.

### 4.3 RunProvenance (`<session_id>.provenance.json`)

Everything needed to reproduce or audit the run without the Tier A/B data.

| field | type | notes |
|---|---|---|
| schema_version | str | |
| session_id, run_id | str | |
| started, finished | str | ISO 8601 UTC |
| host | str | processing machine name |
| input_video_sha256 | str | digest of the raw file (digest is not PII) |
| config | RunConfig | resolved, inline |
| model_digests | dict[str, str] | per stage: model tag -> digest/revision |
| prompt_versions | dict[str, str] | prompt asset -> version id |
| software | dict[str, str] | python, ollama, pinned package versions |
| stage_timings | dict[str, float] | wall seconds per stage |

### 4.4 ScrubReport - two forms, one gate

Internal form (Tier B, `data/` only): every finding with category, matched
pattern, artifact, location and verbatim excerpt - what a human reviews to
clear or fix a run.

Delivered form (Tier C, `<session_id>.scrub_report.json`): the excerpts are
deliberately absent, because a report quoting a leak would itself leak.

| field | type | notes |
|---|---|---|
| schema_version | str | |
| session_id, run_id | str | |
| artifact | str | which deliverable this gate decision covers |
| findings_by_category | dict[str, int] | e.g. `{"appearance": 1, "gendered": 0}` |
| resolution | str | `clean` / `cleared_by_review` / `redacted` / `blocked` |
| reviewer | str? | role, not name, when a human cleared it |
| decided | str | ISO 8601 UTC |

A deliverable ships only with `resolution` of `clean`, `cleared_by_review`,
or `redacted`. `blocked` artifacts do not leave, full stop.

### 4.5 CodedUtteranceRow (future - pending the OSU codebook)

UtteranceRow plus, per code in {AO, PO, PE, NE, I, Apology}: a binary value
(or `NA` for confederate/experimenter rows) and a free-text `rationale`;
plus `coder_model` and `coder_prompt_version`. Code columns follow OSU's
observed naming convention `<CODE>_<coder_id>` (their sample uses
`AO_gemini35flash` etc.), so multiple coders can sit side by side in one
table. The code set is deliberately pluggable - it is defined by the codebook
document, not hardcoded here, so it can change without a schema-version bump
to the anonymizer deliverables.

## 5. Identifier and label schemes

### 5.1 Session and uid

`session_id` is an opaque short id assigned at receipt (`s001`, ...), mapped
to real session details only in a Tier B ledger that never leaves the box.
`uid = u<ord>` zero-padded to 3 digits, matching OSU's observed convention
(`u001` ... `u099` in their sample); it is unique within a `document` only,
so cross-session joins always use the (`document`, `uid`) pair. OSU's own
`document` values embed a recording date and session/task numbering
("Clean Transcript S1 T1 01.23.2025"); ours deliberately do not - the
`session_id` is the whole document name, and the date lives only in the
Tier B ledger.

### 5.2 Speakers

`SPEAKER_NN` exactly as diarization assigns them, never renamed, never
demographic. Identity beyond the label lives nowhere in Tier C. The `role`
column is the only sanctioned extra information about a speaker.

### 5.3 Tone labels

Closed set from emotion2vec+: `angry`, `disgusted`, `fearful`, `happy`,
`neutral`, `sad`, `surprised`, `other`, plus `unknown` (mapped from `<unk>`
or a null classification). Deliverables carry the label only.

## 6. Compatibility with the OSU coded-transcript schema

Verified against OSU's sample (`example_coded.xlsx`, 99 rows, one Zoom
session): columns are `document, uid, .ord, speaker, utterance, time,
filename, prior_utterance, prior_speaker`, then six code columns named
`<CODE>_gemini35flash`, values `0`/`1`/`NA`, with all 12 Confederate rows
`NA` and row 1's prior fields `NA`.

| OSU column | Our column | note |
|---|---|---|
| document | document | theirs embeds a date and session/task ids; ours is the opaque `session_id` (deliberate, see 5.1) |
| uid | uid | same format, `u<3-digit ord>` |
| .ord | ord | leading dot dropped (R artifact); value identical |
| speaker | speaker | see difference below |
| utterance | utterance | scrubbed on our side |
| time | time | delivered as quoted text; their sample's values arrived spreadsheet-mangled (see 4.1) |
| filename | filename | theirs is a source-transcript codename; ours names the delivered artifact, never the raw video |
| prior_utterance | prior_utterance | `NA` for row 1, matching their sample |
| prior_speaker | prior_speaker | `NA` for row 1 |
| (codes) | 4.5 | added by the coder stage, not the anonymizer; `<CODE>_<coder_id>` naming kept |

Two further observations from their sample worth knowing: it is "de-identified"
by their standard yet contains a spoken instructor name in an utterance
("Professor ..."), which is exactly the leak channel our deterministic
spoken-name scrub targets and evidence that our bar is intentionally higher;
and it carries text-encoding damage (curly quotes mis-decoded), so our CSVs
are delivered as UTF-8 and the delivery note says so.

Known difference to settle with OSU: their sample labels speakers
demographically (`White24`, `Latina23`, `Confederate`). Our de-identification
rules forbid encoding demographics anywhere, so we ship `SPEAKER_NN` plus the
`role` column instead, and confederate marking moves from the label into
`role`. If their analysis assumes demographic labels exist, that needs an
explicit conversation before P4 - it cannot be accommodated on our side.

## 7. What OSU receives (summary for the data document)

For each recorded session, OSU receives exactly four files, named by opaque
session id, and nothing else:

1. `<session_id>.utterances.csv` - one row per utterance: who spoke (as
   `SPEAKER_NN`), what was said (de-identified), when, vocal tone, a
   transcription-confidence flag, the speaker's role, and de-identified
   non-verbal notes. Column-compatible with the coded-transcript format
   already in use at OSU.
2. `<session_id>.screenplay.md` - the same content rendered as a readable
   screenplay for human review.
3. `<session_id>.provenance.json` - which models, prompts and settings
   produced the artifacts (no content in it).
4. `<session_id>.scrub_report.json` - the de-identification check record:
   finding counts by category and the pass decision for each delivered file.

What OSU never receives: video, audio, raw transcripts, captions, internal
scrub excerpts, or any file whose name or content identifies a participant,
a location, or a recording date. Every delivered file has passed a scrub gate,
and the gate decision is itself part of the delivery (item 4).

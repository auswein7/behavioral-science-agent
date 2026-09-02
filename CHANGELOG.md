# Changelog

Format: Keep a Changelog style, newest first. Versions are git tags; the
maintainer tags releases. Everything before the fork point is Drew Gibson's
prototype history in the upstream repo.

## [Unreleased]

Forked 2026-08-30 from `Andrew-D-Gibson/Screenplay_Video_Anonymizer` (d4af4db).

### Added
- Per-stage usage accounting from fairlib's own `ModelInvocationEvent`
  (fair_llm #170, PR #175): `FairlibUsageSubscriber` binds a bus per stage
  adapter and feeds the `UsageTally` that `RunProvenance.stage_usage`
  records, replacing the reply-side wrapper for the fairlib backend. The
  backend factory wires it; `StageUsage` gains `calls_failed` and `source`
  (`fairlib_events` / `seam_wrapper`) so a run on a pre-#175 fairlib is
  legibly degraded, not silently different.
- Data model spec (`docs/DATA_MODEL.md`) with Tier A/B/C classification, and
  its executable form in `src/schemas.py` (Pydantic models for every stage
  boundary; Tier C deliverables are frozen and reject undeclared fields).
- Primary OSU deliverable: per-utterance table builder
  (`src/deliverables/utterance_table.py` + `src.cli.build_utterance_table`),
  column-compatible with OSU's coded-transcript format.
- De-identification enforcement (`src/scrub/`): deterministic spoken-name
  scrub (spaCy NER, entity-class tokens) run before any downstream model sees
  dialogue, plus an enforcing gate producing Tier B findings and Tier C scrub
  reports and raising `GateBlockedError` on blocked artifacts.
- Typed configuration + startup preflight (`src/config.py`): missing token,
  model, or GPU fails in seconds, not hours in.
- Run provenance (`<name>.provenance.json`): resolved config, model digests,
  prompt content digests, package versions, input sha256, stage timings.
- Pipeline resumability (`main.py --from <stage>`), per-run log files,
  `--manifest` (SessionManifest feeds diarization and names run records).
- Retry primitive for model calls (`src/reliability.py`); captioning-stage
  incremental checkpoints.
- Screenplay faithfulness checker (`src.cli.check_screenplay`): mechanical
  md-vs-events verification (dropped/duplicated/misattributed dialogue,
  invented action attributions, think-text leaks).
- Leak taxonomy draft (`docs/LEAK_TAXONOMY.md`), HF gated-model onboarding
  doc (`docs/SETUP_HF_MODELS.md`), `Modelfile.qwen` (qwen3-vl-16k alias).
- Test suite (75 tests) and CI (lint + tests on push/PR).

### Changed
- All stage record files now carry a `schema_version` envelope
  (`src/persist.py`); legacy bare-list files still load.
- Sampling options (temperature/seed/num_ctx) are actually sent to Ollama and
  match what provenance records; deterministic temperature-0 default.
- Bare `ValueError`/`ConnectionError` signals replaced with the typed
  `PipelineError` hierarchy (`src/errors.py`).

### Fixed
- `_hhmmss` millisecond rounding could render `".1000"`.
- Utterance-table builder crashed on records with no word-level confidence
  data; such rows are now flagged `low_confidence`.
- Drew's committed sample outputs (identifiable content) removed from the
  working tree.

### Removed
- Tracked sample outputs (Attenborough derived files, stale MockTrial
  transcripts); generated `data/` outputs are now gitignored by tier.

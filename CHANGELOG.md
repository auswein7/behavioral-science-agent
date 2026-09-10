# Changelog

Format: Keep a Changelog style, newest first. Versions are git tags; the
maintainer tags releases. Everything before the fork point is Drew Gibson's
prototype history in the upstream repo.

## [Unreleased]

Forked 2026-08-30 from `Andrew-D-Gibson/Screenplay_Video_Anonymizer` (d4af4db).

### Added
- Measured reproducibility in the coder provenance (`src/reproducibility.py`):
  a `reproducibility` block whose status comes from dated measurements, never
  from a declared seed. Measured 2026-09-10: qwen2.5:14b on Ollama is not
  reproducible across server states (same code, prompt, seed 1234 and
  temperature 0 moved 1-2 of 5 code cells); gemini-3.5-flash is not either.
  `CODER_NUM_CTX` now defaults to 8192 for a local coder and is recorded,
  because the context window changes codes and Ollama otherwise picks it.
- The PII/egress gate (ADR 0001, accepted 2026-09-10) and a Gemini coder:
  `src/egress.py` decides each coder run's destination - a remote model only
  for a `public_domain` session with a `clean` scrub report, and only public
  Gemini (fixed endpoint) on the destination list - and builds the fairlib
  capability bag granting exactly the decided model, which the coder's
  `ToolExecutor` carries so fairlib checks it on every call. `CODER_PROVIDER=gemini`
  and `--manifest` / `--scrub-report` on the coder CLI; the decision is the
  provenance's `egress` block. A fairlib denial is a `GateBlockedError`.
  Anthropic and OpenAI stay refused. A live Gemini test self-skips without
  `GEMINI_API_KEY`.
- Evaluation harness (`src/evaluation`, `python -m src.cli.evaluate_coding`):
  per-code precision, recall, F1, agreement and Cohen's kappa between a
  candidate coding and a reference, never pooled accuracy; NA rows excluded
  and one-sided NA counted; zero denominators reported as null. Reads
  `.coded.json`, `.coded.csv` and OSU coded-transcript `.xlsx`. A declared
  held-out split (`split_of`, rule recorded in every report) fixes tune and
  test rows before any prompt tuning.
- OSU transcript import (`python -m src.cli.import_osu_transcript`): an OSU
  coded-transcript workbook becomes 4.1 utterance rows so the coder runs on
  OSU's own sessions (their text-only condition). Demographic speaker labels
  become `SPEAKER_NN`; the label map and generated manifest stay in the
  git-ignored `data/imports/`. New source class `osu_study`, never cleared
  for egress.
- Codebook file lever `CODER_CODEBOOK`: OSU's definitions load from a JSON
  document (`load_codebook`, template `docs/codebook.example.json`) into the
  typed `Codebook`, which now refuses duplicate code names; unset keeps the
  flagged placeholder. A bad file is a `ConfigurationError` at startup.
- Coder run options on fairlib's neutral channel (fair_llm #186):
  temperature, seed and `max_tokens` travel with every model call, planner
  turns and validator rewrites alike, through
  `SimpleAgent.arun(generation_options=)`; `build_coder_llm` and
  `fairlib_coder_adapter` now take only `num_ctx`, the one option with no
  neutral counterpart. The coder refuses an option its model does not
  declare with a `ConfigurationError` at construction, before the first row.
- `CoderBudgetError` (a `CoderError`): a coding that fails after a model call
  stopped on the output budget reports the budget and the `CODER_MAX_TOKENS`
  lever, observed from `ModelInvocationEvent` usage, instead of surfacing as
  a validator or step-limit failure (TODO item b2). The coder's event bus now
  exists unconditionally.
- `fair-llm==0.6.3` from PyPI in `requirements.txt` (0.6.2 lacks #186, #187
  and #191), so one `pip install -r requirements.txt` sets up every stage
  including the coder; the interim private-git pin file and its token-gated
  CI step are gone, and CI now runs the fairlib-dependent tests.
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

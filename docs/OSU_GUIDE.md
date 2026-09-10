# Running the pipeline at OSU: install, run, check, code, evaluate

For a research assistant who has the repository and a GPU machine. Every
command is run from the repository root. Nothing here sends data off the
machine unless a step says so in capital letters.

## 1. What it does

Video in, two things out:

- a de-identified **screenplay** and **utterance table** (who said what, when,
  in what tone, with non-verbal notes), with no names, appearance, gender or
  on-screen text;
- optionally, **behavior codes** (AO, PO, PE, NE, I, Apology) for each
  utterance, from a local model, and an **agreement report** against a
  reference coding (human codes, or another model's).

## 2. Machine

Tested on (2026-09-10): Linux, four 48 GB GPUs, Ollama 0.33.1, Python 3.12.
What the models need, as loaded there:

| model | role | size on disk / loaded |
|---|---|---|
| Ornith-1.5 35B Q8 (`ornith-1.5-255k`) | writes the screenplay | 38 GB; the largest, needs a big GPU or several |
| Qwen3-VL 8B instruct (`qwen3-vl-instruct-16k`) | captions video frames | 6 GB |
| qwen2.5:14b | codes utterances | 9 GB on disk, about 10 GB at the pinned 8192 context |
| WhisperX + pyannote, emotion2vec+ | transcript, speakers, tone | a few GB, GPU optional |

Plan for about 60 GB of disk for the models. The screenplay stage is the
only one that needs the large GPU; transcription and tone run on CPU if they
must (slowly).

## 3. Install

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
```

`requirements.txt` includes `fair-llm` (import name `fairlib`) from PyPI.

Install [Ollama](https://ollama.com/download), start it, then pull the models
and create the aliases the pipeline uses (each Modelfile's `FROM` line is the
model pulled just before it):

```
ollama pull qwen3-vl:8b-instruct
ollama create qwen3-vl-instruct-16k -f Modelfile.qwen-instruct
ollama pull hf.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q8_0
ollama create ornith-1.5-255k -f Modelfile.ornith
ollama pull qwen2.5:14b
```

Copy `.env.example` to `.env`, set `HF_TOKEN`, and accept the pyannote model
terms as `docs/SETUP_HF_MODELS.md` describes (a missed acceptance fails the
first run).

## 4. Run a session

Write a manifest for the session (no names or personal details in it):
`data/manifests/<session_id>.manifest.json` - see `docs/DATA_MODEL.md` 2.2 for
the fields; `source_class` is `cadet_pii` for study recordings.

```
.venv/bin/python main.py <video> --manifest data/manifests/<session_id>.manifest.json
```

A preflight checks the token, models and GPU in seconds before the long work
starts. If the run dies, rerun with `--from <stage>` to resume from the last
finished stage instead of starting over. The run log is
`data/runs/<run_id>.log`; every stage logs `started`, `finished`, `resumed`,
or `DEGRADED` with a reason.

Then build the utterance table (this also runs the de-identification gate on
it):

```
.venv/bin/python -m src.cli.build_utterance_table \
    data/captions/<video>.captions.json \
    data/transcripts/<video>.formatted.tone.json \
    data/manifests/<session_id>.manifest.json
```

## 5. Check before anything is shared

Run this checklist on every session. Do not send a screenplay or table to
anyone until each line is true.

- [ ] The gate reports for the screenplay and the table
      (`data/deliverables/*.scrub_report.json`) say `clean`. A `blocked` report
      means the findings in `data/gate/` need a reviewer; after review, rebuild
      with `--cleared-by <your role>` to record who cleared it.
- [ ] The run log has no `DEGRADED` line you have not looked at. The common
      one: diarization found a different number of speakers than the manifest
      expects - check who is `SPEAKER_00`, `SPEAKER_01`, and so on.
- [ ] Every utterance made it into the screenplay:
      `.venv/bin/python -m src.cli.check_screenplay data/screenplays/<video>.screenplay.md data/screenplays/<video>.screenplay.json`
      exits 0.
- [ ] You read the screenplay once end to end for anything identifying the
      gate cannot know about (a unique event, a place, a role that names a
      person).

## 6. Code the utterances (local)

```
.venv/bin/python -m src.cli.code_utterances data/deliverables/<session_id>.utterances.json
```

Writes `<session_id>.coded.csv` (OSU's column shape, `<CODE>_<coder_id>`),
`.coded.json` (resume point: rerun the same command after an interruption)
and `.coder_provenance.json` (model, codebook, settings, timing, and whether
the backend is known to reproduce its output).

Until the real codebook is loaded the coder uses a placeholder with one-line
definitions and says so in every run. To use OSU's codebook, write it in the
shape of `docs/codebook.example.json` and set `CODER_CODEBOOK=<path>` in
`.env`.

Local coding is not bit-for-bit repeatable: the same model on the same text
can change a code between runs (measured 2026-09-10). Treat one run as one
sample; for any figure you report, code several times and report the spread.

## 7. Compare against a reference

```
.venv/bin/python -m src.cli.evaluate_coding \
    data/deliverables/<session_id>.coded.json <reference.xlsx or .csv> \
    --candidate-id <our coder id> --reference-id <their coder id> [--split tune|test]
```

Prints precision, recall, F1, agreement and Cohen's kappa for each code and
writes a JSON report with counts only. The reference can be an OSU
coded-transcript workbook (`<CODE>_<coder_id>` columns). Tune prompts on
`--split tune` only; score `--split test` once, at the end.

To code one of OSU's own transcripts (text only), import it first:

```
.venv/bin/python -m src.cli.import_osu_transcript <workbook.xlsx> --session-id <id>
```

Speaker labels become `SPEAKER_NN`; the label map stays in `data/imports/`.

## 8. Sending anything off the machine

Study recordings and anything derived from them never leave the machine,
and the software enforces it: the only remote coder is public Gemini, and it
refuses every session except public-domain footage whose gate report is
`clean`. Cadet and OSU-study sessions are refused whatever else is set. The
rules and their reasons are in `docs/adr/0001-egress-gate.md`.

## 9. Help

Questions and problems: the FAIR Lab team, through the repository's issue
tracker.

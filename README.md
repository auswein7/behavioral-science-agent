# FAIR Screenplay Utility

A utility run on local AI that turns video into a "screenplay" — a combined transcript of speech and on-screen action — for use by researchers or other LLM classifiers.

**De-identification is a core design constraint, not an add-on.** The tool is meant to run on video containing PII, and its output must not let a reader re-identify anyone: no appearance, gender, name, or on-screen text/signage — only speaker labels (`SPEAKER_NN`), gaze/orientation, and behavior. Enforcement is layered (see `docs/DATA_MODEL.md` and `docs/LEAK_TAXONOMY.md`): anonymization rules folded into every prompt, a deterministic NER scrub that replaces spoken names before any downstream model sees dialogue, a warn-only regex spot-check mid-pipeline, and an enforcing gate at the end — a deliverable with findings is *blocked* (typed error, findings persisted for human review), never shipped on a warning.

## Pipeline

Four models, orchestrated end-to-end by `main.py`. All four stages are implemented.

- **[WhisperX](https://github.com/m-bain/whisperx)** — diarization scribe. Generates a speaker-labeled speech transcript. Runs via the `whisperx`/PyTorch stack directly, since Ollama has no ASR/diarization capability I've found.
- **emotion2vec+** (`iic/emotion2vec_plus_large` by default) — tone classifier. Classifies each diarized utterance's vocal tone straight from its audio span. Runs via `funasr`, not Ollama, since emotion2vec+ isn't servable there either.
- **[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)** (`qwen3-vl:8b` by default) — visual action observer. Samples video frames at a fixed fps and captions each one, plus a burst of speech-triggered "reaction shot" frames around each utterance start, feeding prior captions back into the prompt so it describes what *changed* rather than re-describing the whole scene. Served locally by Ollama.
- **[Ornith-1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)** (`ornith-1.5-255k` by default) — orchestrator/screenwriter. Takes the merged transcript + caption timeline and writes the final screenplay Markdown, matching the structure in `screenplay_template.md`, rendering tone as parenthetical direction where meaningful. Also served locally by Ollama.

## Setup

```
pip install -r requirements.txt
pip install -r requirements-fairlib.txt   # coder stage and fairlib backends; needs read access to fair_llm
```

`requirements-fairlib.txt` pins fair-llm to a fair_llm commit rather than a PyPI release until a release carries the features the coder needs; the file says which.

Requires a local [Ollama](https://ollama.com/download) install running in the background for the captioning and orchestrator stages — no GPU/PyTorch setup needed for those. WhisperX and emotion2vec+ still need the real PyTorch/`funasr` stack for diarization/ASR/tone classification.

Copy `.env.example` to `.env` and fill in `HF_TOKEN` (needed for the pyannote diarization model used by WhisperX; the gated-model acceptance steps are in `docs/SETUP_HF_MODELS.md`). Model choices, fps, batch size, etc. are also tunable there. The name scrub additionally needs spaCy's small English model: `python -m spacy download en_core_web_sm`.

## Usage

Drop a video in `data/raw_videos/` (or pass a full path) and run the full pipeline:

```
python main.py <video_filename_or_path>
```

A startup preflight validates the config, token, and required models before any expensive work begins. Useful flags: `--from <stage>` resumes from persisted stage outputs after a crash; `--manifest <path>` supplies per-session metadata (expected speaker count, session id); `--run-id` names the run's log and provenance.

This writes intermediate JSON to `data/transcripts/`, `data/captions/`, and `data/screenplays/`, the final screenplay to `data/screenplays/<name>.screenplay.md`, and the run's provenance + de-identification gate records to `data/deliverables/` and `data/gate/`.

The coder stage (OSU behavior codes, `docs/DATA_MODEL.md` 4.5) runs on the delivered utterance table, not the video, through a fairlib `SimpleAgent` over a local Ollama model:

```
python -m src.cli.code_utterances data/deliverables/<session_id>.utterances.json
```

It writes `<session_id>.coded.csv` / `.coded.json` and `<session_id>.coder_provenance.json`; `CODER_MODEL`, `CODER_ID` and the other `CODER_*` levers are documented in `.env.example`. Until OSU's codebook is loaded it codes with a placeholder codebook and says so.

Each stage can also be run independently as a module — see `CLAUDE.md` for the full breakdown of the `src/cli/` entry points (transcribe-only, caption-only, tone-only, merge-only, write-only, utterance-table, faithfulness check).

## Development

```
python -m pytest tests/ -q
ruff check --isolated --select E4,E7,E9,F,I,UP,B,C4,RUF --target-version py311 src tests main.py
```

CI runs both on every push and pull request. Changes are tracked in `CHANGELOG.md`.

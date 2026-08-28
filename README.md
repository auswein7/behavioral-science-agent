# FAIR Screenplay Utility

A utility run on local AI that turns video into a "screenplay" — a combined transcript of speech and on-screen action — for use by researchers or other LLM classifiers.

**De-identification is a core design constraint, not an add-on.** The tool is meant to run on video containing PII, and its output must not let a reader re-identify anyone: no appearance, gender, name, or on-screen text/signage — only speaker labels (`SPEAKER_NN`), gaze/orientation, and behavior. Prompts fold in anonymization rules, and `src/screenplay/scrub_check.py` runs a warn-only regex spot-check after captioning and after the final screenplay write; it never blocks or edits output, so its silence is not a guarantee.

## Pipeline

Four models, orchestrated end-to-end by `main.py`. All four stages are implemented.

- **[WhisperX](https://github.com/m-bain/whisperx)** — diarization scribe. Generates a speaker-labeled speech transcript. Runs via the `whisperx`/PyTorch stack directly, since Ollama has no ASR/diarization capability I've found.
- **emotion2vec+** (`iic/emotion2vec_plus_large` by default) — tone classifier. Classifies each diarized utterance's vocal tone straight from its audio span. Runs via `funasr`, not Ollama, since emotion2vec+ isn't servable there either.
- **[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)** (`qwen3-vl:8b` by default) — visual action observer. Samples video frames at a fixed fps and captions each one, plus a burst of speech-triggered "reaction shot" frames around each utterance start, feeding prior captions back into the prompt so it describes what *changed* rather than re-describing the whole scene. Served locally by Ollama.
- **[Ornith-1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)** (`ornith-1.5-255k` by default) — orchestrator/screenwriter. Takes the merged transcript + caption timeline and writes the final screenplay Markdown, matching the structure in `screenplay_template.md`, rendering tone as parenthetical direction where meaningful. Also served locally by Ollama.

## Setup

```
pip install -r requirements.txt
```

Requires a local [Ollama](https://ollama.com/download) install running in the background for the captioning and orchestrator stages — no GPU/PyTorch setup needed for those. WhisperX and emotion2vec+ still need the real PyTorch/`funasr` stack for diarization/ASR/tone classification.

Copy `.env.example` to `.env` and fill in `HF_TOKEN` (needed for the pyannote diarization model used by WhisperX). Model choices, fps, batch size, etc. are also tunable there.

## Usage

Drop a video in `data/raw_videos/` (or pass a full path) and run the full pipeline:

```
python main.py <video_filename_or_path>
```

This writes intermediate JSON to `data/transcripts/`, `data/captions/`, and `data/screenplays/`, plus the final screenplay to `data/screenplays/<name>.screenplay.md`.

Each stage can also be run independently as a module — see `CLAUDE.md` for the full breakdown of the `src/cli/` entry points (transcribe-only, caption-only, tone-only, merge-only, write-only).

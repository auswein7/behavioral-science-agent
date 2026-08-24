# FAIR Screenplay Utility

A utility run on local AI (via USAFA's `fair-llm` library) that turns video into a "screenplay" — a combined transcript of speech and on-screen action — for use by researchers or other LLM classifiers.

## Pipeline

Three models, orchestrated end-to-end by `main.py`. All three stages are implemented.

- **[WhisperX](https://github.com/m-bain/whisperx)** — diarization scribe. Generates a speaker-labeled speech transcript. Runs via the `whisperx`/PyTorch stack directly, since Ollama has no ASR/diarization capability.
- **[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)** (`qwen3-vl:8b` by default) — visual action observer. Samples video frames at a fixed fps and captions each one, feeding the previous caption back into the prompt so it describes what *changed* rather than re-describing the whole scene. Served locally by Ollama.
- **[Ornith-1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)** (`ornith-1.5-255k` by default) — orchestrator/screenwriter. Takes the merged transcript + caption timeline and writes the final screenplay Markdown, matching the structure in `screenplay_template.md`. Also served locally by Ollama — note Ornith isn't on the public Ollama registry, so it has to already exist locally (`ollama list`) rather than being auto-pulled.

## Setup

```
pip install -r requirements.txt
```

Requires a local [Ollama](https://ollama.com/download) install running in the background for the captioning and orchestrator stages — no GPU/PyTorch setup needed for those. WhisperX still needs the real PyTorch stack for diarization/ASR.

Copy `.env.example` to `.env` and fill in `HF_TOKEN` (needed for the pyannote diarization model used by WhisperX). Model choices, fps, batch size, etc. are also tunable there.

## Usage

Drop a video in `data/raw_videos/` (or pass a full path) and run the full pipeline:

```
python main.py <video_filename_or_path>
```

This writes intermediate JSON to `data/transcripts/`, `data/captions/`, and `data/screenplays/`, plus the final screenplay to `data/screenplays/<name>.screenplay.md`.

Each stage can also be run independently as a module — see `CLAUDE.md` for the full breakdown of the `src/cli/` entry points (transcribe-only, caption-only, merge-only, write-only).

## Future ideas

- **[Qwen3-Omni-30B-A3B-Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)** (or a speech emotion recognition model exported to ONNX) as a tone analyzer (sarcastic, sad, angry, happy, etc.)

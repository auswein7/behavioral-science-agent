# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FAIR Screenplay Utility — a tool run on local AI (via USAFA's `fair-llm` library) that turns video into a "screenplay": a combined transcript of speech and on-screen action, for use by researchers or other LLM classifiers.

The pipeline orchestrates three models, all served locally:
- **WhisperX** (diarization scribe) — generates speech transcripts with speaker diarization. Runs via the `whisperx`/PyTorch stack directly (not Ollama), since Ollama has no ASR/diarization capability.
- **Qwen3-VL** (visual action observer, `qwen3-vl:8b` by default) — generates captions describing what happens on screen, sampled at a fixed fps. Served locally by Ollama.
- **Ornith-1.5** (orchestrator/screenwriter, `ornith-1.5-255k` by default) — combines the transcript and captions into the final screenplay Markdown, matching `screenplay_template.md`. Also served locally by Ollama. Ornith is not on the public Ollama registry, so it must already exist locally (`ollama list`) — it cannot be auto-pulled like Qwen3-VL.

**All three stages are implemented**, plus a `main.py` entry point that runs the full pipeline end-to-end. This is a change from earlier: the orchestrator stage used to be a placeholder and audio used to be extracted to a file on disk — both are now done differently (see Architecture below).

## Setup

```
pip install -r requirements.txt
```

For local development the user has created the 'screenplay' conda environment.

Both the captioning and orchestrator stages require a local [Ollama](https://ollama.com/download) install running in the background — those modules talk to Ollama's local HTTP API rather than loading model weights themselves, so no GPU/PyTorch setup is needed for those stages. This is a deliberate choice to keep the tool distributable to non-technical users without requiring a Python/CUDA environment; prefer Ollama-servable models over `transformers`-loaded ones for any new locally-run model in this project. WhisperX is the one stage that still needs the real PyTorch/whisperx stack, since diarization/ASR isn't available through Ollama.

Config lives in `.env` (see `.env.example`): `HF_TOKEN` (required, for the pyannote diarization model), `WHISPERX_DEVICE`/`WHISPERX_COMPUTE_TYPE`/`WHISPERX_MODEL`/`WHISPERX_BATCH_SIZE`/`WHISPERX_LANGUAGE`, `CAPTION_FPS`/`CAPTION_MODEL`, and `ORNITH_MODEL`.

## Architecture

- `src/video_utils/` — video preprocessing utilities, no `__main__` blocks (imported by other scripts, not run directly).
  - `extract_audio.load_audio_numpy_array(video_path)` decodes a video's audio track straight into an in-memory float32 mono 16kHz numpy array for WhisperX. It no longer writes an audio file to disk — there's no `extract_audio()`-to-file function or `data/audio/` output dir anymore.
  - `extract_frames.load_frames(video_path, fps=1.0, start_time=0.0, max_frames=None)` samples still frames from a video at a fixed wall-clock rate, returning `(timestamp, PIL.Image)` pairs.
  - Both use **PyAV** (`av`), not a shelled-out `ffmpeg` binary — PyAV ships precompiled ffmpeg libraries in its wheel, so no separate ffmpeg install is required. Keep this dependency-free-install property in mind when adding new video/audio processing: prefer libraries that bundle their own codecs over ones that shell out to external binaries.
- `src/captioning/`
  - `caption_image.caption_image(image_path, prompt=..., model_name="qwen3-vl:8b")` — captions a single image via Ollama. Pulls the model automatically on first use if not already present. Raises `ConnectionError` with a user-facing message if Ollama isn't running.
  - `caption_video.caption_video(video_path, fps=1.0, start_time=0.0, max_frames=None, model_name=...)` — samples frames via `extract_frames.load_frames` and captions each one in order, feeding the previous frame's caption back into the prompt so the model describes *what changed* rather than re-describing the whole scene each time. Returns records with `index`/`timestamp`/`timestamp_hms`/`caption`. `write_json` persists these to `data/captions/<name>.captions.json`.
- `src/transcription/` — WhisperX-based diarization pipeline (`diarize_transcript`) plus `format_transcript.format_segments`, which flattens raw WhisperX segments into per-utterance records (`start`/`end`/`start_hms`/`end_hms`/`speaker`/`text`/`word_count`/`avg_word_score`/`low_confidence_words`). `write_json`/`write_csv` persist to `data/transcripts/`.
- `src/screenplay/` — the orchestrator stage.
  - `merge_screenplay.merge_screenplay(captions, transcript)` combines caption records (point-in-time "visual" events) and transcript records (start/end "speech" events) into one chronological list, sorted and re-indexed. `write_json` persists to `data/screenplays/<name>.screenplay.json`.
  - `write_screenplay.write_screenplay(events, template, model_name="ornith-1.5-255k")` sends the merged timeline JSON plus `screenplay_template.md` to the local Ornith model and returns the generated screenplay Markdown, following the template's scene/action/dialogue structure. `write_md` persists to `data/screenplays/<name>.screenplay.md`.
- CLI entry points (each is the *only* place in its script that reads env vars or resolves default paths — everything else under `src/` stays a plain library that takes what it needs as arguments). `main.py` is the sole entry point kept at the repo root; every single-stage entry point lives under `src/cli/` and is run as a module (`python -m src.cli.<name> ...`) from the repo root, since they import via `from src...` and need the repo root on `sys.path`:
  - `main.py <video>` — runs the full pipeline: transcribe → caption → merge → write screenplay.
  - `src/cli/transcribe.py <video>` — audio extraction + diarized transcript only. Run as `python -m src.cli.transcribe <video>`.
  - `src/cli/caption_video.py <video>` — frame sampling + captioning only. Run as `python -m src.cli.caption_video <video>`.
  - `src/cli/build_screenplay.py <captions_json> <transcript_json>` — merge stage only (given existing captions/transcript JSON). Run as `python -m src.cli.build_screenplay ...`.
  - `src/cli/write_screenplay.py <screenplay_json>` — orchestrator stage only (given an existing merged screenplay JSON). Run as `python -m src.cli.write_screenplay <screenplay_json>`.
  - `src/cli/caption_image.py` — a quick single-image captioning demo, not part of the pipeline proper. Run as `python -m src.cli.caption_image`.
- `screenplay_template.md` — the Markdown structure/formatting template Ornith is instructed to match exactly (scene headings with timestamp ranges, prose action lines, timestamped dialogue attributed by speaker).
- `data/raw_videos/` — source video files (input).
- `data/transcripts/`, `data/captions/`, `data/screenplays/` — per-stage JSON (and final `.md`) outputs. There is no longer a `data/audio/` stage since audio is decoded in-memory rather than written to disk.

## Data flow convention

Utility functions default their output paths to `data/<stage>/` relative to the repo root (each package computes `DEFAULT_OUTPUT_DIR` from `Path(__file__).resolve().parents[2]`). Follow this pattern for new pipeline stages so scripts can be run from anywhere in the repo without needing explicit output paths.

"""CLI entry point: run the full pipeline on a video, start to finish.

Extracts audio and transcribes/diarizes it, samples and captions video frames,
merges both into a single timeline, then feeds that timeline to a local Ornith
model to write the final screenplay .md.

This module is the only place that reads environment variables or resolves
default paths — everything under src/ is a plain library that takes what it
needs as arguments. The video file is the only thing passed on the command
line; other settings are tunable via .env (see .env.example).
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.captioning import caption_video
from src.captioning import write_json as write_captions_json
from src.screenplay import merge_screenplay, scrub_check, write_screenplay
from src.screenplay import write_json as write_screenplay_json
from src.screenplay import write_md as write_screenplay_md
from src.tone import classify_tone
from src.tone import write_json as write_tone_json
from src.transcription import diarize_transcript, format_segments
from src.transcription import write_json as write_transcript_json
from src.video_utils import load_audio_numpy_array

logger = logging.getLogger(__name__)


def _optional_int(value: str | None) -> int | None:
    return int(value) if value else None


REPO_ROOT = Path(__file__).resolve().parent
RAW_VIDEOS_DIR = REPO_ROOT / "data" / "raw_videos"
CAPTIONS_DIR = REPO_ROOT / "data" / "captions"
TRANSCRIPTS_DIR = REPO_ROOT / "data" / "transcripts"
SCREENPLAYS_DIR = REPO_ROOT / "data" / "screenplays"
TEMPLATE_PATH = REPO_ROOT / "screenplay_template.md"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <video_filename_or_path>")
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        video_path = RAW_VIDEOS_DIR / video_path
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    name = video_path.stem

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set. Add it to .env.")

    logger.info("[1/5] Transcribing audio: %s", video_path)
    audio = load_audio_numpy_array(video_path)
    diarize_result = diarize_transcript(
        audio,
        hf_token=hf_token,
        device=os.environ.get("WHISPERX_DEVICE", "cpu"),
        compute_type=os.environ.get("WHISPERX_COMPUTE_TYPE", "int8"),
        whisper_model=os.environ.get("WHISPERX_MODEL", "large-v2"),
        batch_size=int(os.environ.get("WHISPERX_BATCH_SIZE", "16")),
        num_speakers=_optional_int(os.environ.get("WHISPERX_NUM_SPEAKERS")),
        min_speakers=_optional_int(os.environ.get("WHISPERX_MIN_SPEAKERS")),
        max_speakers=_optional_int(os.environ.get("WHISPERX_MAX_SPEAKERS")),
        language=os.environ.get("WHISPERX_LANGUAGE", "en"),
    )
    transcript_records = format_segments(diarize_result["segments"])
    write_transcript_json(transcript_records, name, TRANSCRIPTS_DIR)

    logger.info("[2/5] Classifying utterance tone: %s", video_path)
    tone_records = classify_tone(
        transcript_records,
        audio,
        model_name=os.environ.get("TONE_MODEL", "iic/emotion2vec_plus_large"),
    )
    write_tone_json(tone_records, name, TRANSCRIPTS_DIR)

    logger.info("[3/5] Captioning video frames (fixed-interval + speech-start, one chronological pass): %s", video_path)
    context_raw = os.environ.get("CAPTION_CONTEXT_CAPTIONS", "1").strip()
    context_captions = None if context_raw.lower() == "all" else int(context_raw)
    max_dimension_raw = os.environ.get("CAPTION_MAX_DIMENSION", "1280").strip()
    max_dimension = None if max_dimension_raw.lower() == "none" else int(max_dimension_raw)
    caption_records = caption_video(
        video_path,
        fps=float(os.environ.get("CAPTION_FPS", "1.0")),
        model_name=os.environ.get("CAPTION_MODEL", "qwen3-vl:8b"),
        context_captions=context_captions,
        utterances=[{"start": record["start"], "speaker": record["speaker"]} for record in tone_records],
        speech_start_offset=float(os.environ.get("SPEECH_FRAME_OFFSET", "0.5")),
        burst_count=int(os.environ.get("SPEECH_BURST_FRAMES", "2")),
        burst_spacing=float(os.environ.get("SPEECH_BURST_SPACING", "0.2")),
        max_dimension=max_dimension,
    )
    write_captions_json(caption_records, name, CAPTIONS_DIR)
    scrub_check("\n".join(record["caption"] for record in caption_records), "captions")

    logger.info("[4/5] Merging tone-coded transcript and captions into one timeline")
    events = merge_screenplay(caption_records, tone_records)
    write_screenplay_json(events, name, SCREENPLAYS_DIR)

    logger.info("[5/5] Writing final screenplay with Ornith")
    template = TEMPLATE_PATH.read_text()
    screenplay_md = write_screenplay(
        events,
        template,
        model_name=os.environ.get("ORNITH_MODEL", "ornith-1.5-255k"),
    )
    output_path = write_screenplay_md(screenplay_md, name, SCREENPLAYS_DIR)
    scrub_check(screenplay_md, "screenplay.md")

    logger.info("Pipeline complete: %s", output_path)


if __name__ == "__main__":
    main()

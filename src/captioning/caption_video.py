"""Caption every sampled frame of a video, tracking what's changed since the previous frame."""

import io
import json
import logging
from pathlib import Path

import ollama
from PIL import Image

from src.video_utils.extract_frames import load_frames

from .caption_image import DEFAULT_MODEL, _ensure_model_pulled

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "captions"

CHANGE_PROMPT = (
    "This is a still frame from a video, sampled after the following caption "
    "described the previous frame:\n\n"
    '"{previous_caption}"\n\n'
    "Describe what is happening in this new frame, focusing on what has changed "
    "since the previous one (movement, new or departed subjects, changed actions "
    "or expressions). If nothing meaningful has changed, say so briefly."
)
FIRST_FRAME_PROMPT = "Describe what is happening in this image."


def _image_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _hhmmss(seconds: float) -> str:
    whole = int(seconds)
    hh, rem = divmod(whole, 3600)
    mm, ss = divmod(rem, 60)
    ms = round((seconds - whole) * 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def caption_video(
    video_path: Path,
    fps: float = 1.0,
    start_time: float = 0.0,
    max_frames: int | None = None,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """Sample frames from a video and caption each one, in order, with change-focused context.

    Each frame's caption is generated with the previous frame's caption fed back into the
    prompt, asking the model to focus on what changed rather than re-describing the whole
    scene. Returns a list of records with timestamp (seconds + hh:mm:ss) and caption.
    """
    video_path = Path(video_path)
    _ensure_model_pulled(model_name)

    frames, _ = load_frames(video_path, fps=fps, start_time=start_time, max_frames=max_frames)

    records = []
    previous_caption = None
    for i, (timestamp, image) in enumerate(frames):
        prompt = (
            FIRST_FRAME_PROMPT
            if previous_caption is None
            else CHANGE_PROMPT.format(previous_caption=previous_caption)
        )

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt, "images": [_image_bytes(image)]}],
        )
        caption = response.message.content.strip()
        logger.info("Captioned frame %d @ %.2fs: %s", i, timestamp, caption)

        records.append(
            {
                "index": i,
                "timestamp": round(timestamp, 3),
                "timestamp_hms": _hhmmss(timestamp),
                "caption": caption,
            }
        )
        previous_caption = caption

    return records


def write_json(records: list[dict], name: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.captions.json"
    output_path.write_text(json.dumps(records, indent=2))
    logger.info("Wrote %d records to %s", len(records), output_path)
    return output_path

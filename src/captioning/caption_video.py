"""Caption every sampled frame of a video, tracking what's changed since the previous frame."""

import io
import logging
import re
from pathlib import Path

from PIL import Image

from src.adapters import AbstractChatModel, ChatMessage, OllamaChatModel
from src.persist import write_records
from src.prompts import (
    CHANGE_PROMPT,
    FIRST_FRAME_PROMPT,
    SPEECH_START_BURST_CONTINUATION_PROMPT,
    SPEECH_START_PROMPT,
)
from src.video_utils.extract_frames import load_burst_frames_at, load_frames

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3-vl:8b"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "captions"

_NO_PRIOR_CAPTION = "(no earlier description available)"
_SPEAKER_TAG_RE = re.compile(r"SPEAKER_\d+")


def _numbered_captions(previous_captions: list[str]) -> str:
    if not previous_captions:
        return _NO_PRIOR_CAPTION
    return "\n".join(f'{i + 1}. "{c}"' for i, c in enumerate(previous_captions))


def _change_prompt(previous_captions: list[str]) -> str:
    return CHANGE_PROMPT.format(previous_captions=_numbered_captions(previous_captions))


def _speech_start_prompt(speaker: str, previous_captions: list[str]) -> str:
    return SPEECH_START_PROMPT.format(
        speaker=speaker, previous_captions=_numbered_captions(previous_captions)
    )


def _resize_for_captioning(
    image: Image.Image, max_dimension: int | None
) -> Image.Image:
    """Downscale so the image's long edge is at most `max_dimension`, never upscale.

    Qwen3-VL's vision tokens scale linearly with pixel count (H*W/1024), and Ollama does
    not cap this for you — the model's own default max_pixels (~12.8MP) is far above a
    source video's native resolution, so a 2K/4K frame sent as-is pays for vision tokens
    proportionally, well past the point where extra pixels help a captioner reading gaze
    and body language rather than fine text. `max_dimension=None` (or an image already
    at or under it) skips resizing entirely.
    """
    if max_dimension is None:
        return image
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_dimension:
        return image
    scale = max_dimension / long_edge
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _image_bytes(image: Image.Image, max_dimension: int | None = None) -> bytes:
    image = _resize_for_captioning(image, max_dimension)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _hhmmss(seconds: float) -> str:
    whole = int(seconds)
    hh, rem = divmod(whole, 3600)
    mm, ss = divmod(rem, 60)
    ms = round((seconds - whole) * 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def _caption_single(
    model: AbstractChatModel,
    image: Image.Image,
    prompt: str,
    max_dimension: int | None = None,
    options: dict | None = None,
) -> str:
    message = ChatMessage(
        role="user", content=prompt, images=(_image_bytes(image, max_dimension),)
    )
    response = model.invoke([message], **(options or {}))
    return response.content.strip()


def _caption_speech_burst(
    model: AbstractChatModel,
    burst: list[tuple[float, Image.Image]],
    speaker: str,
    outer_context: list[str],
    burst_spacing: float,
    max_dimension: int | None = None,
    options: dict | None = None,
) -> tuple[str, list[str]]:
    """Caption a burst of frames sampled around one utterance's speech-start moment.

    Chains sequential single-image calls — the first frame via `_speech_start_prompt`
    (seeded with the outer rolling caption context, same as any other frame), each
    subsequent frame via SPEECH_START_BURST_CONTINUATION_PROMPT seeded with the *prior
    burst frame's* caption, so the model can compare frames for who's speaking. This
    deliberately avoids sending multiple images in one Ollama call: as of writing,
    same-dimension images in a single qwen-vl chat message can be silently fused into
    one frame by the tokenizer (ollama/ollama#17321), or the request can return no
    response at all (ollama/ollama#13047) — both fail silently, which is disqualifying
    for a pipeline whose entire purpose is trustworthy de-identification. Sequential
    single-image calls reuse the exact call shape used everywhere else in this module.

    Returns (final_caption, all_burst_captions) — only the final caption is meant to be
    promoted to the outer timeline; all_burst_captions is available as an optional debug
    trail of how the model's read of the situation evolved across the burst.
    """
    captions: list[str] = []
    for i, (_, image) in enumerate(burst):
        if i == 0:
            prompt = _speech_start_prompt(speaker, outer_context)
        else:
            prompt = SPEECH_START_BURST_CONTINUATION_PROMPT.format(
                speaker=speaker, spacing=burst_spacing, previous_caption=captions[-1]
            )
        captions.append(_caption_single(model, image, prompt, max_dimension, options))

    return captions[-1], captions


def caption_video(
    video_path: Path,
    fps: float = 1.0,
    start_time: float = 0.0,
    max_frames: int | None = None,
    model_name: str = DEFAULT_MODEL,
    context_captions: int | None = 1,
    utterances: list[dict] | None = None,
    speech_start_offset: float = 0.5,
    burst_count: int = 1,
    burst_spacing: float = 0.2,
    max_dimension: int | None = 1280,
    temperature: float = 0.0,
    seed: int | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 25,
    model: AbstractChatModel | None = None,
) -> list[dict]:
    """Caption a video in one chronologically-ordered, context-chained pass.

    Combines two kinds of sampled frames into a single timestamp-ordered sequence:
    fixed-interval frames (`fps`, via `load_frames`) and, if `utterances` is given (a
    list of `{"start": float, "speaker": str}` dicts — e.g. a tone-enriched transcript's
    per-utterance start time and diarized speaker label, deliberately not the full
    record so dialogue `text` never reaches the vision prompt), a burst of `burst_count`
    "reaction shot" frames per utterance starting at `start + speech_start_offset`,
    spaced `burst_spacing` seconds apart (via `load_burst_frames_at`). Utterances whose
    first burst frame falls past the end of the video are skipped with a warning.

    Every frame is downscaled (long edge to `max_dimension` px, never upscaled — pass
    `None` to send frames at native resolution) and JPEG-encoded before being sent to
    Ollama, since Qwen3-VL's vision-token cost scales with pixel count and a source
    video's native resolution (1080p/2K/etc.) is well past the point of diminishing
    returns for reading gaze and body language. See `_resize_for_captioning`.

    Every frame (or burst) in the combined sequence is captioned in timestamp order,
    each with the previous `context_captions` captions fed back into the prompt (most
    recent last) so the model describes what changed rather than re-describing the whole
    scene — a speech-triggered caption is available as context for the next
    fixed-interval one and vice versa. Pass `context_captions=None` to feed back every
    prior caption instead of a fixed window.

    A fixed-interval frame is captioned with a single Ollama call. A speech-start burst
    is captioned by chaining `burst_count` sequential single-image calls (see
    `_caption_speech_burst`) — the model is told which SPEAKER_NN just started talking
    and asked to determine, from posture/gesture/gaze evidence across the burst, which
    visible person that is (falling back to neutral language if it can't tell) — but
    only the final call's caption is promoted to this entry's "caption" field, so burst
    frames never inflate the merged timeline's event count. `burst_count=1` (the
    default) skips the chain and costs exactly one call, like today.

    Returns a list of records with timestamp (seconds + hh:mm:ss), caption, a "trigger"
    field ("fixed_interval" or "speech_start"), and "mentioned_speakers" (any SPEAKER_NN
    tags found in the caption text, from either a grounded speech-start identification
    or a tag carried forward from context — a structured cross-check for downstream
    consumers, since prompt compliance is probabilistic). Speech-triggered records also
    carry the originating "speech_start" timestamp and, when burst_count > 1, a
    "burst_captions" debug trail of the full per-frame chain.

    Sampling is deterministic by default (temperature 0, design principle 8); the
    options sent to Ollama match what the run's provenance records. When
    `checkpoint_path` is set, the records so far are flushed there every
    `checkpoint_every` captions (principle 6: a multi-hour stage never holds its
    only copy of the work in memory).

    Pass `model` to run the stage against any AbstractChatModel backend (a fairlib
    adapter, a fake in tests); when None, a local OllamaChatModel for `model_name`
    is constructed as before. The stage never learns which backend it got.
    """
    video_path = Path(video_path)
    # Injected backend or the default local one; the stage only ever sees the
    # AbstractChatModel interface (design principle 2).
    if model is None:
        model = OllamaChatModel(model_name, auto_pull=True, description="frame caption")
    model.ensure_available()

    options: dict = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed

    fixed_frames, _ = load_frames(
        video_path, fps=fps, start_time=start_time, max_frames=max_frames
    )
    entries = [
        {"timestamp": timestamp, "trigger": "fixed_interval", "image": image}
        for timestamp, image in fixed_frames
    ]

    if utterances:
        targets = [u["start"] + speech_start_offset for u in utterances]
        bursts = load_burst_frames_at(
            video_path, targets, burst_count=burst_count, burst_spacing=burst_spacing
        )

        for utterance, target, burst in zip(utterances, targets, bursts, strict=True):
            if not burst:
                logger.warning(
                    "Speech-start frame at %.2fs (utterance start %.2fs + %.2fs offset) is past "
                    "the end of the video; skipping",
                    target,
                    utterance["start"],
                    speech_start_offset,
                )
                continue
            if len(burst) < burst_count:
                logger.info(
                    "Speech-start burst at %.2fs got %d/%d frames (near end of video)",
                    target,
                    len(burst),
                    burst_count,
                )
            entries.append(
                {
                    "timestamp": target,
                    "trigger": "speech_start",
                    "burst": burst,
                    "speaker": utterance["speaker"],
                    "speech_start": utterance["start"],
                }
            )

    entries.sort(key=lambda e: e["timestamp"])

    records = []
    previous_captions: list[str] = []
    for i, entry in enumerate(entries):
        timestamp = entry["timestamp"]
        context = (
            previous_captions
            if context_captions is None
            else previous_captions[-context_captions:]
        )

        burst_captions = None
        if entry["trigger"] == "fixed_interval":
            prompt = FIRST_FRAME_PROMPT if not context else _change_prompt(context)
            caption = _caption_single(
                model, entry["image"], prompt, max_dimension, options
            )
        else:
            caption, burst_captions = _caption_speech_burst(
                model,
                entry["burst"],
                entry["speaker"],
                context,
                burst_spacing,
                max_dimension,
                options,
            )

        logger.info(
            "Captioned %s frame %d @ %.2fs: %s", entry["trigger"], i, timestamp, caption
        )

        record = {
            "index": i,
            "timestamp": round(timestamp, 3),
            "timestamp_hms": _hhmmss(timestamp),
            "caption": caption,
            "trigger": entry["trigger"],
            "mentioned_speakers": sorted(set(_SPEAKER_TAG_RE.findall(caption))),
        }
        if "speech_start" in entry:
            record["speech_start"] = round(entry["speech_start"], 3)
        if burst_captions is not None and len(burst_captions) > 1:
            record["burst_captions"] = burst_captions

        records.append(record)
        previous_captions.append(caption)

        if checkpoint_path is not None and len(records) % checkpoint_every == 0:
            write_records(records, checkpoint_path)
            logger.info(
                "Checkpointed %d caption records to %s", len(records), checkpoint_path
            )

    if checkpoint_path is not None and records:
        write_records(records, checkpoint_path)

    return records


def write_json(
    records: list[dict], name: str, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.captions.json"
    write_records(records, output_path)
    logger.info("Wrote %d records to %s", len(records), output_path)
    return output_path

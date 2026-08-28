"""Classify the emotional tone of each diarized utterance using emotion2vec+.

emotion2vec+ isn't servable through Ollama, so this stage loads its model directly
via a dedicated library (funasr) instead. It's small (largest variant is a few
hundred MB) and runs comfortably on CPU, so it needs no CUDA/MPS-specific handling.
"""

import functools
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "transcripts"

DEFAULT_MODEL = "iic/emotion2vec_plus_large"
DEFAULT_HUB = "hf"
SAMPLE_RATE = 16000  # matches src/video_utils/extract_audio.py's WhisperX-ready audio


@functools.lru_cache(maxsize=1)
def _load_model(model_name: str, hub: str):
    from funasr import AutoModel

    logger.info("Loading %s (hub=%s)...", model_name, hub)
    return AutoModel(model=model_name, hub=hub)


def _write_temp_wav(samples: np.ndarray, sample_rate: int) -> Path:
    """emotion2vec's funasr pipeline takes a file path, not an in-memory array."""
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(out_path, samples, sample_rate, subtype="PCM_16")
    return Path(out_path)


def _classify_clip(model, samples: np.ndarray) -> tuple[str, dict[str, float]] | None:
    if samples.size == 0:
        return None

    wav_path = _write_temp_wav(samples, SAMPLE_RATE)
    try:
        result = model.generate(
            str(wav_path), granularity="utterance", extract_embedding=False, disable_pbar=True
        )
    finally:
        wav_path.unlink(missing_ok=True)

    top = result[0]
    # Labels can come back as bare English ("happy") or bilingual ("生气/angry")
    # depending on model/hub; normalize to the English tag either way.
    labels = [label.split("/")[-1] for label in top["labels"]]
    scores = {label: round(float(score), 4) for label, score in zip(labels, top["scores"])}
    top_label = max(scores, key=scores.get)
    return top_label, scores


def classify_tone(
    records: list[dict],
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    hub: str = DEFAULT_HUB,
    sample_rate: int = SAMPLE_RATE,
) -> list[dict]:
    """Add an emotion classification to each utterance record via emotion2vec+.

    `audio` is the full-video mono audio array at `sample_rate` (e.g. from
    src/video_utils/extract_audio.load_audio_numpy_array), and `records` are
    formatted transcript utterances (src/transcription/format_transcript.format_segments)
    with "start"/"end" in seconds. Returns a new list of records, each the original
    fields plus "emotion" (top label) and "emotion_scores" (per-label confidence);
    both are None for utterances too short to classify.
    """
    model = _load_model(model_name, hub)

    enriched = []
    for record in records:
        start_sample = int(record["start"] * sample_rate)
        end_sample = int(record["end"] * sample_rate)
        clip = audio[start_sample:end_sample]

        classification = _classify_clip(model, clip)
        if classification is None:
            logger.warning(
                "Utterance %d (%.2fs-%.2fs) too short to classify; skipping",
                record["index"], record["start"], record["end"],
            )
            emotion, scores = None, None
        else:
            emotion, scores = classification
            logger.info("Utterance %d (%.2fs-%.2fs): %s", record["index"], record["start"], record["end"], emotion)

        enriched.append({**record, "emotion": emotion, "emotion_scores": scores})

    return enriched


def write_json(records: list[dict], name: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.formatted.tone.json"
    output_path.write_text(json.dumps(records, indent=2))
    logger.info("Wrote %d records to %s", len(records), output_path)
    return output_path

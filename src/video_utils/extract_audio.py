"""Decode the audio track from an MP4 file into an in-memory array."""

import logging
from pathlib import Path

import av
import numpy as np

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


def load_audio_numpy_array(video_path: Path) -> np.ndarray:
    """Decode a video's audio track to a float32 mono 16kHz array, e.g. for WhisperX."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    resampler = av.AudioResampler(format="s16", layout="mono", rate=WHISPER_SAMPLE_RATE)
    chunks = []

    with av.open(str(video_path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray())

    samples = np.concatenate(chunks, axis=1).flatten()
    audio = samples.astype(np.float32) / 32768.0
    duration_sec = len(audio) / WHISPER_SAMPLE_RATE
    logger.info("Extracted %.1fs of audio from %s", duration_sec, video_path)
    return audio

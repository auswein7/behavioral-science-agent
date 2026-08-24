"""Generate a diarized speech transcript from an audio array using WhisperX."""

import gc
import logging
import warnings

import numpy as np
import whisperx
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)


def diarize_transcript(
    audio: np.ndarray,
    hf_token: str,
    device: str = "cpu",
    compute_type: str = "int8",
    whisper_model: str = "large-v2",
    batch_size: int = 16,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    language: str = "en",
) -> dict:
    duration_sec = len(audio) / 16000
    logger.info(
        "Diarizing %.1fs of audio (whisper_model=%s, device=%s, compute_type=%s)",
        duration_sec,
        whisper_model,
        device,
        compute_type,
    )

    # 1. Transcribe with Whisper (batched)
    model = whisperx.load_model(whisper_model, device, compute_type=compute_type, language=language)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    logger.info("Transcribed %d segments (language=%s)", len(result["segments"]), result["language"])
    del model
    gc.collect()

    # 2. Align output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    del model_a
    gc.collect()

    # 3. Assign speaker labels
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    logger.info("Diarization complete")

    return result

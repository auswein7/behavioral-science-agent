"""Integration tests over a synthesized fixture clip: real PyAV decode paths
(audio + frames) and the captioning stage end to end with Ollama mocked.

The clip is generated in-test (4 s, 64x64, 4 fps video + 440 Hz mono audio) so
no media file lives in the repo and the video/audio decode path is exercised
for real on every run.
"""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from src import adapters
from src.persist import read_records
from src.video_utils import load_audio_numpy_array
from src.video_utils.extract_frames import load_burst_frames_at, load_frames

# The package __init__ re-exports the caption_video function under the same
# name, so a plain attribute import would grab the function, not the module.
cv = importlib.import_module("src.captioning.caption_video")

DURATION = 4
SAMPLE_RATE = 16000


@pytest.fixture(scope="module")
def fixture_clip(tmp_path_factory):
    import av

    path = tmp_path_factory.mktemp("clip") / "fixture.mp4"
    container = av.open(str(path), "w")
    video = container.add_stream("libx264", rate=4)
    video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
    audio = container.add_stream("aac", rate=SAMPLE_RATE, layout="mono")

    for i in range(DURATION * 4):
        img = np.full((64, 64, 3), (i * 16) % 256, dtype=np.uint8)
        for packet in video.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            container.mux(packet)

    t = np.arange(DURATION * SAMPLE_RATE) / SAMPLE_RATE
    tone = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    for start in range(0, len(tone), 1024):
        frame = av.AudioFrame.from_ndarray(tone[start : start + 1024].reshape(1, -1),
                                           format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        for packet in audio.encode(frame):
            container.mux(packet)

    for stream in (video, audio):
        for packet in stream.encode():
            container.mux(packet)
    container.close()
    return path


class TestVideoUtils:
    def test_audio_decodes_to_16k_mono_float32(self, fixture_clip):
        audio = load_audio_numpy_array(fixture_clip)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert abs(len(audio) / SAMPLE_RATE - DURATION) < 0.5

    def test_fixed_interval_frames(self, fixture_clip):
        frames, _meta = load_frames(fixture_clip, fps=1.0)
        assert len(frames) == DURATION
        timestamps = [t for t, _ in frames]
        assert timestamps == sorted(timestamps)

    def test_burst_frames_positionally_aligned(self, fixture_clip):
        bursts = load_burst_frames_at(fixture_clip, [0.5, 99.0], burst_count=2, burst_spacing=0.2)
        assert len(bursts) == 2
        assert len(bursts[0]) == 2
        assert bursts[1] == []  # past the end: an empty slot, never an omission


class FakeOllama:
    """Stands in for the ollama module behind the adapter layer."""

    def __init__(self):
        self.chat_calls: list[dict] = []

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model="fake-vl:latest")])

    def chat(self, model, messages, options=None):
        self.chat_calls.append({"model": model, "messages": messages, "options": options})
        return SimpleNamespace(
            message=SimpleNamespace(content=f"SPEAKER_00 gestures (call {len(self.chat_calls)}).")
        )


class TestCaptionStage:
    def test_caption_video_end_to_end_with_mocked_model(self, fixture_clip, tmp_path, monkeypatch):
        fake = FakeOllama()
        monkeypatch.setattr(adapters, "ollama", fake)
        checkpoint = tmp_path / "captions.partial.json"

        records = cv.caption_video(
            fixture_clip,
            fps=1.0,
            model_name="fake-vl",
            utterances=[{"start": 1.0, "speaker": "SPEAKER_00"}],
            speech_start_offset=0.5,
            burst_count=2,
            burst_spacing=0.2,
            checkpoint_path=checkpoint,
            checkpoint_every=2,
        )

        # 4 fixed-interval frames plus 1 speech-start record (burst promotes one).
        assert len(records) == 5
        timestamps = [r["timestamp"] for r in records]
        assert timestamps == sorted(timestamps)
        speech = [r for r in records if r["trigger"] == "speech_start"]
        assert len(speech) == 1
        assert speech[0]["speech_start"] == 1.0
        assert len(speech[0]["burst_captions"]) == 2
        assert speech[0]["mentioned_speakers"] == ["SPEAKER_00"]
        # 4 fixed calls + 2 burst calls, each carrying deterministic sampling.
        assert len(fake.chat_calls) == 6
        assert all(c["options"]["temperature"] == 0.0 for c in fake.chat_calls)
        # The checkpoint holds the complete record set at the end of the stage.
        assert read_records(checkpoint) == records

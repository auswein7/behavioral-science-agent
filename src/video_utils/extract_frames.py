"""Extract still frames from an MP4 file at a fixed wall-clock rate."""

import av
from pathlib import Path
from PIL import Image


def load_frames(
    video_path: Path,
    fps: float = 1.0,
    start_time: float = 0.0,
    max_frames: int | None = None,
) -> tuple[list[tuple[float, Image.Image]], float | None]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    interval = 1.0 / fps
    frames: list[tuple[float, Image.Image]] = []
    next_sample_time = start_time

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]

        if start_time > 0:
            container.seek(int(start_time / stream.time_base), stream=stream)

        for frame in container.decode(stream):
            timestamp = float(frame.time)

            if timestamp < start_time:
                continue

            if timestamp < next_sample_time:
                continue

            if max_frames is not None and len(frames) >= max_frames:
                return frames, timestamp

            frames.append((timestamp, frame.to_image()))
            next_sample_time += interval

    return frames, None

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


def load_burst_frames_at(
    video_path: Path,
    timestamps: list[float],
    burst_count: int = 1,
    burst_spacing: float = 0.2,
) -> list[list[tuple[float, Image.Image]]]:
    """For each of `timestamps`, grab up to `burst_count` frames spaced `burst_spacing`
    seconds apart starting at that timestamp (t, t+spacing, t+2*spacing, ...), in one
    decode pass across every target combined.

    The result is positionally aligned with `timestamps` — one inner list per input
    timestamp, in the same order — rather than a dict keyed by timestamp value. This
    matters once a caller attaches identity (e.g. a speaker label) to each timestamp: a
    dict keyed by a derived offset can silently collide and drop an entry if two
    timestamps land on the same target, which would silently drop or swap identity
    rather than just a frame. Positional alignment makes that class of bug impossible
    by construction. A timestamp near the end of the video may come back with fewer
    than burst_count frames, or an empty list, if some of its burst offsets fall past
    the video's duration — this never affects any other timestamp's result.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not timestamps or burst_count < 1:
        return [[] for _ in timestamps]

    # Every individual frame to fetch, tagged with which (timestamp index, burst
    # position) it belongs to. A single decoded frame can satisfy more than one owner
    # (e.g. two bursts landing on the same rounded offset) — both just receive it.
    owners_by_target: dict[float, list[tuple[int, int]]] = {}
    for ts_idx, ts in enumerate(timestamps):
        for burst_idx in range(burst_count):
            target = round(ts + burst_idx * burst_spacing, 6)
            owners_by_target.setdefault(target, []).append((ts_idx, burst_idx))

    sorted_targets = sorted(owners_by_target)
    found: dict[float, Image.Image] = {}

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        target_idx = 0

        if sorted_targets[0] > 0:
            container.seek(int(sorted_targets[0] / stream.time_base), stream=stream)

        for frame in container.decode(stream):
            if target_idx >= len(sorted_targets):
                break

            timestamp = float(frame.time)
            if timestamp < sorted_targets[target_idx]:
                continue

            image = frame.to_image()
            while target_idx < len(sorted_targets) and sorted_targets[target_idx] <= timestamp:
                found[sorted_targets[target_idx]] = image
                target_idx += 1

    bursts: list[list[tuple[float, Image.Image]]] = [[] for _ in timestamps]
    for target, image in found.items():
        for ts_idx, burst_idx in owners_by_target[target]:
            bursts[ts_idx].append((target, image))
    for burst in bursts:
        burst.sort(key=lambda pair: pair[0])

    return bursts

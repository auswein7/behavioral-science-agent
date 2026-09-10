"""CLI entry point: run the full pipeline on a video, start to finish.

Extracts audio and transcribes/diarizes it, scrubs spoken names, samples and
captions video frames, merges both into a single timeline, feeds that timeline
to a local Ornith model to write the final screenplay .md, then records run
provenance and gates the result.

This module is the only place that reads environment variables or resolves
default paths - everything under src/ is a plain library that takes what it
needs as arguments. Environment values become a typed RunConfig via
src/config.py, and a preflight check fails fast on a missing token, model, or
GPU before any expensive work starts.

Observability: every stage emits typed events on a fairlib AgentEventBus
(src/events.py) - started, finished, resumed, degraded, failed - and the gate
emits its decision. The run log and the provenance's stage timings are
subscribers to that bus.

Resumability: --from <stage> skips earlier stages and loads their persisted
outputs instead (e.g. --from screenplay redoes only the Ornith stage after a
crash, instead of five hours of captioning). Stage logs also stream to
data/runs/<run_id>.log by default.
"""

import argparse
import hashlib
import logging
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from fairlib.core.event_bus import AgentEventBus

from src.adapters import UsageTally
from src.backends import build_caption_model, build_screenplay_model
from src.captioning import caption_video
from src.captioning import write_json as write_captions_json
from src.config import load_run_config, preflight
from src.events import PipelineEvents, RunLog, StageTimings
from src.persist import read_records
from src.reliability import RETRYABLE
from src.schemas import RunConfig, RunProvenance, SessionManifest, StageUsage
from src.screenplay import merge_screenplay, scrub_check, write_screenplay
from src.screenplay import write_json as write_screenplay_json
from src.screenplay import write_md as write_screenplay_md
from src.screenplay.write_screenplay import ORNITH_UNAVAILABLE_HINT
from src.scrub import (
    evaluate_gate,
    require_pass,
    scrub_names,
    write_findings,
    write_report,
    write_scrubbed_json,
)
from src.tone import classify_tone
from src.tone import write_json as write_tone_json
from src.transcription import diarize_transcript, format_segments
from src.transcription import write_json as write_transcript_json
from src.video_utils import load_audio_numpy_array

logger = logging.getLogger(__name__)

STAGES = ("transcribe", "tone", "caption", "merge", "screenplay")

REPO_ROOT = Path(__file__).resolve().parent
RAW_VIDEOS_DIR = REPO_ROOT / "data" / "raw_videos"
CAPTIONS_DIR = REPO_ROOT / "data" / "captions"
TRANSCRIPTS_DIR = REPO_ROOT / "data" / "transcripts"
SCREENPLAYS_DIR = REPO_ROOT / "data" / "screenplays"
DELIVERABLES_DIR = REPO_ROOT / "data" / "deliverables"
RUN_LOGS_DIR = REPO_ROOT / "data" / "runs"
TEMPLATE_PATH = REPO_ROOT / "screenplay_template.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_digests(config: RunConfig) -> dict[str, str]:
    """Best-effort Ollama digests for the models this run uses; preflight has
    already established Ollama is reachable."""
    import ollama

    digests = {}
    try:
        for model in ollama.list().models:
            base = model.model.split(":")[0]
            if base in (config.captioning.caption_model.split(":")[0],
                        config.screenplay.ornith_model.split(":")[0]):
                digests[model.model] = model.digest
    except RETRYABLE:
        logger.warning("Could not read model digests from Ollama; provenance will omit them")
    return digests


def _software_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": platform.python_version()}
    for package in ("whisperx", "funasr", "spacy", "ollama", "pydantic", "av", "fair-llm"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="video filename (under data/raw_videos/) or path")
    parser.add_argument("--from", dest="start", choices=STAGES, default="transcribe",
                        help="resume from this stage, loading earlier stages' persisted outputs")
    parser.add_argument("--run-id", default=None, help="run id (default: local-<utc stamp>)")
    parser.add_argument("--manifest", default=None,
                        help="SessionManifest JSON (data/manifests/<session_id>.manifest.json); "
                             "its expected_speaker_count overrides WHISPERX_NUM_SPEAKERS and its "
                             "session_id names the provenance and gate records")
    args = parser.parse_args()

    run_id = args.run_id or "local-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(RUN_LOGS_DIR / f"{run_id}.log")],
    )
    load_dotenv()
    config = load_run_config()
    hf_token = os.environ.get("HF_TOKEN")

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = RAW_VIDEOS_DIR / video_path
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    name = video_path.stem
    start = STAGES.index(args.start)
    started = datetime.now(UTC).isoformat(timespec="seconds")

    manifest = None
    if args.manifest:
        manifest = SessionManifest.model_validate_json(Path(args.manifest).read_text())
        logger.info("Session manifest loaded: %s (%s, expected %d speakers)",
                    manifest.session_id, manifest.source_class, manifest.expected_speaker_count)
    session_id = manifest.session_id if manifest else name
    num_speakers = manifest.expected_speaker_count if manifest else config.transcription.num_speakers

    # Stage events: the run log and the provenance timings subscribe here;
    # nothing below times or narrates a stage by hand (design principle 4).
    bus = AgentEventBus()
    pipeline = PipelineEvents(bus, run_id)
    stage_timings = StageTimings()
    stage_timings.subscribe(bus)
    RunLog().subscribe(bus)

    preflight(config, hf_token, need_transcription=start <= STAGES.index("tone"),
              need_captioning=start <= STAGES.index("caption"))

    usage_tallies: dict[str, UsageTally] = {}
    audio = None
    if start <= STAGES.index("tone"):
        audio = load_audio_numpy_array(video_path)

    if start <= STAGES.index("transcribe"):
        with pipeline.stage("transcribe") as stage:
            diarize_result = diarize_transcript(
                audio,
                hf_token=hf_token,
                device=config.transcription.device,
                compute_type=config.transcription.compute_type,
                whisper_model=config.transcription.whisper_model,
                batch_size=config.transcription.batch_size,
                num_speakers=num_speakers,
                min_speakers=config.transcription.min_speakers,
                max_speakers=config.transcription.max_speakers,
                language=config.transcription.language,
            )
            transcript_records = format_segments(diarize_result["segments"])
            write_transcript_json(transcript_records, name, TRANSCRIPTS_DIR)
            stage.records = len(transcript_records)
        if manifest:
            found = len({r["speaker"] for r in transcript_records})
            if found != manifest.expected_speaker_count:
                pipeline.degraded(
                    "transcribe",
                    f"diarization found {found} speakers but the manifest expects "
                    f"{manifest.expected_speaker_count}; review speaker assignments "
                    "before delivering",
                )
    else:
        source = TRANSCRIPTS_DIR / f"{name}.formatted.json"
        transcript_records = read_records(source)
        pipeline.resumed("transcribe", len(transcript_records), source.name)

    if start <= STAGES.index("tone"):
        with pipeline.stage("tone") as stage:
            tone_records = classify_tone(transcript_records, audio, model_name=config.tone.tone_model)
            write_tone_json(tone_records, name, TRANSCRIPTS_DIR)
            stage.records = len(tone_records)
    else:
        source = TRANSCRIPTS_DIR / f"{name}.formatted.tone.json"
        tone_records = read_records(source)
        pipeline.resumed("tone", len(tone_records), source.name)

    # Deterministic spoken-name scrub: downstream stages (Ornith, the utterance
    # table) only ever see the scrubbed dialogue; the verbatim transcript stays
    # Tier B on disk alongside the replacement log. Cheap, so it always reruns.
    with pipeline.stage("scrub_names") as stage:
        tone_records, replacements = scrub_names(tone_records)
        write_scrubbed_json(tone_records, replacements, name, TRANSCRIPTS_DIR)
        stage.records = len(replacements)

    if start <= STAGES.index("caption"):
        with pipeline.stage("caption") as stage:
            usage_tallies["caption"] = UsageTally()
            caption_model = build_caption_model(
                config.captioning.backend,
                config.captioning.caption_model,
                usage_tally=usage_tallies["caption"],
            )
            caption_records = caption_video(
                video_path,
                fps=config.captioning.fps,
                model=caption_model,
                context_captions=config.captioning.context_captions,
                utterances=[{"start": r["start"], "speaker": r["speaker"]} for r in tone_records],
                speech_start_offset=config.captioning.speech_frame_offset,
                burst_count=config.captioning.burst_frames,
                burst_spacing=config.captioning.burst_spacing,
                max_dimension=config.captioning.max_dimension,
                temperature=config.sampling.temperature,
                seed=config.sampling.seed,
                checkpoint_path=CAPTIONS_DIR / f"{name}.captions.partial.json",
            )
            write_captions_json(caption_records, name, CAPTIONS_DIR)
            stage.records = len(caption_records)
        flagged = scrub_check("\n".join(record["caption"] for record in caption_records), "captions")
        if flagged:
            pipeline.degraded(
                "caption",
                f"scrub_check flagged {len(flagged)} caption term(s); the screenplay gate decides",
            )
    else:
        source = CAPTIONS_DIR / f"{name}.captions.json"
        caption_records = read_records(source)
        pipeline.resumed("caption", len(caption_records), source.name)

    if start <= STAGES.index("merge"):
        with pipeline.stage("merge") as stage:
            events = merge_screenplay(caption_records, tone_records)
            write_screenplay_json(events, name, SCREENPLAYS_DIR)
            stage.records = len(events)
    else:
        source = SCREENPLAYS_DIR / f"{name}.screenplay.json"
        events = read_records(source)
        pipeline.resumed("merge", len(events), source.name)

    with pipeline.stage("screenplay") as stage:
        template = TEMPLATE_PATH.read_text()
        usage_tallies["screenplay"] = UsageTally()
        screenplay_model = build_screenplay_model(
            config.screenplay.backend,
            config.screenplay.ornith_model,
            unavailable_hint=ORNITH_UNAVAILABLE_HINT,
            usage_tally=usage_tallies["screenplay"],
        )
        screenplay_md = write_screenplay(
            events,
            template,
            model=screenplay_model,
            temperature=config.sampling.temperature,
            seed=config.sampling.seed,
            num_ctx=config.screenplay.num_ctx,
        )
        output_path = write_screenplay_md(screenplay_md, name, SCREENPLAYS_DIR)
        # One Markdown document, not records; the faithfulness check counts
        # what it contains.
        stage.records = None

    provenance = RunProvenance(
        session_id=session_id,
        run_id=run_id,
        started=started,
        finished=datetime.now(UTC).isoformat(timespec="seconds"),
        host=platform.node(),
        input_video_sha256=_sha256(video_path),
        config=config,
        model_digests=_model_digests(config),
        prompt_versions=config.prompt_versions,
        software=_software_versions(),
        stage_timings=stage_timings.rounded(),
        stage_usage={
            stage: StageUsage(
                calls=tally.calls,
                calls_failed=tally.calls_failed,
                calls_reporting=tally.calls_reporting,
                source=tally.source,
                prompt_tokens=tally.prompt_tokens if tally.calls_reporting else None,
                completion_tokens=(
                    tally.completion_tokens if tally.calls_reporting else None
                ),
            )
            for stage, tally in usage_tallies.items()
        },
    )
    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    provenance_path = DELIVERABLES_DIR / f"{name}.provenance.json"
    provenance_path.write_text(provenance.model_dump_json(indent=2))
    logger.info("Wrote run provenance to %s", provenance_path)

    # Enforcing gate on the finished screenplay: findings persist for review
    # either way, and a blocked artifact fails the run loudly (typed error)
    # rather than shipping on a warning. The md exists on disk regardless -
    # the gate governs what may leave the box, not what may be written to it.
    gate_result = evaluate_gate(
        screenplay_md, artifact=f"{name}.screenplay.md", session_id=session_id, run_id=run_id
    )
    write_findings(gate_result)
    write_report(gate_result, DELIVERABLES_DIR)
    pipeline.gate_decided(
        gate_result.report.artifact,
        gate_result.report.resolution,
        sum(gate_result.report.findings_by_category.values()),
    )
    logger.info("Pipeline complete: %s", output_path)
    require_pass(gate_result)


if __name__ == "__main__":
    main()

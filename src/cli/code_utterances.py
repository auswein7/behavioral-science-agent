"""CLI entry point: code a delivered utterance table (DATA_MODEL 4.5).

Run from the repo root:
python -m src.cli.code_utterances <session>.utterances.json [--output-dir DIR]

The input is the 4.1 JSON deliverable, a Tier C artifact that already
passed the scrub gate; this stage adds OSU's codes to it and writes
<session_id>.coded.json (rewritten after every row, so an interrupted run
resumes from it), <session_id>.coded.csv in OSU's flattened column shape,
and <session_id>.coder_provenance.json with the resolved config, codebook
digest, prompt version and per-stage usage. Levers come from the
environment through load_coder_config; nothing here reads os.environ.
"""

import argparse
import json
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from src.adapters import UsageTally
from src.backends import build_coder_llm
from src.coder import (
    FairlibAgentCoder,
    iter_coded_rows,
    placeholder_codebook,
    write_coded_csv,
    write_coded_json,
)
from src.coder.deliverable import DEFAULT_OUTPUT_DIR, read_coded_json
from src.config import load_coder_config, prompt_versions
from src.errors import DeliverableError, PipelineError
from src.schemas import SCHEMA_VERSION, StageUsage, UtteranceRow

logger = logging.getLogger(__name__)


def _load_rows(path: Path) -> tuple[str, list[UtteranceRow]]:
    try:
        data = json.loads(path.read_text())
        rows = [UtteranceRow.model_validate(r) for r in data["rows"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DeliverableError(f"cannot read utterance rows from {path}: {exc}") from exc
    if not rows:
        raise DeliverableError(f"{path} holds no utterance rows")
    return rows[0].document, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("utterances_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    config = load_coder_config()
    codebook = placeholder_codebook()
    if codebook.is_placeholder:
        logger.warning(
            "coding with the PLACEHOLDER codebook (%s): code names only, not "
            "OSU's definitions; the output is a scaffold check, not a result",
            codebook.version,
        )
    session_id, rows = _load_rows(args.utterances_json)

    coded_path = args.output_dir / f"{session_id}.coded.json"
    coded = read_coded_json(coded_path) if coded_path.exists() else []
    if coded:
        logger.info("resuming: %d rows already coded in %s", len(coded), coded_path)

    tally = UsageTally()
    llm = build_coder_llm(
        config.provider,
        config.coder_model,
        temperature=config.temperature,
        seed=config.seed,
        num_ctx=config.num_ctx,
    )
    coder = FairlibAgentCoder(
        llm,
        codebook,
        max_retries=config.max_retries,
        max_steps=config.max_steps,
        usage_tally=tally,
    )
    started = datetime.now(UTC)
    clock = time.monotonic()
    for row in iter_coded_rows(
        rows,
        coder,
        codebook,
        coder_id=config.coder_id,
        prompt_version=prompt_versions()["CODER_ROLE_PROMPT"],
        context_utterances=config.context_utterances,
        already_coded=frozenset(r.uid for r in coded),
    ):
        coded.append(row)
        write_coded_json(coded, session_id, args.output_dir)
        logger.info("coded %s (%d/%d)", row.uid, len(coded), len(rows))
    coded.sort(key=lambda r: r.ord)
    write_coded_json(coded, session_id, args.output_dir)
    csv_path = write_coded_csv(coded, session_id, codebook.code_names, args.output_dir)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "started": started.isoformat(),
        "finished": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "config": config.model_dump(),
        "codebook_version": codebook.version,
        "codebook_digest": codebook.digest(),
        "prompt_versions": {"CODER_ROLE_PROMPT": prompt_versions()["CODER_ROLE_PROMPT"]},
        "coder_model": coder.model_name,
        "stage_timings": {"coder": round(time.monotonic() - clock, 2)},
        "stage_usage": {
            "coder": StageUsage(
                calls=tally.calls,
                calls_failed=tally.calls_failed,
                calls_reporting=tally.calls_reporting,
                source=tally.source,
                prompt_tokens=tally.prompt_tokens if tally.calls_reporting else None,
                completion_tokens=tally.completion_tokens if tally.calls_reporting else None,
            ).model_dump()
        },
    }
    provenance_path = args.output_dir / f"{session_id}.coder_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2))
    logger.info("wrote %s and %s", csv_path, provenance_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)

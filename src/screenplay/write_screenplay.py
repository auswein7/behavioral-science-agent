"""Turn a merged screenplay JSON timeline into a formatted screenplay .md, using a locally-run
Ornith model served by Ollama as the orchestrator/screenwriter.
"""

import json
import logging
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ornith-1.5-255k"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "screenplays"

SYSTEM_PROMPT = (
    "You are a professional screenwriter. You are given a JSON timeline of a video, made up "
    "of 'visual' events (what's on screen, at a single timestamp) and 'speech' events "
    "(transcribed dialogue, with a speaker label and a start/end time range). Combine these "
    "into a single, well-formatted screenplay in Markdown, following the structure and "
    "conventions of the template you are given exactly. Every 'speech' event's text must "
    "appear as a dialogue line, in order, with its timestamp. Do not invent dialogue, "
    "speakers, or events that are not in the source JSON. Output only the finished Markdown "
    "screenplay — no commentary before or after it."
)


def _ensure_model_available(model_name: str) -> None:
    try:
        local_models = {m.model for m in ollama.list().models}
    except ConnectionError as e:
        raise ConnectionError(
            "Could not reach the local Ollama server. Make sure Ollama is installed "
            "and running (https://ollama.com/download), then try again."
        ) from e

    if any(name == model_name or name.startswith(f"{model_name}:") for name in local_models):
        return

    raise ValueError(
        f"Model '{model_name}' was not found in your local Ollama models "
        f"(`ollama list`). Ornith is not on the public Ollama registry, so it can't be "
        "pulled automatically — make sure it's been created locally before running this."
    )


def write_screenplay(events: list[dict], template: str, model_name: str = DEFAULT_MODEL) -> str:
    """Prompt the orchestrator model to format a timeline of events into a screenplay .md,
    matching the structure of the given template. Returns the generated Markdown text.
    """
    _ensure_model_available(model_name)

    user_prompt = (
        "TEMPLATE (match this structure and formatting exactly):\n\n"
        f"{template}\n\n"
        "---\n\n"
        "SOURCE JSON TIMELINE (combine every event from this into the screenplay above):\n\n"
        f"{json.dumps(events, indent=2)}"
    )

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    screenplay_md = response.message.content.strip()
    logger.info("Generated screenplay (%d chars) with model %s", len(screenplay_md), model_name)
    return screenplay_md


def write_md(screenplay_md: str, name: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.screenplay.md"
    output_path.write_text(screenplay_md)
    logger.info("Wrote screenplay to %s", output_path)
    return output_path

"""Turn a merged screenplay JSON timeline into a formatted screenplay .md, using a locally-run
Ornith model served by Ollama as the orchestrator/screenwriter.
"""

import json
import logging
from pathlib import Path

import ollama

from src.errors import ConfigurationError
from src.prompts import SCREENPLAY_SYSTEM_PROMPT
from src.reliability import call_with_retry

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ornith-1.5-255k"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "screenplays"

SYSTEM_PROMPT = SCREENPLAY_SYSTEM_PROMPT


def _ensure_model_available(model_name: str) -> None:
    try:
        local_models = {m.model for m in ollama.list().models}
    except ConnectionError as e:
        raise ConfigurationError(
            "Could not reach the local Ollama server. Make sure Ollama is installed "
            "and running (https://ollama.com/download), then try again."
        ) from e

    if any(name == model_name or name.startswith(f"{model_name}:") for name in local_models):
        return

    raise ConfigurationError(
        f"Model '{model_name}' was not found in your local Ollama models "
        f"(`ollama list`). Ornith is not on the public Ollama registry, so it can't be "
        "pulled automatically — make sure it's been created locally before running this."
    )


def write_screenplay(
    events: list[dict],
    template: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    seed: int | None = None,
    num_ctx: int | None = None,
) -> str:
    """Prompt the orchestrator model to format a timeline of events into a screenplay .md,
    matching the structure of the given template. Returns the generated Markdown text.

    Sampling is deterministic by default (temperature 0; design principle 8) and the
    options actually sent to Ollama are exactly what the run's provenance records.
    """
    _ensure_model_available(model_name)

    options: dict = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    user_prompt = (
        "TEMPLATE (match this structure and formatting exactly):\n\n"
        f"{template}\n\n"
        "---\n\n"
        "SOURCE JSON TIMELINE (combine every event from this into the screenplay above):\n\n"
        f"{json.dumps(events, indent=2)}"
    )

    response = call_with_retry(
        lambda: ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options=options,
        ),
        description=f"screenplay generation with {model_name}",
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

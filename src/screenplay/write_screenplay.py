"""Turn a merged screenplay JSON timeline into a formatted screenplay .md, using a locally-run
Ornith model served by Ollama as the orchestrator/screenwriter.
"""

import json
import logging
from pathlib import Path

from src.adapters import AbstractChatModel, ChatMessage, OllamaChatModel
from src.errors import AdapterError
from src.prompts import SCREENPLAY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ornith-1.5-255k"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "screenplays"

# The screenplay's H1 line is fixed, not written by the model. A generated
# title is content the artifact does not need, it defeats the likely_name
# gate check by putting title-case words on a heading line, and for a real
# session a descriptive title summarizes what happened, which the leak
# taxonomy counts as a re-identification channel. The artifact's identity
# lives in its filename and its provenance record, not in its heading; no
# identifier is interpolated here because the obvious candidate, the video
# filename stem, can itself carry a name.
SCREENPLAY_TITLE = "Screenplay"
TITLE_LINE = f"# {SCREENPLAY_TITLE}"

SYSTEM_PROMPT = SCREENPLAY_SYSTEM_PROMPT

ORNITH_UNAVAILABLE_HINT = (
    "Ornith is not on the public Ollama registry, so it can't be pulled "
    "automatically - make sure it's been created locally before running this."
)


def write_screenplay(
    events: list[dict],
    template: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    seed: int | None = None,
    num_ctx: int | None = None,
    model: AbstractChatModel | None = None,
) -> str:
    """Prompt the orchestrator model to format a timeline of events into a screenplay .md,
    matching the structure of the given template. Returns the generated Markdown text.

    Sampling is deterministic by default (temperature 0; design principle 8) and the
    options actually sent to Ollama are exactly what the run's provenance records.

    Pass `model` to run the stage against any AbstractChatModel backend; when
    None, a local OllamaChatModel for `model_name` is constructed as before.
    The stage never learns which backend it got (design principle 2).
    """
    if model is None:
        model = OllamaChatModel(
            model_name,
            auto_pull=False,
            description="screenplay generation",
            unavailable_hint=ORNITH_UNAVAILABLE_HINT,
        )
    model.ensure_available()

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

    response = model.invoke(
        [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ],
        **options,
    )
    screenplay_md = response.content.strip()
    if not screenplay_md:
        # A reasoning model that runs out of context mid-think returns 200 OK with
        # empty content and a truncated stop reason - a silent empty string here
        # once produced a 0-char screenplay that sailed through the gate.
        # Principle 6: a degraded result is a typed error, never a swallowed
        # empty value. The branch reads the seam's normalized truncated signal,
        # never the provider's raw reason string (principle 1); the raw value
        # still rides along in the message for the run record.
        detail = (
            f"(done_reason={response.done_reason!r}, "
            f"prompt_eval={response.prompt_eval_count}, "
            f"eval={response.eval_count})"
        )
        if response.truncated:
            remedy = (
                "the prompt plus the model's thinking overflowed num_ctx - "
                "raise ORNITH_NUM_CTX or shorten the caption stage's output"
            )
        elif response.truncated is None:
            remedy = (
                "this backend does not report stop reasons, so overflow cannot "
                "be distinguished from other failures - if output stays empty, "
                "try raising ORNITH_NUM_CTX or use a backend that reports them"
            )
        else:
            remedy = "the model stopped without a context overflow; check the model and prompt"
        raise AdapterError(
            f"{response.model} returned an empty screenplay {detail}; {remedy}"
        )
    screenplay_md = enforce_title(screenplay_md)
    logger.info("Generated screenplay (%d chars) with model %s", len(screenplay_md), model_name)
    return screenplay_md


def enforce_title(screenplay_md: str) -> str:
    """Replace the generated H1 line with the fixed title, or insert it when
    the model wrote no heading at all.

    The prompt and the template both ask for this title, but a gate must not
    depend on a model complying (principle 6): prompt compliance is
    probabilistic and this rewrite is not. Only the first H1 is touched, so
    scene headings are left alone.
    """
    lines = screenplay_md.split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line.startswith("# "):
            if line != TITLE_LINE:
                logger.info("Replaced generated title %r with %r", line, TITLE_LINE)
                lines[index] = TITLE_LINE
            return "\n".join(lines)
        break
    logger.info("Model wrote no title heading; inserting %r", TITLE_LINE)
    return f"{TITLE_LINE}\n\n{screenplay_md}"


def write_md(screenplay_md: str, name: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.screenplay.md"
    output_path.write_text(screenplay_md)
    logger.info("Wrote screenplay to %s", output_path)
    return output_path

"""Caption an image using Qwen3-VL, served locally by Ollama."""

import logging
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3-vl:8b"
DEFAULT_PROMPT = "Describe what is happening in this image."


def _ensure_model_pulled(model_name: str) -> None:
    try:
        local_models = {m.model for m in ollama.list().models}
    except ConnectionError as e:
        raise ConnectionError(
            "Could not reach the local Ollama server. Make sure Ollama is installed "
            "and running (https://ollama.com/download), then try again."
        ) from e

    if any(name == model_name or name.startswith(f"{model_name}:") for name in local_models):
        return

    logger.info("Model %s not found locally, pulling via Ollama...", model_name)
    ollama.pull(model_name)


def caption_image(
    image_path: Path,
    prompt: str = DEFAULT_PROMPT,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Generate a text caption for an image using a locally-run Qwen3-VL model served by Ollama.

    Requires the Ollama app (https://ollama.com/download) to be installed and running.
    Pulls the model automatically on first use if it isn't already present locally.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    _ensure_model_pulled(model_name)

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
    )
    caption = response.message.content.strip()
    logger.info("Captioned %s: %s", image_path.name, caption)
    return caption

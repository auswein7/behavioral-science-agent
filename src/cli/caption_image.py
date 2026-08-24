"""Quick demo of the captioning system: caption a sample image with Qwen3-VL.

Run from the repo root: python -m src.cli.caption_image
"""

import logging
from pathlib import Path

from src.captioning import caption_image

logging.basicConfig(level=logging.INFO)

IMAGE_PATH = Path(__file__).resolve().parents[2] / "data" / "playing_catch.jpg"


def main():
    caption = caption_image(IMAGE_PATH)
    print(f"\nCaption: {caption}\n")


if __name__ == "__main__":
    main()

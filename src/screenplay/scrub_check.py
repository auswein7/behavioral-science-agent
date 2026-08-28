"""Warn-only mechanical scan for likely de-identification leaks in generated text.

Prompt-engineered anonymization (src/prompts.py's ANONYMIZATION_RULES) is probabilistic —
a model can still slip and mention an appearance detail, a gender, or a name despite being
told not to. This is a cheap, non-LLM backstop: a regex spot-check over already-generated
text, meant to flag likely slips for a human to review. It is NOT a guarantee and NOT a
filter — it never raises, blocks, or modifies the text it scans. The "likely name" check in
particular is a blunt heuristic (any capitalized word outside a small allowlist) and will
have false positives (title/heading words, sentence-initial words it fails to catch);
treat its silence as "nothing obvious", not as proof of a clean transcript.
"""

import logging
import re

logger = logging.getLogger(__name__)

_GENDERED_WORDS_RE = re.compile(
    r"\b(he|she|him|her|his|hers|man|woman|men|women|boy|girl|guy|lady|gentleman)\b",
    re.IGNORECASE,
)

_APPEARANCE_WORDS_RE = re.compile(
    r"\b(uniform|wearing|wears|dressed|shirt|dress|jacket|coat|hair|hat|haircut|hairstyle|"
    r"blonde|brunette|redhead|bald|beard|mustache)\b",
    re.IGNORECASE,
)

_SPEAKER_TAG_RE = re.compile(r"SPEAKER_\d+")
_CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z]*\b")
# A capitalized word right after a sentence boundary (start of text, ".", "!", "?", or a
# blank line) is excluded from the "likely name" check below — ordinary sentence-initial
# words can't be told apart from names by regex alone, so this trades missing some real
# leaks at sentence starts for far fewer false positives everywhere else.
_SENTENCE_INITIAL_RE = re.compile(r"(?:\A|[.!?]\s+|\n\s*\n)\s*([A-Z][a-zA-Z]*)")
_NAME_ALLOWLIST = {"scene", "speaker"}


def _likely_names(text: str) -> list[str]:
    sentence_initial = {m.group(1) for m in _SENTENCE_INITIAL_RE.finditer(text)}
    speaker_tags = set(_SPEAKER_TAG_RE.findall(text))

    hits = set()
    for match in _CAPITALIZED_WORD_RE.finditer(text):
        word = match.group(0)
        if word in sentence_initial or word in speaker_tags or word.lower() in _NAME_ALLOWLIST:
            continue
        hits.add(word)
    return sorted(hits)


def scrub_check(text: str, label: str) -> list[str]:
    """Scan `text` for likely anonymization leaks and log a warning per category found.

    `label` identifies the text's origin in the log (e.g. "captions" or "screenplay.md")
    so a leak's source stage is clear. Never raises, blocks, or modifies `text` — this is
    a spot-check aid for human review, not an enforcement mechanism. Returns the list of
    warning messages logged, for callers that also want to persist/display them.
    """
    warnings = []

    gendered = sorted({m.group(0).lower() for m in _GENDERED_WORDS_RE.finditer(text)})
    if gendered:
        warnings.append(f"[{label}] possible gendered language: {', '.join(gendered)}")

    appearance = sorted({m.group(0).lower() for m in _APPEARANCE_WORDS_RE.finditer(text)})
    if appearance:
        warnings.append(f"[{label}] possible appearance description: {', '.join(appearance)}")

    names = _likely_names(text)
    if names:
        preview = ", ".join(names[:10])
        warnings.append(f"[{label}] possible proper names (best-effort, expect false positives): {preview}")

    for warning in warnings:
        logger.warning(warning)

    return warnings

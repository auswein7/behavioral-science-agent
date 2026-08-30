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

from src.scrub.patterns import iter_findings

logger = logging.getLogger(__name__)


def scrub_check(text: str, label: str) -> list[str]:
    """Scan `text` for likely anonymization leaks and log a warning per category found.

    `label` identifies the text's origin in the log (e.g. "captions" or "screenplay.md")
    so a leak's source stage is clear. Never raises, blocks, or modifies `text` — this is
    a spot-check aid for human review, not an enforcement mechanism. Returns the list of
    warning messages logged, for callers that also want to persist/display them.
    """
    warnings = []

    gendered: set[str] = set()
    appearance: set[str] = set()
    names: set[str] = set()
    for category, term, _position in iter_findings(text):
        if category == "gendered":
            gendered.add(term.lower())
        elif category == "appearance":
            appearance.add(term.lower())
        else:
            names.add(term)

    if gendered:
        warnings.append(f"[{label}] possible gendered language: {', '.join(sorted(gendered))}")
    if appearance:
        warnings.append(f"[{label}] possible appearance description: {', '.join(sorted(appearance))}")
    if names:
        preview = ", ".join(sorted(names)[:10])
        warnings.append(f"[{label}] possible proper names (best-effort, expect false positives): {preview}")

    for warning in warnings:
        logger.warning(warning)

    return warnings

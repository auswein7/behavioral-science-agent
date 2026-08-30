"""Shared leak-detection patterns: single source of truth for the warn-only
scrub_check and the enforcing gate. Regex only, no models."""

import re
from collections.abc import Iterator

GENDERED_RE = re.compile(
    r"\b(he|she|him|her|his|hers|man|woman|men|women|boy|girl|guy|lady|gentleman)\b",
    re.IGNORECASE,
)

APPEARANCE_RE = re.compile(
    r"\b(uniform|wearing|wears|dressed|shirt|dress|jacket|coat|hair|hat|haircut|hairstyle|"
    r"blonde|brunette|redhead|bald|beard|mustache)\b",
    re.IGNORECASE,
)

SPEAKER_TAG_RE = re.compile(r"SPEAKER_\d+")
CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z]*\b")
# A capitalized word right after a sentence boundary (start of text, ".", "!", "?", or a
# blank line) is excluded from the "likely name" check - ordinary sentence-initial words
# can't be told apart from names by regex alone, so this trades missing some real leaks
# at sentence starts for far fewer false positives everywhere else.
SENTENCE_INITIAL_RE = re.compile(r"(?:\A|[.!?]\s+|\n\s*\n)\s*([A-Z][a-zA-Z]*)")
# "na" is the OSU prior-field convention in delivered tables; the last four are
# the name-scrub's own replacement tokens, which must not re-trigger the gate.
NAME_ALLOWLIST = {"scene", "speaker", "na", "name", "org", "place", "group"}


def iter_findings(text: str) -> Iterator[tuple[str, str, int]]:
    """Yield (category, matched term, character position) for every individual
    match: "gendered", "appearance", and "likely_name" (capitalized words with
    the sentence-initial/speaker-tag/allowlist exclusions applied)."""
    for match in GENDERED_RE.finditer(text):
        yield "gendered", match.group(0), match.start()
    for match in APPEARANCE_RE.finditer(text):
        yield "appearance", match.group(0), match.start()

    sentence_initial = {m.group(1) for m in SENTENCE_INITIAL_RE.finditer(text)}
    speaker_tags = set(SPEAKER_TAG_RE.findall(text))
    for match in CAPITALIZED_WORD_RE.finditer(text):
        word = match.group(0)
        if word in sentence_initial or word in speaker_tags or word.lower() in NAME_ALLOWLIST:
            continue
        yield "likely_name", word, match.start()

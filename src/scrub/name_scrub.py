"""Deterministic spoken-name scrub: NER over transcript text, names replaced
before any downstream stage (Ornith, the utterance table) sees the dialogue.

This is the deterministic half of spoken-name handling (DATA_MODEL 4.1's
"spoken names replaced"); Ornith's prompt-level best-effort substitution
remains as a probabilistic second layer, never the primary one. Uses spaCy's
small English model - a fixed local model with deterministic inference, not an
LLM, so it sits with emotion2vec in the sanctioned non-Ollama exceptions.

A spoken name cannot be reliably mapped to the SPEAKER_NN who bears it (the
name usually refers to someone other than the current speaker), so every
detected person name is replaced with the neutral token "[NAME]" rather than
a guessed speaker tag.

The default scope is wider than PERSON alone, for two reasons observed on run
1: the small model can mistag a person name as ORG (it reads "Trump" as ORG in
a garbled sentence), and location/organization/group cues are themselves leak
channels in the draft taxonomy (TODO 2.1 - narrow the map there if the team
decides content preservation wins for some channels). Each entity class gets
its own token so coding-relevant sentence structure survives.
"""

import logging
from pathlib import Path

from src.errors import ConfigurationError
from src.persist import write_records
from src.schemas import NameReplacement

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "transcripts"

DEFAULT_ENTITY_TOKENS = {
    "PERSON": "[NAME]",
    "ORG": "[ORG]",
    "GPE": "[PLACE]",
    "LOC": "[PLACE]",
    "FAC": "[PLACE]",
    "NORP": "[GROUP]",
}
SPACY_MODEL = "en_core_web_sm"

_nlp = None


def _load_model():
    global _nlp
    if _nlp is None:
        try:
            import spacy
        except ImportError as exc:
            raise ConfigurationError(
                "spacy is not installed; it is required for the spoken-name scrub"
            ) from exc
        try:
            _nlp = spacy.load(SPACY_MODEL, disable=["parser", "lemmatizer"])
        except OSError as exc:
            raise ConfigurationError(
                f"spaCy model '{SPACY_MODEL}' is not installed - run: "
                f"python -m spacy download {SPACY_MODEL}"
            ) from exc
    return _nlp


def scrub_names(
    records: list[dict],
    entity_tokens: dict[str, str] = DEFAULT_ENTITY_TOKENS,
) -> tuple[list[dict], list[NameReplacement]]:
    """Return a copy of transcript records with named entities in "text" replaced
    by class tokens, plus the Tier B replacement log (which does contain the names).

    Input records are not mutated. Records are matched to replacements by list
    position (record_index). Only "text" is scrubbed; low_confidence_words and
    every other field stay verbatim - they are Tier B and never delivered.
    """
    nlp = _load_model()
    scrubbed: list[dict] = []
    replacements: list[NameReplacement] = []

    for i, record in enumerate(records):
        text = record["text"]
        doc = nlp(text)
        spans = [e for e in doc.ents if e.label_ in entity_tokens]
        if spans:
            # Replace back-to-front so earlier character offsets stay valid.
            new_text = text
            for ent in sorted(spans, key=lambda e: e.start_char, reverse=True):
                token = entity_tokens[ent.label_]
                new_text = new_text[: ent.start_char] + token + new_text[ent.end_char :]
                replacements.append(
                    NameReplacement(record_index=i, entity=ent.text, label=ent.label_, replacement=token)
                )
            scrubbed.append({**record, "text": new_text})
        else:
            scrubbed.append(dict(record))

    if replacements:
        logger.info("Name scrub replaced %d spans across %d records",
                    len(replacements), len({r.record_index for r in replacements}))
    return scrubbed, replacements


def write_json(
    records: list[dict],
    replacements: list[NameReplacement],
    name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Persist the scrubbed transcript next to the verbatim one, plus the
    replacement log. Both are Tier B files under data/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.formatted.tone.scrubbed.json"
    write_records(records, output_path)
    log_path = output_dir / f"{name}.name_replacements.json"
    write_records([r.model_dump() for r in replacements], log_path)
    logger.info("Wrote %d scrubbed records to %s (%d replacements logged)",
                len(records), output_path, len(replacements))
    return output_path

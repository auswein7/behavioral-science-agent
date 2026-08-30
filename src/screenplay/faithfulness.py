"""Mechanical faithfulness check of a screenplay .md against its merged events.

Ornith is generative, so its output must be verified, not trusted: every speech
event appears exactly once with its timestamp and speaker preserved, action
prose attributes behavior only to speakers the source data named, and no
reasoning-model think text leaks into the artifact. Pure text comparison, no
models - the point is that this check cannot itself hallucinate.
"""

import re
from dataclasses import dataclass, field

DIALOGUE_RE = re.compile(r"^\*\*(SPEAKER_\d+)\*\* \*\((\d{2}:\d{2}:\d{2}\.\d{3})\)\*", re.MULTILINE)
SPEAKER_TAG_RE = re.compile(r"SPEAKER_\d+")
THINK_RE = re.compile(r"</?think>", re.IGNORECASE)


@dataclass(frozen=True)
class FaithfulnessReport:
    total_speech_events: int
    matched: int
    missing: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    extra_dialogue: list[str] = field(default_factory=list)
    speaker_mismatches: list[str] = field(default_factory=list)
    invented_action_tags: list[str] = field(default_factory=list)
    think_leak: bool = False

    @property
    def ok(self) -> bool:
        return not (
            self.missing
            or self.duplicated
            or self.extra_dialogue
            or self.speaker_mismatches
            or self.invented_action_tags
            or self.think_leak
        )

    def summary_lines(self) -> list[str]:
        lines = [f"speech events: {self.total_speech_events}, matched in md: {self.matched}"]
        for label, items in (
            ("missing from md", self.missing),
            ("duplicated in md", self.duplicated),
            ("extra dialogue in md", self.extra_dialogue),
            ("speaker mismatches", self.speaker_mismatches),
            ("invented action attributions", self.invented_action_tags),
        ):
            if items:
                lines.append(f"{label} ({len(items)}):")
                lines.extend(f"  {item}" for item in items)
        if self.think_leak:
            lines.append("think-text leak: <think> markup present in md")
        return lines


def check_faithfulness(screenplay_md: str, events: list[dict]) -> FaithfulnessReport:
    """Compare the rendered screenplay against the merged timeline it was
    generated from. `events` is merge_screenplay output (speech + visual)."""
    speech = [e for e in events if e.get("type") == "speech"]
    visual = [e for e in events if e.get("type") == "visual"]

    md_dialogue = [(m.group(1), m.group(2)) for m in DIALOGUE_RE.finditer(screenplay_md)]
    md_by_hms: dict[str, list[str]] = {}
    for speaker, hms in md_dialogue:
        md_by_hms.setdefault(hms, []).append(speaker)

    missing, duplicated, mismatches = [], [], []
    matched = 0
    event_hms = set()
    for event in speech:
        hms = event["start_hms"]
        event_hms.add(hms)
        rendered = md_by_hms.get(hms, [])
        label = f"{hms} {event['speaker']}: {event['text'][:60]}"
        if not rendered:
            missing.append(label)
        else:
            matched += 1
            if len(rendered) > 1:
                duplicated.append(label)
            if event["speaker"] not in rendered:
                mismatches.append(f"{label} (md says {', '.join(rendered)})")

    extra = [f"{hms} {speakers[0]}" for hms, speakers in md_by_hms.items() if hms not in event_hms]

    # Action prose may attribute behavior only to speakers grounded in the
    # source: named in a caption's mentioned_speakers or present as a speech
    # event's speaker. Dialogue/heading lines are excluded from the scan.
    allowed = {e["speaker"] for e in speech}
    for event in visual:
        allowed.update(event.get("mentioned_speakers", []))
    action_text = "\n".join(
        line for line in screenplay_md.splitlines()
        if not line.startswith("**SPEAKER") and not line.startswith(">")
    )
    invented = sorted({tag for tag in SPEAKER_TAG_RE.findall(action_text) if tag not in allowed})

    return FaithfulnessReport(
        total_speech_events=len(speech),
        matched=matched,
        missing=missing,
        duplicated=duplicated,
        extra_dialogue=sorted(extra),
        speaker_mismatches=mismatches,
        invented_action_tags=invented,
        think_leak=bool(THINK_RE.search(screenplay_md)),
    )

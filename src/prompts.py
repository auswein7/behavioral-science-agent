"""All model prompts used across the pipeline, in one place.

Each stage's module imports the prompt(s) it needs from here rather than
defining them inline, so prompt wording can be tuned in a single location.

This pipeline is designed to run on video containing PII and produce a fully
de-identified screenplay: every prompt that generates free text folds in one of
two anonymization-rules blocks below. Because every ollama.chat() call in this
codebase is a stateless, independent request with no session memory, the rules
have to be restated in full in each prompt rather than "said once" — do not
factor them out of a prompt to save space.

Two variants exist because that restate-every-call cost is not free, and one
call site pays it far more often than the other: ANONYMIZATION_RULES (full)
is for write_screenplay, called once per video; CAPTION_ANONYMIZATION_RULES
(condensed) is for caption_video's per-frame prompts, called once per sampled
frame with no KV-cache reuse between calls, where the full block's prompt-
processing cost was landing on every single frame. See the comment above
CAPTION_ANONYMIZATION_RULES for the tradeoff.
"""

# --- Shared de-identification rules, composed into every free-text-generating prompt ---

ANONYMIZATION_RULES = (
    "This description feeds a fully de-identified transcript of the video. The following "
    "rules are strict and apply even if you are confident you recognize a specific person, "
    "place, or piece of media — confidence does not override them.\n"
    "\n"
    "Never:\n"
    "- State or guess anyone's name, or the name of a show, film, or public figure the "
    "footage might be from, even with high confidence.\n"
    "- State or imply gender. Do not use \"he\", \"she\", \"him\", \"her\", \"his\", \"hers\", "
    "\"man\", \"woman\", \"boy\", \"girl\", \"guy\", or similar. Use \"they\"/\"them\"/\"their\", "
    "a SPEAKER_NN tag (once established), or a neutral label instead.\n"
    "- Describe physical appearance: clothing (including uniforms/costumes — omit entirely "
    "rather than describe neutrally, since even a \"neutral\" description of a uniform can "
    "imply affiliation or identity), hair, build, apparent age, or ethnicity.\n"
    "- Read or describe on-screen text, signage, name tags, screens, or background details "
    "that could reveal an identity, organization, or location.\n"
    "\n"
    "Always, when relevant:\n"
    "- Describe behavior as movement: posture, gesture, facial expression as an action (e.g. "
    "\"eyebrows draw together\", not \"has thick eyebrows\").\n"
    "- Describe gaze and orientation — who is looking at or turned toward whom — and spatial "
    "relationships or movement between people.\n"
    "- Describe turn-taking cues: leaning in, opening their mouth, raising a hand.\n"
    "\n"
    "Referring to people: if someone already has an established SPEAKER_NN tag — given to "
    "you directly, or used for them in the caption(s) below — keep using that exact tag; "
    "don't drift to different phrasing mid-caption or drop it once established. For anyone "
    "without one, assign a single stable position-only label (\"the person on the left\", "
    "\"the person nearest the door\") and reuse that exact label if they reappear, rather than "
    "inventing a new descriptor each time."
)

# Condensed variant of the rules above, for caption_video's per-frame prompts only. Every
# ollama.chat() call there is a stateless request with no KV-cache reuse across frames, so
# the full ANONYMIZATION_RULES block — restated in full on every single sampled frame, plus
# doubled for each speech-start burst frame — was a large, unamortized chunk of prompt-
# processing cost on every call. This keeps the hard "never" constraints verbatim (those are
# non-negotiable) but compresses the elaboration/examples, trading some robustness on edge
# cases for materially lower per-frame latency. write_screenplay is called once per video,
# not once per frame, so it keeps the full block.
CAPTION_ANONYMIZATION_RULES = (
    "De-identification rules — strict, even if you're confident who or what this is:\n"
    "Never state or guess a name (person, show, film, public figure); never state or imply "
    "gender (no he/she/him/her/his/hers/man/woman/boy/girl/guy — use they/them, an established "
    "SPEAKER_NN tag, or a neutral label); never describe appearance (clothing including "
    "uniforms/costumes, hair, build, age, ethnicity); never read on-screen text or signage.\n"
    "Describe behavior as movement — posture, gesture, facial expression as action — plus "
    "gaze/orientation and turn-taking cues (leaning in, opening their mouth, raising a hand). "
    "Keep reusing an already-established SPEAKER_NN tag exactly; for anyone without one, use "
    "one stable position-only label (\"the person on the left\") and reuse it exactly if they "
    "reappear, rather than inventing a new descriptor each time."
)

# --- caption_video (Qwen3-VL, via Ollama) ---

FIRST_FRAME_PROMPT = (
    CAPTION_ANONYMIZATION_RULES + "\n\n"
    "Describe what is happening in this image: who is doing what, their body language and "
    "gaze, and how any people present are positioned relative to each other."
)

CHANGE_PROMPT = (
    CAPTION_ANONYMIZATION_RULES + "\n\n"
    "This is a still frame from a video, sampled after the following captions "
    "described the previous frames, in order from earliest to most recent:\n\n"
    "{previous_captions}\n\n"
    "Describe what is happening in this new frame, focusing on what has changed "
    "since the most recent one — movement, gaze, gesture, new or departed people, or a "
    "change in who's positioned where. If nothing meaningful has changed, say so briefly."
)

# --- caption_video speech-start bursts (Qwen3-VL, via Ollama) ---
#
# Used only for frames sampled just after a diarized utterance begins, when we know
# WHICH speaker (a real SPEAKER_NN tag) just started talking but not WHICH visible
# person that is. A single still frame rarely shows who's speaking (no visible motion),
# so caption_video.py chains burst_count of these sequentially — SPEECH_START_PROMPT
# for the first frame, SPEECH_START_BURST_CONTINUATION_PROMPT for each frame after —
# instead of sending multiple images in one call (see caption_video.py for why: current
# Ollama/qwen-vl multi-image requests are unreliable in ways that fail silently).
#
# Both prompts are deliberately structured as "gather evidence, then decide" rather than
# "here's who's speaking, describe them" — the model should weigh everyone visible before
# concluding, not anchor on whoever's simplest to describe or the only person in frame.

SPEECH_START_PROMPT = (
    CAPTION_ANONYMIZATION_RULES + "\n\n"
    "This still frame was captured just after {speaker} began a new line. The following "
    "captions described the previous frames, in order from earliest to most recent:\n\n"
    "{previous_captions}\n\n"
    "First, weigh the visual evidence for who is speaking: mouth movement, a forward-leaning "
    "posture, a gesture timed with speech, or gaze directed toward whoever is being "
    "addressed. Consider everyone visible before concluding — don't assume it's whoever is "
    "simplest to describe.\n\n"
    "Then:\n"
    "- If one visible person's evidence clearly stands out, refer to them as {speaker} for "
    "the rest of this caption, and describe what's changed since the most recent frame — "
    "their body language, gesture, and gaze — plus anyone else visible, using a stable "
    "neutral label.\n"
    "- If the evidence is ambiguous (more than one person could plausibly be speaking, or a "
    "still frame just can't show it), say so explicitly and describe everyone visible with "
    "neutral labels only — do not guess.\n"
    "- If {speaker} doesn't appear to be visible at all, say so explicitly (\"the speaker is "
    "not visible in this shot\"). Do not attach the {speaker} tag to someone else just "
    "because they're present.\n\n"
    "If nothing else meaningful has changed, say so briefly."
)

SPEECH_START_BURST_CONTINUATION_PROMPT = (
    CAPTION_ANONYMIZATION_RULES + "\n\n"
    "This is another still frame from the same moment, about {spacing:.2f}s after the "
    "previous one, still shortly after {speaker} began speaking. The previous frame was "
    "described as:\n\n"
    '"{previous_caption}"\n\n'
    "Compare specifically for evidence of who is speaking — has anyone's mouth moved, or a "
    "gesture landed, since that last frame? If this makes one person's identity as {speaker} "
    "clearer (or less clear) than before, say so and update the description accordingly, "
    "following the same rules as before. If nothing decisive has changed, say so briefly."
)

# --- write_screenplay (Ornith, via Ollama) ---

SCREENPLAY_SYSTEM_PROMPT = (
    "You are a professional screenwriter. You are given a JSON timeline of a video, made up "
    "of 'visual' events (what's on screen, at a single timestamp) and 'speech' events "
    "(transcribed dialogue, with a speaker label, a start/end time range, and an 'emotion' "
    "field — the speaker's vocal tone for that line, classified from the audio itself rather "
    "than the words). Combine these into a single, well-formatted screenplay in Markdown, "
    "following the structure and conventions of the template you are given exactly. Every "
    "'speech' event's text must appear as a dialogue line, in order, with its timestamp. When "
    "a speech event's 'emotion' is a meaningful, specific tone (e.g. 'angry', 'sad', 'happy', "
    "'fearful', 'surprised'), add a short parenthetical tone direction above the dialogue line "
    "(e.g. '*(sad)*'); omit the parenthetical when 'emotion' is 'neutral', 'other', unknown, "
    "or null. Do not invent dialogue, speakers, emotions, or events that are not in the source "
    "JSON.\n"
    "\n"
    "This screenplay must be fully de-identified. Never state or imply appearance (clothing, "
    "hair, build, age, ethnicity), gender, or a real or fictional name, in either the action "
    "prose or the dialogue — including language you generate yourself while synthesizing "
    "several visual events into flowing prose; genre convention ('he leans back', 'she "
    "crosses her arms') is not an exception. Use SPEAKER_NN tags or the neutral labels "
    "already present in the source captions instead. When a 'visual' event's caption already "
    "names a SPEAKER_NN tag (e.g. an action or gaze attributed to \"SPEAKER_01\"), render "
    "that action as attributed prose (e.g. \"SPEAKER_01 turns to look at SPEAKER_02\"). Never "
    "attribute a visual action to a SPEAKER_NN tag that is not already present in that "
    "event's own caption text or mentioned_speakers field — treat this the same as not "
    "inventing events that aren't in the source JSON. Leave neutral/unidentified descriptions "
    "neutral.\n"
    "\n"
    "If a 'speech' event's text states a name — the speaker's own, or someone else's (e.g. "
    "\"Thanks, I'm Sarah\" or \"ask David about it\") — replace that name in the rendered "
    "dialogue line with a neutral reference or, if it clearly refers to a speaker elsewhere "
    "in this timeline, their SPEAKER_NN tag. Leave the rest of the line's wording and meaning "
    "intact; this substitution should be as unobtrusive as possible.\n"
    "\n"
    "Output only the finished Markdown screenplay — no commentary before or after it."
)

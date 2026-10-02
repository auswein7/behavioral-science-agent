# What the OSU team receives from the FAIR Lab pipeline

Draft for discussion, September 2026.

## Per session, four files

Each file is named by an opaque session id (for example `s017`). A file is
released only after it passes an automated de-identification check and a
study-team member has read it.

1. `<session_id>.utterances.csv` - the analysis file. One row per utterance:
   `document, uid, ord, speaker, utterance, time, end_time, filename,
   prior_utterance, prior_speaker, role, tone, low_confidence, nonverbal_notes`.
   The first columns follow your coded-transcript format (`uid` keeps the
   `u001` convention; the first row's prior fields are `NA`). The added columns
   are the speaker's role, a vocal-tone label, a transcription-confidence flag,
   and de-identified notes on non-verbal behavior. Files are UTF-8. Times are
   quoted `HH:MM:SS.mmm` text, so import them as text to stop spreadsheets
   converting them.
2. `<session_id>.screenplay.md` - the same session as a readable screenplay
   (scenes, action lines, tone directions, timestamped dialogue), for reading
   and spot-checking.
3. `<session_id>.provenance.json` - the processing record: model versions,
   prompt versions and settings. It contains no session content.
4. `<session_id>.scrub_report.json` - the de-identification check record for
   the delivery: finding counts by category and the pass decision.

Once coding runs on a session, a coded table follows in your column format
(`<CODE>_<coder_id>`), with a short rationale per code.

## How speakers are identified

Speakers are `SPEAKER_00`, `SPEAKER_01`, and so on, plus a `role` column
(`participant`, `confederate` or `experimenter`), so confederate rows can be
coded `NA` as in your earlier work. Demographic speaker labels (such as
`White24`) cannot be included. If an analysis depends on demographic
information, we need to discuss it explicitly.

## What never leaves USAFA

Video, audio, unreviewed transcripts, frame-by-frame descriptions, participant
names or descriptions, recording dates, and any filename derived from the
original recording. Everything that touches those runs on one FAIR Lab
computer; only the files above are shared.

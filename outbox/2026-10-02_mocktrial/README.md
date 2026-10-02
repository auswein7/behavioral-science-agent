# MockTrial sample (session s001) - FAIR Lab pipeline output

Source: public-domain YouTube mock-jury deliberation (MockTrial_trimmed, about 5
minutes, 8 people visible, 7 speakers detected). Run 2026-10-02, run id
`run5_mocktrial`, entirely on one FAIR Lab computer.

## Files

| file | what it is |
|---|---|
| `s001.screenplay.md` | Readable screenplay: speakers as SPEAKER_00-06, timestamped dialogue, tone directions, gaze and gesture lines |
| `s001.utterances.csv` | Analysis table, one row per utterance (79), OSU coded-transcript columns plus role, tone, low_confidence, nonverbal_notes |
| `s001.coded.csv` | The same rows with AO/PO/PE/NE/I/Apology codes and a one-line rationale per code (`<CODE>_fairlib_local`) |
| `s001.provenance.json` | Pipeline record: model digests, prompt versions, settings, stage timings. No session content |
| `s001.coder_provenance.json` | Coder record: model, codebook version, settings, egress decision |
| `*.scrub_report.json` | De-identification gate result per file: all three "clean" (0 findings) |

## Read before using

- The codes are a SCAFFOLD CHECK, not results. They come from a local model
  (Qwen2.5-14B) given one-line PLACEHOLDER definitions, because OSU's codebook
  has not arrived yet. Counts: AO 6, PO 6, PE 0, NE 2, I 3, Apology 0 over 79
  utterances. On OSU's S1T1 transcript the same setup under-codes PO, PE, NE and
  I badly. They show the pipeline runs video to codes, nothing more.
- Faithfulness check: 78 of 79 utterances exact; one (SPEAKER_01 at 00:01:59.555)
  is present with its timestamp shifted +0.5 s in the screenplay. The tables use
  the original timestamps.
- Pronouns are kept, per the 2026-09-16 ruling. Names, places and organizations
  heard in dialogue are replaced with [NAME]/[PLACE]/[ORG]; some replacements
  over-scrub (for example "[ORG] change").
- Human review notes (what the automated check cannot catch): dialogue mentions
  a witness "in the gray suit" and "the insurance capital of the world" (a
  location cue). Both are acceptable for public-domain footage; they illustrate
  why every cadet transcript is also read by a person before release.

# Leak taxonomy (draft v0.1, 2026-08-30)

The channels through which a screenplay or utterance table could let a reader
re-identify a participant, with the pipeline's current coverage per channel.
Status: DRAFT for team review (Shane owns the IRB framing, Dave the analysis
side); once agreed, this document drives the gate's category list
(`src/scrub/patterns.py`), the name scrub's entity map
(`src/scrub/name_scrub.py`), and the red-team protocol's checklist.

Evidence codes: run 1 = the MockTrial_trimmed baseline run; OSU sample =
`example_coded.xlsx` (their own de-identified transcript retained a spoken
instructor name - the bar this taxonomy sets is deliberately higher).

| # | Channel | Example | Prevention today | Detection today | Gap |
|---|---|---|---|---|---|
| 1 | Spoken name | run 1: "Trump" (misheard) in dialogue; OSU sample: instructor name | deterministic NER scrub, [NAME] token, before Ornith/table | gate `likely_name` | NER misses (sm model mistags); no recall measurement yet |
| 2 | On-screen text name | whiteboard names visible in run-1 frames | prompt rule (held in run 1: THEMIS logo, whiteboard never captioned) | gate `likely_name` | prompt compliance is probabilistic; no OCR-based check |
| 3 | Recognized person | Attenborough sample named the presenter | prompt rule | gate `likely_name` | VLM can name a public figure from appearance alone; re-test on Attenborough after prompt fixes |
| 4 | Gender | run 1 captions: "her", "man", "woman" | prompt rule (leaks: 94 captions in run 1) | gate `gendered` | caption model does not comply reliably; needs prompt fix or redaction stage |
| 5 | Appearance (clothing, hair, build) | run 1: "light blue shirt" reached the final screenplay | prompt rule (leaks: 94x "shirt") | gate `appearance` | same as 4; the worst channel by volume in run 1 |
| 6 | Age | "elderly", "young" | prompt rule | none | add regex category to the gate |
| 7 | Ethnicity | OSU's own labels encode it ("White24", "Latina23") | prompt rule; our labels are SPEAKER_NN by design | none | add regex category (NORP entity + word list) |
| 8 | Uniform / insignia / rank | cadet recordings: rank on uniform, name tapes | prompt rule ("uniform" in appearance regex) | gate `appearance` (partial) | cadet-specific: rank words, unit names; extend before P4 |
| 9 | Location / organization cues | run 1 dialogue: "Connecticut", "insurance capital of the world" (= Hartford) | NER scrub [PLACE]/[ORG] tokens | gate `likely_name` | multi-word cues without proper nouns ("this town is the insurance capital...") pass NER |
| 10 | Voice-print proxy (verbatim idiolect) | distinctive catchphrases, verbatim garble | none - dialogue ships verbatim by design | none | open question for the team: is verbatim text an acceptable channel? OSU coding needs it |
| 11 | Rare-event re-identification | a described event so unusual it identifies the group/session | none | none | red-team protocol is the only realistic detector; human review step |
| 12 | Recording-context metadata | dates in document names (OSU sample), raw video filenames | opaque session_id, artifact-derived filenames (DATA_MODEL 5.1) | schema: Tier C models forbid undeclared fields | keep manifest `notes` PII-free by review |

## Decisions this draft asks the team to make

1. Channels 4/5 (gender/appearance in captions): fix by better prompting, by a
   redaction pass over captions, or both? Current volume makes human clearance
   per finding impractical (run 1: 162 findings on the table).
2. Channel 10 (verbatim dialogue): accepted as-is for the coding task, or is
   paraphrase-on-flag needed for cadet data?
3. Channels 6/7: agree word lists so the gate categories can be added without
   guessing.
4. Channel 8: the cadet-uniform extension list (rank, unit, insignia terms) -
   needs USAFA input before P4.

## How coverage is verified

Each channel needs: (a) a detector (gate category, NER class, or human step),
(b) at least one regression fixture exercising it, and (c) for channels with a
history of leaking (1, 3, 4, 5), a measured recall number against a
hand-labeled leak set (TODO 2.3's LLM-judge pass is the planned second
detector). The red-team protocol (TODO 2.6) is the end-to-end evidence on top.

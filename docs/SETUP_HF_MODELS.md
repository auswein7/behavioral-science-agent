# Hugging Face gated-model onboarding

The diarization stage downloads pyannote models from Hugging Face on first
run, and those models are gated: an account must accept their conditions
before a token can fetch them. Skipping any step below produces a failed run
that looks like an authentication bug (this cost us a run during setup), so do
this once per machine/account before the first pipeline run.

## 1. Account and token

1. Create or sign in to a Hugging Face account (https://huggingface.co).
2. Create an access token: Settings -> Access Tokens -> New token. A
   fine-grained token with read access to gated repos (or a classic `read`
   token) is enough; the pipeline never writes to the Hub.
3. Put it in the repo's `.env` as `HF_TOKEN=hf_...` (see `.env.example`).
   The token is a secret: it lives only in `.env` (gitignored) and is never
   written into any run artifact - the preflight check verifies it is set
   before starting work.

## 2. Accept the gated-model conditions

While signed in to the SAME account the token belongs to, open each model
page and click through the access/conditions form:

- `pyannote/speaker-diarization-community-1` - the diarization pipeline this
  project uses (see `docs/PIPELINE_INTERNALS.md` section 2.4, in the
  workspace docs).
- If the page names additional dependencies it relies on (earlier pyannote
  pipelines split segmentation and embedding into separately gated repos such
  as `pyannote/segmentation-3.0`), accept those too - the pipeline page lists
  its own requirements.

Acceptance is instant for these models; there is no review wait.

## 3. Verify before burning GPU time

`main.py`'s preflight only proves the token exists, not that it can reach the
gated repos. A cheap end-to-end check that stops before real work:

```
.venv/bin/python -m src.cli.transcribe <any short clip>
```

A gating problem fails inside the first minute with a Hub 401/403 that names
the repo; fix by re-checking step 2 with the token's account. Downloads are
cached under `~/.cache/huggingface/` after the first success, so later runs
work offline.

## Notes for the distribution doc (OSU RAs)

- The token belongs to whoever operates the machine, not to FAIR Lab; each
  installation needs its own account + acceptance clicks.
- No other stage needs Hub access: Whisper weights come ungated via whisperx,
  emotion2vec+ downloads ungated through funasr, and the two Ollama models
  come from Ollama, not the Hub.

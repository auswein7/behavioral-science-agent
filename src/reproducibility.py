"""Whether a backend reproduces its output: measured, not declared.

Design principle 4 says a run must be reproducible from its trace alone.
Recording the seed does not make that true: a backend can accept a seed and
still vary (gemini-3.5-flash did, at temperature 0, 2026-09-08), and a
capability declaration only says the seed reaches the request. So each
run's record carries a status taken from MEASUREMENTS below, with the
evidence that set it. A backend nobody has measured is "unmeasured" - never
assumed reproducible because its adapter declares a seed.

A local model is not reproducible by default either. On 2026-09-10 the same
5 coder calls to qwen2.5:14b - same code, same planner prompt digest, seed
1234, temperature 0 - gave 1273, 1256, 1254 and 1252 completion tokens at
different times and moved one or two of five code cells. Output is stable
within one loaded model and request sequence, but the server's state (the
context window Ollama picks when num_ctx is unset, KV-cache reuse, how the
model was loaded) changes it. num_ctx is therefore pinned by the config
loader and recorded, and a parity figure needs repeated runs on the local
side as well as the remote one.

An entry changes only with new evidence, added here with a date and the
numbers, the same way a finding enters TODO.md.
"""

from types import MappingProxyType
from typing import Literal

from src.schemas import _Deliverable

Status = Literal["measured_reproducible", "measured_not_reproducible", "unmeasured"]


class Measurement(_Deliverable):
    status: Status
    evidence: str


class ReproducibilityRecord(_Deliverable):
    """The reproducibility claim a run's provenance makes, and its basis."""

    provider: str
    model: str
    seed: int | None
    temperature: float
    status: Status
    evidence: str


MEASUREMENTS = MappingProxyType(
    {
        ("ollama", "qwen2.5:14b"): Measurement(
            status="measured_not_reproducible",
            evidence=(
                "2026-09-10: 5 coder calls, identical code, planner prompt digest, "
                "seed 1234 and temperature 0, gave 1273, 1256, 1254 and 1252 "
                "completion tokens at different times and moved 1-2 of 5 code cells. "
                "num_ctx changes codes (4096 vs 8192 differ; unset loads 32768 on "
                "Ollama 0.33.1), and even at a fixed num_ctx the server's state moved "
                "output. Stable within one load and request sequence (2026-09-08: "
                "0/36 unstable cells over 3 back-to-back passes; num_ctx 4096 "
                "identical across a forced reload), and OSU session S1T1 (87 coded "
                "rows) was identical across two loads at num_ctx 32768 and 8192, so "
                "the variation is intermittent, not constant. Repeat runs for any "
                "parity figure."
            ),
        ),
        ("ollama", "llama3.1:8b"): Measurement(
            status="unmeasured",
            evidence=(
                "2026-09-08/10: identical output over back-to-back passes only; never "
                "measured across model loads, where qwen2.5:14b varied"
            ),
        ),
        ("gemini", "gemini-3.5-flash"): Measurement(
            status="measured_not_reproducible",
            evidence=(
                "2026-09-08 (fairlib-agent, live API): paired calls with seed 1234 "
                "gave different text at temperature 1.0 and at temperature 0.0; one "
                "counterexample settles it. A parity check on this model needs "
                "repeated sampling with a dispersion estimate. 2026-09-11 (fork "
                "coder, Attenborough, 6 rows x 3 repeats, temperature 0, no seed): "
                "all 36 code cells identical across repeats while the reply text "
                "varied - 1 and 2 rows needed a planner retry on the second and "
                "third pass for an off-shape reply, none on the first."
            ),
        ),
    }
)

UNMEASURED = (
    "no measurement recorded for this provider and model; a declared seed only "
    "means the seed reaches the request"
)


def reproducibility_of(
    provider: str, model: str, *, seed: int | None, temperature: float
) -> ReproducibilityRecord:
    """The record for one run's backend and sampling. A positive measurement
    holds only at temperature 0, where it was taken; a negative one holds at
    any temperature."""
    measured = MEASUREMENTS.get((provider, model))
    if measured is None:
        status: Status = "unmeasured"
        evidence = UNMEASURED
    elif temperature != 0 and measured.status == "measured_reproducible":
        status, evidence = "unmeasured", (
            f"measured reproducible only at temperature 0; this run uses {temperature}"
        )
    else:
        status, evidence = measured.status, measured.evidence
    return ReproducibilityRecord(
        provider=provider,
        model=model,
        seed=seed,
        temperature=temperature,
        status=status,
        evidence=evidence,
    )

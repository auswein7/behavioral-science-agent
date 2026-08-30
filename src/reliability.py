"""Retry primitive for model-backend calls.

Reliability lives in one place (workspace design principle 7): a hand-rolled
retry loop inside a stage signals a missing primitive, so stages wrap their
Ollama calls in call_with_retry instead. Transient transport failures retry
with exponential backoff; a definitive rejection (an HTTP 4xx such as
model-not-found) fails immediately; exhaustion raises a typed AdapterError.
"""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

import httpx
import ollama

from src.errors import AdapterError

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE = (
    ConnectionError,
    TimeoutError,
    httpx.TimeoutException,
    httpx.TransportError,
    ollama.ResponseError,
)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ollama.ResponseError):
        status = getattr(exc, "status_code", None)
        # A 4xx is a definitive rejection (bad model name, bad request);
        # retrying it just repeats the answer. 5xx and unknown are worth retries.
        return status is None or status >= 500
    return isinstance(exc, RETRYABLE)


def call_with_retry(
    fn: Callable[[], T],
    description: str,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> T:
    """Run fn(), retrying transient failures up to `attempts` times with
    exponential backoff. Raises AdapterError once attempts are exhausted or
    immediately on a non-retryable rejection."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except RETRYABLE as exc:
            if not _is_retryable(exc):
                raise AdapterError(f"{description} was rejected by the backend: {exc}") from exc
            last = exc
            if attempt < attempts:
                delay = backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                    description, attempt, attempts, exc, delay,
                )
                time.sleep(delay)
    raise AdapterError(f"{description} failed after {attempts} attempts: {last}") from last

"""The held-out split, declared before any prompt tuning (TODO section 5).

A row's split is a pure function of its session and uid, so the same row is
always in the same half no matter who runs the harness, in what order, or
on which subset. Tuning reads only "tune" rows; "test" rows are scored once,
at the end.

The split is utterance-level because OSU has sent a single session. Rows in
one session are correlated, so once several sessions exist the rule should
move to whole sessions; that change gets a new salt, which re-deals every
row and makes the change visible in every report's split_rule.
"""

import hashlib
from typing import Literal

SPLIT_SALT = "bsa-split-v1"
TEST_PERMILLE = 200

SPLIT_RULE = (
    f"utterance-level: blake2b-64 of '{SPLIT_SALT}|<session>|<uid>' mod 1000 "
    f"< {TEST_PERMILLE} is test, else tune (declared 2026-09-10, before tuning)"
)

Split = Literal["tune", "test"]


def split_of(session: str, uid: str) -> Split:
    """Which half of the declared split a row belongs to."""
    key = f"{SPLIT_SALT}|{session}|{uid}".encode()
    bucket = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % 1000
    return "test" if bucket < TEST_PERMILLE else "tune"

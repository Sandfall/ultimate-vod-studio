"""Cross-frame voting helpers for resolving tag/round detections."""

from collections import Counter
from collections.abc import Sequence

from vodparse.schema import Event


def _recover_original(cleaned: list[str], winner_key: str) -> str:
    """The first original-cased value whose .lower() equals the (lowercased) winning key."""
    return next(v for v in cleaned if v.lower() == winner_key)


def vote(values: Sequence[str | None]) -> str | None:
    """Most frequent string across frames (case-insensitive tally, original casing kept)."""
    cleaned = [v for v in values if v]
    if not cleaned:
        return None
    counts = Counter(v.lower() for v in cleaned)
    winner_key, _ = counts.most_common(1)[0]
    return _recover_original(cleaned, winner_key)


def vote_weighted(entries: list[tuple[str, float]]) -> str | None:
    """Most frequent value across frames, but ties broken by the highest single-frame confidence
    instead of by frame order. `entries` is (value, confidence) per detected frame. Vote count
    still dominates: a class with more votes always wins, so a lone high-confidence misread can't
    beat a well-supported reading: confidence only arbitrates an exact count tie. This matters for
    sparse matches: a 2-frame game split one fighter reading (detector, 0.96) vs one m_char reading
    (cascade, 0.86) resolves to the more confident detection rather than whichever frame happened to be
    sampled first. (The two confidences come from different models, so this is a heuristic tilt,
    not a calibrated comparison, but a defensible one when the only alternative is arbitrary
    order.) Falls back to insertion order when confidences also tie."""
    cleaned = [(v, c) for v, c in entries if v]
    if not cleaned:
        return None
    counts = Counter(v.lower() for v, _ in cleaned)
    best_conf: dict[str, float] = {}
    for v, c in cleaned:
        key = v.lower()
        best_conf[key] = max(best_conf.get(key, 0.0), c if c is not None else 0.0)
    top = max(counts.values())
    winner_key = max((k for k, n in counts.items() if n == top),
                     key=lambda k: best_conf[k])
    return _recover_original([v for v, _ in cleaned], winner_key)


def vote_optional(values: list[str | None]) -> str | None:
    """Like vote but absence competes: the winning value is kept only if it's present in at least
    as many frames as it's absent, so a field empty in most frames stays empty (a one-frame
    seed/handle misread as a sponsor is dropped) while a genuine but flaky reading still wins a
    tie. Returns the most common value, or None when absence dominates."""
    present = [v for v in values if v]
    winner  = vote(present)
    if winner is None:
        return None
    seen   = sum(1 for v in present if v.lower() == winner.lower())
    absent = len(values) - len(present)
    return winner if seen >= absent else None


def is_unanimous(values: list[str], winner: str | None) -> bool:
    """True if every detected value in this position matched the winner."""
    if winner is None:
        return False
    return all(v.lower() == winner.lower() for v in values if v)


def positional_columns[T](rows: list[list[T]]) -> list[list[T]]:
    """Transpose positioned rows into per-position columns: columns[i] holds every row's
    value at position i (rows shorter than i simply don't contribute). The whole tag-voting
    layer leans on this shape: "vote per position across frames/matches"."""
    columns: list[list[T]] = []
    for row in rows:
        for i, value in enumerate(row):
            while len(columns) <= i:
                columns.append([])
            columns[i].append(value)
    return columns


def collect_positioned_entries(frames: list[Event], detector) -> list[list[dict]]:
    """Run a per-frame detector returning positioned entries; gather one column of entry
    dicts per position across all frames. Keeping the whole dict lets callers vote on more
    than one field (e.g. the gamertag and its sponsor) from a single detector pass."""
    rows = []
    for frame in frames:
        entries = detector(frame.get("texts", []))
        if entries:
            rows.append(entries)
    return positional_columns(rows)


def collect_positioned(frames: list[Event], detector) -> list[list[str]]:
    """Like collect_positioned_entries, but each column holds just the entries' text."""
    entry_columns = collect_positioned_entries(frames, detector)
    return [[entry["text"] for entry in column] for column in entry_columns]

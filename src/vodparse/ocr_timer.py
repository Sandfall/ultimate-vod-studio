"""Timer/Clock OCR primitives: tuned for the outlined/bubbly tournament clock font.

preprocess_variants isolates digit fills from their thick outlines (high-threshold
white extraction + morphological cleanup) and parse_time turns an OCR string into a
ParsedTime.
"""

import cv2
import re
import numpy as np
from pathlib import Path

from vodparse.schema import ParsedTime


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_variants(crop: np.ndarray, debug_dir: Path | None = None) -> list[tuple[str, np.ndarray]]:
    """The (name, image) preprocessing variants of a clock crop, in the order a reader should
    try them: the white-fill extractions first, from the loosest threshold (t230) to the
    strictest (t250), a small then a larger opening kernel (k2, k3) at each, then the plain
    Otsu fallback. The order is load-bearing: timer.read_timer_at_bbox stops at the first
    variant that parses, and a caller that caps the sweep (its `max_variants`) gets the
    FIRST ones. Over ~500 real Ultimate clock crops the first variant (t230_k2) read ~91 %
    of them on its own; trimming the list was measured and rejected, so all seven stay for
    the full sweep. With `debug_dir`, every variant image is written there as
    crop_<name>.jpg."""
    variants: list[tuple[str, np.ndarray]] = []

    # Upscale to ~200px tall: more headroom for the smaller .83 digits
    scale = max(2, int(200 / crop.shape[0]) + 1)
    big   = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    # ── Outlined font approach: extract pure white fill only ─────────────────
    # The outline is in the ~150-220 range; pure white fill is 235+
    for thresh_val in [230, 240, 250]:
        _, white = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        # Opening (erode then dilate): removes thin outline remnants
        # without shrinking the main digit body much
        for kernel_size in [2, 3]:
            k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened  = cv2.morphologyEx(white, cv2.MORPH_OPEN, k, iterations=1)
            # Slight dilate to restore thickness after erosion
            dilated = cv2.dilate(opened, k, iterations=1)
            # Invert → black digits on white background (OCR prefers this)
            inverted = cv2.bitwise_not(dilated)
            name = f"t{thresh_val}_k{kernel_size}"
            variants.append((name, inverted))

            if debug_dir:
                cv2.imwrite(str(debug_dir / f"crop_{name}.jpg"), inverted)

    # ── Fallback: standard Otsu (inverted) ──────────────────────────────────
    _, otsu   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv  = cv2.bitwise_not(otsu)
    variants.append(("otsu_inv", otsu_inv))
    if debug_dir:
        cv2.imwrite(str(debug_dir / "crop_otsu_inv.jpg"), otsu_inv)

    return variants


# ── Time parsing ──────────────────────────────────────────────────────────────

# Matches: 6:15  06:15  6:15.83  6:15:83  06:15.83
# The fraction is exactly two digits: the tournament clock only ever shows centiseconds, and
# parse_time divides it by 100. Allowing 3 digits let a digit-fusing OCR misread ('6:15.835')
# capture 835 and inflate total_seconds by ~8s; with {2} the same misread degrades to '6:15'.
TIME_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})(?:[\.:](\d{2}))?\b")

# The dot-as-colon fallback (parse_time(dot_as_colon=True)): "7.28" read as 7:28. Some
# overlay fonts have a colon whose small square dots EasyOCR's recognizer returns as ".",
# so every variant of a perfectly clean crop fails the colon pattern above. Only tried when
# TIME_PATTERN finds nothing, and only for callers that opt in: on Ultimate's clock "." is
# the centiseconds separator, so the built-in detector path never uses it.
DOT_COLON_PATTERN = re.compile(r"\b(\d{1,2})\.(\d{2})(?:[\.:](\d{2}))?\b")

# Unseparated centiseconds (parse_time(unseparated_centiseconds=True)): Melee and Rivals 2
# draw the centiseconds as smaller digits after a space ("07:46 84"). MISREAD_MAP strips the
# space, so the recognizer's pieces join to "07:4684", which TIME_PATTERN rejects (its seconds
# must end at a word boundary). These take exactly two trailing digits as the centiseconds.
# Tried only after the patterns above miss, and only for games that draw the clock this way:
# on Ultimate's clock a trailing digit pair is a digit-fusing misread, not centiseconds.
TRAILING_CENTIS_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})(\d{2})\b")
TRAILING_CENTIS_DOT_PATTERN = re.compile(r"\b(\d{1,2})\.(\d{2})(\d{2})\b")

# The last resort for a spaced-centiseconds clock (unseparated_centiseconds=True): when the
# recognizer mangles the small centiseconds digits ("7:44 6.3", a dot inside the pair), none
# of the patterns above fit the space-stripped string ("7:446.3"). The space the game draws
# is real, though: the clock proper is the first whitespace-separated token, so that token is
# parsed on its own and the centiseconds are dropped. A clock without centiseconds is worth
# as much to the splitter as one with them (it works in seconds). Tried only after every
# pattern above misses, so a clean "7:44 63" still keeps its centiseconds.
CLOCK_TOKEN_PATTERNS = (TIME_PATTERN, DOT_COLON_PATTERN)

MISREAD_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8", "b": "8",
    "G": "6",
    " ": "",
})

def _first_token_clock(text: str, dot_as_colon: bool) -> re.Match[str] | None:
    """The clock in the FIRST whitespace-separated token of a spaced-centiseconds reading,
    the centiseconds after the space discarded (see CLOCK_TOKEN_PATTERNS). None when the
    reading has no space to split on (nothing new to try) or the token is no clock."""
    tokens = text.split()
    if len(tokens) < 2:
        return None
    head = tokens[0].translate(MISREAD_MAP)
    for pattern in CLOCK_TOKEN_PATTERNS[:2 if dot_as_colon else 1]:
        match = pattern.search(head)
        if match:
            return match
    return None


def parse_time(text: str, dot_as_colon: bool = False,
               unseparated_centiseconds: bool = False) -> ParsedTime | None:
    """Parse an OCR string into a clock reading (None when no clock pattern is found or the
    seconds are not < 60). `dot_as_colon` also accepts "M.SS" as minutes:seconds when the
    colon pattern fails (see DOT_COLON_PATTERN); `unseparated_centiseconds` also accepts two
    digits glued to the seconds as centiseconds ("7:4684" -> 7:46.84, see
    TRAILING_CENTIS_PATTERN), and, when even that misses, the clock in the first
    whitespace-separated token alone with the mangled centiseconds dropped ("7:44 6.3" ->
    7:44, see CLOCK_TOKEN_PATTERNS). The returned `raw` is the matched text as OCR wrote it,
    dot included, so the misread stays visible downstream. Games declare the two rules on
    their clock profile (vodparse.games.ClockHud); the strict pattern is always tried first."""
    cleaned = text.translate(MISREAD_MAP)
    patterns = [TIME_PATTERN]
    if dot_as_colon:
        patterns.append(DOT_COLON_PATTERN)
    if unseparated_centiseconds:
        patterns.append(TRAILING_CENTIS_PATTERN)
        if dot_as_colon:
            patterns.append(TRAILING_CENTIS_DOT_PATTERN)
    match = None
    for pattern in patterns:
        match = pattern.search(cleaned)
        if match:
            break
    if not match and unseparated_centiseconds:
        match = _first_token_clock(text, dot_as_colon)
    if not match:
        return None

    mins_s, secs_s, ms_s = match.groups()
    mins = int(mins_s)
    secs = int(secs_s)
    ms   = int(ms_s) if ms_s else None

    if secs >= 60:
        return None

    total = mins * 60 + secs + (ms / 100 if ms else 0)
    fmt   = f"{mins:01d}:{secs:02d}.{ms:02d}" if ms is not None else f"{mins:01d}:{secs:02d}"

    return {
        "raw":           match.group(),
        "minutes":       mins,
        "seconds":       secs,
        "milliseconds":  ms,
        "total_seconds": round(total, 2),
        "formatted":     fmt,
    }

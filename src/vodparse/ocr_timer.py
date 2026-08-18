"""Timer/Clock OCR primitives: tuned for the outlined/bubbly tournament clock font.

preprocess_variants isolates digit fills from their thick outlines (high-threshold
white extraction + morphological cleanup) and parse_time turns an OCR string into a
ParsedTime. The interactive single-image demo lives in tools/ocr_timer_demo.py.
"""

import cv2
import re
import numpy as np
from pathlib import Path

from vodparse.schema import ParsedTime


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_variants(crop: np.ndarray, debug_dir: Path | None = None) -> list[tuple[str, np.ndarray]]:
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

MISREAD_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8", "b": "8",
    "G": "6",
    " ": "",
})

def parse_time(text: str) -> ParsedTime | None:
    cleaned = text.translate(MISREAD_MAP)
    match   = TIME_PATTERN.search(cleaned)
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

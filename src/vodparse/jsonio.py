"""Atomic JSON file writing.

Every JSON artifact the app writes (pipeline.json, matches.json, reviewed.json, the
Review draft, debug dumps) goes through write_json_atomic so a crash mid-write can
never leave a truncated file: the reader either sees the old complete content or the
new complete content. This is the same tmp-then-replace pattern the pipeline's
embedded-preview writer uses for its JPEG frames.
"""

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file as UTF-8. The counterpart to write_json_atomic (which
    always writes UTF-8), so non-ASCII tags/tournament names round-trip regardless of the
    platform default encoding."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: str | Path, data: Any, indent: int = 2) -> None:
    """Serialize `data` and replace `path` in one atomic step: write to a sibling temp
    file, then `os.replace` it over the target. The temp file sits in the same directory
    because `os.replace` is only atomic within one filesystem. Serialization errors (or a
    crash) leave the existing file untouched and clean up the temp file.

    Assumes one writer per target path at a time (true throughout the app: each artifact
    has a single owning process), so a fixed ".tmp" sibling name is safe."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

"""Standard-stream setup shared by the pipeline/extract/export stage mains.

Source checkouts don't go through ``freeze_entry`` (the frozen build's single entry
point, which reconfigures streams before any ``vodparse`` import; see
``freeze_entry._force_utf8_streams``), so each stage's ``main()`` has to do the same
setup itself before it prints anything.
"""

import sys


def prepare_stage_streams() -> None:
    """Reconfigure stdout/stderr for a GUI-spawned stage subprocess.

    Two independent fixes, both aimed at the GUI's piped child process:

    UTF-8, replace on error: on Windows a piped stage child gets the locale codec
    (cp1252) for its standard streams, and stage log lines use characters like "→"
    that cp1252 can't encode; the resulting UnicodeEncodeError would abort the whole
    stage. The GUI decodes stage output as UTF-8 (``app._read_output``), so emitting
    UTF-8 here matches the other end. (The frozen app handles this in
    ``freeze_entry._force_utf8_streams``, which runs before any vodparse import;
    source checkouts never pass through that file, hence this helper.)

    Line buffering: piped stdout is block-buffered by default, which would hold every
    line until the process exits and snap the GUI's export progress bar 0 → full at
    the end; line-buffer so each clip's "  Saved:" line arrives as that clip finishes.

    Guarded like ``freeze_entry._force_utf8_streams``: frozen windowed builds can have
    None or stand-in streams without a ``reconfigure()`` method.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

"""Review tools shared by the main GUI (vodparse.gui.app) and the vod_grabber tool.

Both windows carry the same Tools menu over a ReviewScreen: Reread tags…, Change tournament
name… and the smart-tags summary popup. Each keeps its own wiring (which menu bar, which
subprocess runner, where its export root comes from), but what the tools *say* and *decide*
must stay identical in both, so those pieces live here once: the confirmation before a
reread discards edits, the note a rename needs when clips were already exported, the busy
note when an export is asked for mid-run, and the popup that reports a Smart sweep's pick.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox, QWidget

from vodparse.gui.paths import draft_path, export_csv_for, resolve_export_csv
from vodparse.smart_tags import read_report_summary

if TYPE_CHECKING:  # the Review screen pulls in QtMultimedia; only its type is needed here
    from vodparse.gui.review.screen import ReviewScreen

# ── Wording both windows show ───────────────────────────────────────────────────────────
# After Change tournament name… when clips already exist under the old name: the export-output
# buttons fall back to that older export (resolve_export_csv), and only a fresh export carries
# the new name into the clips, CSV and titles.
RENAME_EXPORT_NOTE = "Clips already exported keep their previous name. Export again to apply the new one."

# Export clips pressed while a subprocess (a reread, or the run itself) is still going: the
# request is dropped, and this says so, instead of leaving the user to assume an export began.
BUSY_EXPORT_NOTE = "Busy: wait for the current run to finish, then export again."


def confirm_reread_discard(parent: QWidget, review: ReviewScreen, stem: str, vod_name: str
                           ) -> bool:
    """Whether a Reread tags… regeneration may go ahead and discard the Review edits of the
    VOD `stem` (`vod_name` is its file name, for the prompt). Asks only when there is
    something to lose: live edits not yet flushed to the draft (has_unsaved_edits), or a
    saved draft on disk. The draft-file check alone would miss edits made inside the
    debounced-save window, so the live-edit check is needed too. Nothing to lose means no
    prompt and True."""
    if not review.has_unsaved_edits() and not draft_path(stem).exists():
        return True
    reply = QMessageBox.question(
        parent, "Reread tags?",
        f"Rereading the tags for {vod_name} regenerates the list of games/sets and discards "
        "your current Review edits. Continue?")
    return reply == QMessageBox.StandardButton.Yes


def rename_leaves_exported_clips_behind(review: ReviewScreen, export_root: Path) -> bool:
    """Whether, after a tournament rename, the review's export-output buttons point at an
    export under the previous name: the CSV the current name would export does not exist
    yet, so resolve_export_csv falls back to the last completed export. The caller then
    shows RENAME_EXPORT_NOTE. False when nothing is loaded or nothing was exported."""
    target = review.export_target()
    if target is None:
        return False
    return resolve_export_csv(*target, export_root) != export_csv_for(*target, export_root)


def show_smart_tags_summary(parent: QWidget, report_path: Path, log_menu_path: str) -> None:
    """Pop the smart-tags outcome: the winning flag combination and the runner-up, with their
    scores, read from the report the extract stage just wrote at `report_path`. The full
    ranking table always goes to the log transcript; `log_menu_path` names the menu entry
    that opens it in this window ("Debug → Show log…" or "View → Show log…"). A missing or
    malformed report shows nothing: the popup is informational and must never block Review."""
    summary = read_report_summary(report_path)
    if summary is None:
        return
    QMessageBox.information(parent, "Smart tag reading",
                            summary + f"\n\nFull ranking of the tag reading options: {log_menu_path}")

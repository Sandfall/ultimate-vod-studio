"""Console-script entry point for the Ultimate VOD Studio desktop GUI.

A thin, Qt-free wrapper around ``vodparse.gui.app``: importing the app pulls in PyQt6 at
module load, so the friendly missing-extra hint below can only live in a module that does
*not* import Qt itself. The ``ultimate-vod-studio`` console script points here; the frozen
build enters through ``freeze_entry.py`` instead (PyQt6 is always bundled there).

Requires the ``gui`` extra (PyQt6): ``pip install "vodparse[gui]"`` or, from a source
checkout, ``pip install -e ".[gui]"``.
"""


def main() -> None:
    """Launch the Ultimate VOD Studio desktop GUI."""
    try:
        from vodparse.gui import app
    except ModuleNotFoundError as e:
        # Most likely the GUI extra (PyQt6) isn't installed. Give an actionable hint
        # instead of a bare import traceback; the original error is chained for detail.
        raise SystemExit(
            "Ultimate VOD Studio couldn't start: PyQt6, the library it uses for the "
            'interface, isn\'t installed. Install it with `pip install "vodparse[gui]"` '
            '(or `pip install -e ".[gui]"` if you\'re running from a source checkout), '
            f"then launch again.\n({e})"
        ) from e
    app.main()


if __name__ == "__main__":
    main()

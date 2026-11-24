"""
Shared form columns for the Analyze page's option boxes
---------------------------------------------------------
The Source, Game and Analysis boxes each stack rows of the shape
"Label:  [field]  [side button]  (i)". Laid out box by box, every label is a different
width, so each field starts and ends somewhere else and the page looks ragged. A
``FormColumns`` gives all the boxes one set of columns instead: every label column is as
wide as the widest label in any box, every field starts at the same place, runs up to the
info buttons and so has the same width, and the info buttons line up down the right edge.

The four columns, left to right:

    label | field | side button | info button

A row always has a label and a field. The side button (the Game box's "Edit…" beside the
Overlay layout combo) and the info button are optional: a row without one gives its space
to the field, which then runs further right. A widget can also take a whole row to itself
(a note, or a button as wide as the box).

Each box keeps its own QGridLayout (a QGroupBox cannot share one with its neighbours);
``align`` is what makes their label columns match, so it must be called once every row of
every box has been added.

Usage:

    columns = FormColumns()
    game_grid = columns.new_grid(game_box)
    columns.add_row(game_grid, 0, QLabel("Game:"), game_combo, info=game_info_button)
    columns.add_full_width(game_grid, 1, game_note)
    ...                       # the other boxes' grids and rows
    columns.align()

Import this only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since this module imports PyQt6 widgets at load time.
"""

from PyQt6.QtWidgets import QComboBox, QGridLayout, QGroupBox, QSizePolicy, QWidget

LABEL_COLUMN = 0
FIELD_COLUMN = 1
SIDE_BUTTON_COLUMN = 2
INFO_COLUMN = 3
COLUMN_COUNT = 4


class FormColumns:
    """One set of label / field / side button / info columns shared by several boxes (see
    the module docstring). Build each box's grid with ``new_grid``, fill it with
    ``add_row`` and ``add_full_width``, then call ``align`` once."""

    def __init__(self) -> None:
        self._grids: list[QGridLayout] = []
        # The widgets sitting in a label column, in any grid: align() measures them.
        self._label_column_widgets: list[QWidget] = []

    def new_grid(self, box: QGroupBox) -> QGridLayout:
        """A grid laid on `box` whose field column takes up the box's spare width."""
        grid = QGridLayout(box)
        grid.setColumnStretch(FIELD_COLUMN, 1)
        self._grids.append(grid)
        return grid

    def add_row(self, grid: QGridLayout, row: int, label: QWidget, field: QWidget,
                side_button: QWidget | None = None, info: QWidget | None = None) -> None:
        """Add one "label | field | side button | info" row to `grid` (one made by
        ``new_grid``). The field runs right as far as the next widget of its row: up to
        `side_button`, else up to `info`, else to the box's right edge.

        `label` is whatever sits in the label column: usually a QLabel, but the Source
        box's "Choose video…" button sits there too. A QComboBox field is stretched to fill
        its space (left alone, a combo stays as narrow as its longest item, which is what
        made the combos' right edges ragged)."""
        if side_button is not None:
            field_column_span = SIDE_BUTTON_COLUMN - FIELD_COLUMN
        elif info is not None:
            field_column_span = INFO_COLUMN - FIELD_COLUMN
        else:
            field_column_span = COLUMN_COUNT - FIELD_COLUMN

        if isinstance(field, QComboBox):
            policy = field.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
            field.setSizePolicy(policy)

        grid.addWidget(label, row, LABEL_COLUMN)
        grid.addWidget(field, row, FIELD_COLUMN, 1, field_column_span)
        if side_button is not None:
            grid.addWidget(side_button, row, SIDE_BUTTON_COLUMN)
        if info is not None:
            grid.addWidget(info, row, INFO_COLUMN)
        self._label_column_widgets.append(label)

    def add_full_width(self, grid: QGridLayout, row: int, widget: QWidget) -> None:
        """Give `widget` the whole of `row`, from the box's left edge to its right."""
        grid.addWidget(widget, row, LABEL_COLUMN, 1, COLUMN_COUNT)

    def align(self) -> None:
        """Make every grid's label column as wide as the widest widget in any label column,
        so the fields of all the boxes start at the same place. Call once, after every row
        has been added."""
        label_column_width = max(
            (widget.sizeHint().width() for widget in self._label_column_widgets), default=0)
        for grid in self._grids:
            grid.setColumnMinimumWidth(LABEL_COLUMN, label_column_width)

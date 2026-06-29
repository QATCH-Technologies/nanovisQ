"""Import mode - select a folder/ZIP source and import runs into local data.

PORT FROM export_widget.Ui_Export:
    build()                  <- tab1 (source folder/ZIP, existing-files policy,
                                archive preview, progress, Import/Cancel)
    select_import_folder     -> _select_folder
    select_file_source       -> _select_zip
    select_folder/select_file-> _pick_folder / _pick_file
    generateImportDescription-> _append_source_to_tree (one source at a time)
    list_files               -> _list_files
    checkChanged5            -> (removed) files now reveal via tree expansion
    doImport                 -> _do_import (services.run_task)
    importTask               -> _import_task(abort, path)
    copytree                 -> _copytree

Behavioural parity with the original:
  * Source can be a folder or a ZIP archive; preview shows folders (and
    optionally files) before importing.
  * Existing-files policy Replace / Merge / Skip maps to the original
    btnGroup4 ids 1 / 2 / 3 (used by both the ZIP and folder import paths).
  * The threaded import polls the abort Event, emits progress on the "import"
    channel, and appends a history entry on completion (same HTML format the
    History view parses).
"""

import os
import time
import shutil
import zipfile
import datetime
from datetime import timezone as tz
from xml.dom import minidom

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.core.constants import Constants
from QATCH.common.logger import Logger as Log

from QATCH.ui.widgets.data_mode_base import DataModeWidget
from QATCH.ui.components import GlassPushButton, GlassOptionCard, GlassOptionCardGroup

TAG = "[DataImport]"

# Existing-files policy ids - preserved from the original btnGroup4.
POLICY_REPLACE = 1
POLICY_MERGE = 2
POLICY_SKIP = 3


class _DropZone(QtWidgets.QFrame):
    """Drag-and-drop target for adding import sources (folders or ZIPs).

    Accepts OS file/folder drops via mimedata URLs; the "Browse…" link (and a
    click anywhere in the zone) opens the same picker dialog as before.
    """

    filesDropped = QtCore.pyqtSignal(list)
    browseRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(86)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        lay.setAlignment(QtCore.Qt.AlignCenter)

        self._icon_lbl = QtWidgets.QLabel()
        self._icon_lbl.setAlignment(QtCore.Qt.AlignCenter)
        icon = self._zone_icon()
        if not icon.isNull():
            self._icon_lbl.setPixmap(icon.pixmap(22, 22))
        lay.addWidget(self._icon_lbl, 0, QtCore.Qt.AlignCenter)

        self._main_lbl = QtWidgets.QLabel("Drop a folder or a .zip here")
        self._main_lbl.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self._main_lbl, 0, QtCore.Qt.AlignCenter)

        self._browse_lbl = QtWidgets.QLabel('or <a href="#">Browse…</a>')
        self._browse_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self._browse_lbl.setTextFormat(QtCore.Qt.RichText)
        self._browse_lbl.setOpenExternalLinks(False)
        self._browse_lbl.linkActivated.connect(lambda _=None: self.browseRequested.emit())
        lay.addWidget(self._browse_lbl, 0, QtCore.Qt.AlignCenter)

        self._set_qss(active=False)

    @staticmethod
    def _zone_icon():
        try:
            from QATCH.common.architecture import Architecture

            path = os.path.join(Architecture.get_path(), "QATCH", "icons", "import.svg")
            if os.path.exists(path):
                return QtGui.QIcon(path)
        except Exception:
            pass
        return QtGui.QIcon()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.browseRequested.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_qss(active=True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._set_qss(active=False)
        super().dragLeaveEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        self._set_qss(active=False)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if p]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _set_qss(self, active):
        if active:
            frame_qss = """
                QFrame#dropZone {
                    background: rgba(10, 163, 230, 70);
                    border: 2px dashed rgba(0, 118, 174, 230);
                    border-radius: 10px;
                }
            """
        else:
            frame_qss = """
                QFrame#dropZone {
                    background: rgba(222, 238, 248, 150);
                    border: 2px dashed rgba(70, 130, 180, 200);
                    border-radius: 10px;
                }
                QFrame#dropZone:hover {
                    background: rgba(222, 238, 248, 210);
                    border: 2px dashed rgba(0, 118, 174, 230);
                }
            """
        self.setStyleSheet(frame_qss)
        self._main_lbl.setStyleSheet(
            "QLabel { color: rgba(40, 50, 65, 200); font-size: 12px; "
            "font-weight: 600; background: transparent; }"
        )
        self._browse_lbl.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 170); font-size: 11px; background: transparent; }"
            " QLabel a { color: rgba(0, 118, 174, 235); font-weight: 600; text-decoration: none; }"
        )


class _FlowLayout(QtWidgets.QLayout):
    """Left-to-right layout that wraps to additional rows as needed.

    Used for the source chips so they read as a simple flowing row (per the
    wireframe) instead of a bordered, vertically-scrolling "well".
    """

    def __init__(self, parent=None, margin=0, spacing=8):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)
        self.invalidate()  # tell Qt to schedule a real layout pass

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        item = self._items.pop(index) if 0 <= index < len(self._items) else None
        if item is not None:
            self.invalidate()
        return item

    def expandingDirections(self):
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        return size + QtCore.QSize(left + right, top + bottom)

    def _do_layout(self, rect, test_only):
        left, top, right, bottom = self.getContentsMargins()
        effective = rect.adjusted(left, top, -right, -bottom)
        x, y = effective.x(), effective.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + spacing
            if next_x - spacing > effective.right() and line_height > 0:
                x = effective.x()
                y += line_height + spacing
                next_x = x + hint.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + bottom


class _PopEffect(QtWidgets.QGraphicsEffect):
    """Paint-time scale + fade effect anchored on a widget's own center.

    Used for the chip "pop" in/out. Earlier attempts animated the chip's
    real geometry directly, which proved fragile: it depends on the flow
    layout already having computed a correct rect (not guaranteed before
    the panel's first real layout pass) and can briefly hand the chip's
    internal row layout surplus space with nowhere defined to put it. This
    effect instead scales the already-rendered widget at paint time only -
    the widget's actual (layout-managed) geometry is never touched, so a
    bouncy overshoot (QEasingCurve.OutBack) is perfectly safe here.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scale = 1.0
        self._fade = 1.0

    def get_scale(self):
        return self._scale

    def set_scale(self, value):
        self._scale = value
        self.update()

    def get_fade(self):
        return self._fade

    def set_fade(self, value):
        self._fade = value
        self.update()

    popScale = QtCore.pyqtProperty(float, get_scale, set_scale)
    popFade = QtCore.pyqtProperty(float, get_fade, set_fade)

    def boundingRectFor(self, rect):
        if self._scale <= 1.0:
            return rect
        dx = rect.width() * (self._scale - 1.0) / 2.0
        dy = rect.height() * (self._scale - 1.0) / 2.0
        return rect.adjusted(-dx, -dy, dx, dy)

    def draw(self, painter):
        pixmap, offset = self.sourcePixmap(QtCore.Qt.LogicalCoordinates)
        if pixmap.isNull():
            return
        painter.save()
        painter.setOpacity(self._fade)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        w, h = pixmap.width(), pixmap.height()
        painter.translate(offset.x() + w / 2.0, offset.y() + h / 2.0)
        painter.scale(self._scale, self._scale)
        painter.translate(-w / 2.0, -h / 2.0)
        painter.drawPixmap(0, 0, pixmap)
        painter.restore()


class ImportMode(DataModeWidget):
    MODE_KEY = "import"
    MODE_LABEL = "Import"

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------
    def build(self):
        self._sources = []  # list of selected source paths (folders/ZIPs)
        self._chip_widgets = {}  # path -> chip QFrame
        self._source_root_items = {}  # path -> top-level QTreeWidgetItem
        self._suspend_check_signal = False  # guards programmatic checkState writes

        # ---- Source card ----------------------------------------------
        src_card = self._card(
            "Import Source",
            "Drop a folder or a .zip - the type is detected automatically.",
        )
        src_lay = src_card.layout()
        # Tighten this section so it doesn't dominate the widget height.
        src_lay.setContentsMargins(14, 10, 14, 10)
        src_lay.setSpacing(8)

        self.drop_zone = _DropZone()
        self.drop_zone.filesDropped.connect(self._add_sources)
        self.drop_zone.browseRequested.connect(self._select_source)
        src_lay.addWidget(self.drop_zone)

        # Source row: caption + "Clear" action (adding sources is now done
        # via the drop zone above, by dropping or via its Browse… link).
        head_row = QtWidgets.QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        src_caption = QtWidgets.QLabel("Sources")
        src_caption.setStyleSheet(self._caption_qss())
        head_row.addWidget(src_caption)
        head_row.addStretch(1)

        self.btn_clear_sources = GlassPushButton(" Clear", variant="default")
        self.btn_clear_sources.setFixedHeight(26)
        self.btn_clear_sources.setIcon(self._icon("clear.svg"))
        self.btn_clear_sources.setToolTip("Remove all selected sources")
        self.btn_clear_sources.setEnabled(False)
        self.btn_clear_sources.clicked.connect(self._clear_sources)
        head_row.addWidget(self.btn_clear_sources)
        src_lay.addLayout(head_row)

        # Sources appear as removable "chips" flowing left-to-right directly
        # in the card (no bordered container) - matches the wireframe, which
        # shows staged sources as a simple row of boxes under the drop zone.
        self._sources_host = QtWidgets.QWidget()
        self._sources_host.setStyleSheet("background: transparent;")
        self._sources_flow = _FlowLayout(self._sources_host, margin=0, spacing=8)

        self.sources_placeholder = QtWidgets.QLabel(
            "No sources yet - drop a folder or .zip above, or click Browse…"
        )
        self.sources_placeholder.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 140); font-size: 12px; "
            "font-style: italic; background: transparent; }"
        )
        self._sources_flow.addWidget(self.sources_placeholder)
        src_lay.addWidget(self._sources_host)

        # Existing-files policy: caption + three labelled, selectable cards.
        # Laid out in a small responsive grid so they stack instead of
        # crushing when the panel is minimized.
        src_lay.addSpacing(4)
        policy_caption = QtWidgets.QLabel("When a run already exists")
        policy_caption.setStyleSheet(self._caption_qss())
        src_lay.addWidget(policy_caption)

        self.policy_group = GlassOptionCardGroup(self)
        self.card_merge = GlassOptionCard("Merge", "Add new, keep both")
        self.card_replace = GlassOptionCard("Replace", "Overwrite existing")
        self.card_skip = GlassOptionCard("Skip", "Leave existing")
        self.policy_group.addCard(self.card_merge, POLICY_MERGE)
        self.policy_group.addCard(self.card_replace, POLICY_REPLACE)
        self.policy_group.addCard(self.card_skip, POLICY_SKIP)
        self.card_merge.setChecked(True)  # original default

        self.policy_host = QtWidgets.QWidget()
        self.policy_host.setStyleSheet("background: transparent;")
        self.policy_grid = QtWidgets.QGridLayout(self.policy_host)
        self.policy_grid.setContentsMargins(0, 0, 0, 0)
        self.policy_grid.setHorizontalSpacing(8)
        self.policy_grid.setVerticalSpacing(8)
        self._policy_cards = [self.card_merge, self.card_replace, self.card_skip]
        self._policy_wide = None
        src_lay.addWidget(self.policy_host)

        # ---- Preview card ---------------------------------------------
        prev_card = QtWidgets.QFrame()
        prev_card.setObjectName("glassPanel")
        prev_card.setStyleSheet("""
            QFrame#glassPanel {
                background: rgba(255, 255, 255, 110);
                border: 1px solid rgba(218, 224, 232, 170);
                border-radius: 10px;
            }
        """)
        prev_lay = QtWidgets.QVBoxLayout(prev_card)
        prev_lay.setContentsMargins(14, 12, 14, 12)
        prev_lay.setSpacing(8)

        prev_header_row = QtWidgets.QHBoxLayout()
        prev_header_row.setContentsMargins(0, 0, 0, 0)
        prev_header = QtWidgets.QLabel("Data to Import")
        prev_header.setStyleSheet(
            "QLabel { color: #333; font-size: 12px; font-weight: bold; "
            "background: transparent; }"
        )
        prev_header_row.addWidget(prev_header)
        prev_header_row.addStretch(1)
        self.runs_count_label = QtWidgets.QLabel("0 runs selected")
        self.runs_count_label.setStyleSheet(self._caption_qss())
        prev_header_row.addWidget(self.runs_count_label)
        prev_lay.addLayout(prev_header_row)

        self.tree_hint = QtWidgets.QLabel("Expand a source to see its runs and files.")
        self.tree_hint.setStyleSheet(self._desc_qss())
        prev_lay.addWidget(self.tree_hint)

        self.archive_tree = QtWidgets.QTreeWidget()
        self.archive_tree.setObjectName("archiveTree")
        self.archive_tree.setColumnCount(2)
        self.archive_tree.setHeaderLabels(["Name", "Type"])
        self.archive_tree.setRootIsDecorated(True)
        self.archive_tree.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.archive_tree.setFocusPolicy(QtCore.Qt.NoFocus)
        self.archive_tree.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.archive_tree.setAnimated(True)  # smooth expand/collapse of branches
        self.archive_tree.setMinimumHeight(160)
        self.archive_tree.header().setStretchLastSection(False)
        self.archive_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.archive_tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.archive_tree.setStyleSheet("""
            QTreeWidget#archiveTree {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QTreeWidget#archiveTree::item {
                padding: 4px 2px;
                border: none;
                color: rgba(30, 42, 56, 220);
            }
            QHeaderView::section {
                background-color: transparent;
                padding: 6px 8px;
                border: none;
                border-bottom: 1px solid rgba(210, 218, 228, 170);
                font-weight: bold;
                color: #333;
            }
            QScrollBar:vertical {
                background: transparent; width: 8px; margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: rgba(120, 130, 145, 90);
                border-radius: 4px; min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background: rgba(120, 130, 145, 140); }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)
        self.archive_tree.itemChanged.connect(self._on_tree_item_changed)

        # Placeholder shown in place of the tree until at least one source
        # has been added - mirrors the sources placeholder above.
        self.tree_placeholder = QtWidgets.QWidget()
        self.tree_placeholder.setStyleSheet("background: transparent;")
        ph_lay = QtWidgets.QVBoxLayout(self.tree_placeholder)
        ph_lay.setContentsMargins(12, 20, 12, 20)
        ph_lay.setSpacing(6)
        ph_lay.setAlignment(QtCore.Qt.AlignCenter)

        self.tree_placeholder_icon = QtWidgets.QLabel()
        self.tree_placeholder_icon.setAlignment(QtCore.Qt.AlignCenter)
        icon = self._icon("info-circle.svg")
        if not icon.isNull():
            self.tree_placeholder_icon.setPixmap(icon.pixmap(28, 28))
            ph_lay.addWidget(self.tree_placeholder_icon, 0, QtCore.Qt.AlignCenter)

        ph_text = QtWidgets.QLabel(
            "Nothing to import yet - add a folder or .zip above to preview its runs here."
        )
        ph_text.setWordWrap(True)
        ph_text.setAlignment(QtCore.Qt.AlignCenter)
        ph_text.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 140); font-size: 12px; "
            "font-style: italic; background: transparent; }"
        )
        ph_lay.addWidget(ph_text)

        self.tree_stack = QtWidgets.QStackedWidget()
        self.tree_stack.setMinimumHeight(160)
        self.tree_stack.addWidget(self.tree_placeholder)  # index 0 - empty state
        self.tree_stack.addWidget(self.archive_tree)  # index 1 - populated
        prev_lay.addWidget(self.tree_stack)
        self._update_tree_placeholder()

        # --- Scrollable content host -------------------------------------
        # Wrap both cards in a scroll area so nothing clips or overlaps when
        # the Data Management overlay is minimized (matches the Export page).
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setObjectName("importScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setStyleSheet(self._scroll_qss())
        self.scroll.viewport().setStyleSheet("background: transparent;")

        self.scroll_host = QtWidgets.QWidget()
        self.scroll_host.setObjectName("importScrollHost")
        self.scroll_host.setStyleSheet("QWidget#importScrollHost { background: transparent; }")
        content = QtWidgets.QVBoxLayout(self.scroll_host)
        content.setContentsMargins(2, 2, 6, 2)  # right pad = room for scrollbar
        content.setSpacing(12)
        content.addWidget(src_card)
        content.addWidget(prev_card, 1)

        self.scroll.setWidget(self.scroll_host)
        self.root.addWidget(self.scroll, 1)
        self._relayout_policy(force=True)

        # ---- Pinned footer: progress bar + status + actions -------------
        footer = QtWidgets.QFrame()
        footer.setObjectName("importFooter")
        footer.setStyleSheet("QFrame#importFooter { background: transparent; border: none; }")
        flay = QtWidgets.QVBoxLayout(footer)
        flay.setContentsMargins(0, 4, 0, 0)
        flay.setSpacing(8)

        # Slim determinate progress bar - matches the Export page's footer bar.
        self.import_progress = QtWidgets.QProgressBar()
        self.import_progress.setObjectName("importProgress")
        self.import_progress.setRange(0, 100)
        self.import_progress.setValue(0)
        self.import_progress.setTextVisible(False)
        self.import_progress.setFixedHeight(3)
        self.import_progress.setVisible(False)
        self.import_progress.setStyleSheet("""
            QProgressBar#importProgress {
                background: rgba(255, 255, 255, 35);
                border: none;
                border-radius: 1px;
            }
            QProgressBar#importProgress::chunk {
                background: rgba(0, 118, 174, 120);
                border-radius: 1px;
            }
        """)
        flay.addWidget(self.import_progress)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(False)
        self.status_label.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 165); font-size: 11px; "
            "background: transparent; border: none; padding: 0px; }"
        )
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )

        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        action_row.addWidget(self.status_label, 1)
        self.btn_cancel = GlassPushButton(" Cancel", variant="default")
        self.btn_cancel.setFixedHeight(34)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.services.request_abort)
        self.btn_import = GlassPushButton(" Import 0 runs", variant="primary")
        self.btn_import.setFixedHeight(34)
        self.btn_import.setEnabled(False)
        self.btn_import.clicked.connect(self._do_import)
        action_row.addWidget(self.btn_cancel)
        action_row.addWidget(self.btn_import)
        flay.addLayout(action_row)

        self.root.addWidget(footer, 0)

    @staticmethod
    def _scroll_qss():
        return """
            QScrollArea#importScroll { background: transparent; border: none; }
            QScrollBar:vertical {
                background: transparent; width: 8px; margin: 2px 0 2px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(120, 134, 150, 110);
                border-radius: 4px; min-height: 28px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(90, 104, 120, 150);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """

    # ------------------------------------------------------------------
    #  Responsive policy-card relayout
    # ------------------------------------------------------------------
    POLICY_BREAKPOINT = 480

    def _relayout_policy(self, force=False):
        """Lay the three policy cards out 3-across when there's room, or
        stacked full-width when the panel is minimized/narrow."""
        avail = self.scroll.viewport().width() if hasattr(self, "scroll") else self.width()
        wide = avail >= self.POLICY_BREAKPOINT
        if not force and wide == self._policy_wide:
            return
        self._policy_wide = wide

        while self.policy_grid.count():
            item = self.policy_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(self.policy_host)

        if wide:
            for c in range(3):
                self.policy_grid.setColumnStretch(c, 1)
            for idx, card in enumerate(self._policy_cards):
                self.policy_grid.addWidget(card, 0, idx)
        else:
            self.policy_grid.setColumnStretch(0, 1)
            for idx, card in enumerate(self._policy_cards):
                self.policy_grid.addWidget(card, idx, 0)
        for card in self._policy_cards:
            card.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "policy_grid"):
            self._relayout_policy()

    def on_enter(self):
        self._relayout_policy(force=True)

    # ------------------------------------------------------------------
    #  Shared hooks
    # ------------------------------------------------------------------
    def on_freeze(self, frozen: bool):
        # During a running import the worker drives enable-state via _set_running;
        # this guards against cross-mode freezes touching our controls.
        for w in (self.drop_zone, self.btn_clear_sources, self.btn_import):
            w.setDisabled(frozen)

    def on_progress(self, label, pct, color):
        # The slim bar conveys progress (matches the Export page's footer bar).
        try:
            self.import_progress.setValue(max(0, min(100, int(pct))))
        except Exception:
            pass
        if not label:
            return
        # Tint the implicit status line to match the message severity.
        self._set_status_tint(color)
        self.status_label.setText(label)

    # ------------------------------------------------------------------
    #  Source selection - one "Add Source" picker; backend detects type
    # ------------------------------------------------------------------
    def _start_dir(self):
        if self._sources:
            return os.path.split(self._sources[-1].rstrip("/\\"))[0]
        return os.path.expanduser(os.path.join("~", "Downloads"))

    def _select_source(self):
        """Open a picker that accepts either a folder or one/more ZIP files.

        Qt has no single native dialog that selects both files and directories,
        so we use a non-native dialog whose accept logic allows either. The
        chosen path(s) are classified by the backend in _add_sources.
        """
        dlg = QtWidgets.QFileDialog(self, "Add Import Source")
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dlg.setDirectory(self._start_dir())
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.setNameFilters(
            [
                "Import sources (*.zip)",
                "ZIP archives (*.zip)",
                "All files (*)",
            ]
        )
        # Let the user dive into and pick a directory too: a tree-view list with
        # a "choose current folder" affordance via the Open button on a dir.
        dlg.setLabelText(QtWidgets.QFileDialog.Accept, "Add")
        try:
            # Show directories in the list so they can be entered or selected.
            dlg.setFilter(dlg.filter() | QtCore.QDir.AllDirs)
        except Exception:
            pass

        if dlg.exec_() != QtWidgets.QFileDialog.Accepted:
            Log.w(TAG, "User cancelled source selection.")
            return
        paths = [p for p in dlg.selectedFiles() if p]
        if paths:
            self._add_sources(paths)

    def _classify(self, path):
        """Return 'zip', 'folder', or None for a chosen path."""
        if os.path.isdir(path):
            return "folder"
        if os.path.isfile(path) and path.lower().endswith(".zip"):
            return "zip"
        if os.path.isfile(path) and zipfile.is_zipfile(path):
            return "zip"  # accept zips with non-.zip extensions
        return None

    def _add_sources(self, paths):
        """Append new sources (folder or ZIP), de-dup. Only the newly added
        sources are touched - existing chips and tree branches (and their
        expand/checkbox state) are left completely alone."""
        new_paths = []
        for p in paths:
            p = p.rstrip("/\\") if os.path.isdir(p) else p
            if not p or p in self._sources:
                continue
            kind = self._classify(p)
            if kind is None:
                Log.w(TAG, f"Ignoring unsupported source (not a folder or ZIP): {p}")
                self.status_label.setText(
                    f"Skipped “{os.path.basename(p)}” - only folders and ZIP archives are supported."
                )
                continue
            self._sources.append(p)
            Log.i(TAG, f"Added import source ({kind}): {p}")
            new_paths.append(p)
        if not new_paths:
            return
        self._append_chips(new_paths)
        for p in new_paths:
            self._append_source_to_tree(p)

    def _clear_sources(self):
        if not self._sources:
            return
        self._sources = []
        self._source_root_items = {}
        self.archive_tree.clear()
        self._recount_runs()
        self._clear_chips()
        self.status_label.setText("")

    def _remove_source(self, path):
        if path not in self._sources:
            return
        Log.i(TAG, f"Removed import source: {path}")
        self._sources.remove(path)

        # Remove just this source's branch - sibling sources keep their
        # expand/checkbox state untouched (no full tree rebuild).
        root_item = self._source_root_items.pop(path, None)
        if root_item is not None:
            idx = self.archive_tree.indexOfTopLevelItem(root_item)
            if idx != -1:
                self.archive_tree.takeTopLevelItem(idx)
        self._recount_runs()
        if not self._sources:
            self.status_label.setText("")

        chip = self._chip_widgets.pop(path, None)
        if chip is None:
            return
        self._sources_flow.removeWidget(chip)
        self._sources_flow.activate()  # reflow the remaining chips immediately
        self._animate_chip_pop(chip, entering=False)
        if not self._sources:
            self.sources_placeholder.setVisible(True)
            self._sources_flow.addWidget(self.sources_placeholder)
        self.btn_clear_sources.setEnabled(bool(self._sources))

    # ------------------------------------------------------------------
    #  Source chips
    # ------------------------------------------------------------------
    def _append_chips(self, paths):
        """Add chip widgets for newly-added sources and pop them in."""
        if self.sources_placeholder.isVisible():
            self._sources_flow.removeWidget(self.sources_placeholder)
            self.sources_placeholder.setVisible(False)
        for path in paths:
            chip = self._make_chip(path)
            self._chip_widgets[path] = chip
            self._sources_flow.addWidget(chip)
            self._animate_chip_pop(chip, entering=True)
        self.btn_clear_sources.setEnabled(True)

    def _clear_chips(self):
        """Bulk-clear every chip instantly (deliberate "Clear" action - no
        per-chip pop, unlike single removal)."""
        while self._sources_flow.count():
            item = self._sources_flow.takeAt(0)
            w = item.widget()
            if w is not None and w is not self.sources_placeholder:
                w.deleteLater()
        self._chip_widgets = {}
        self.sources_placeholder.setVisible(True)
        self._sources_flow.addWidget(self.sources_placeholder)
        self.btn_clear_sources.setEnabled(False)

    def _animate_chip_pop(self, chip, entering):
        """Bouncy scale + fade "pop", anchored on the chip's own center.

        Uses ``_PopEffect`` (paint-time scale, not real geometry) so the
        chip's actual layout-managed size/position is never touched -
        animating that directly is what caused the earlier bugs.
        """
        eff = _PopEffect(chip)
        chip.setGraphicsEffect(eff)

        scale_anim = QtCore.QPropertyAnimation(eff, b"popScale", self)
        fade_anim = QtCore.QPropertyAnimation(eff, b"popFade", self)
        if entering:
            scale_anim.setDuration(320)
            scale_anim.setEasingCurve(QtCore.QEasingCurve.OutBack)
            scale_anim.setStartValue(0.35)
            scale_anim.setEndValue(1.0)
            fade_anim.setDuration(160)
            fade_anim.setStartValue(0.0)
            fade_anim.setEndValue(1.0)
        else:
            scale_anim.setDuration(180)
            scale_anim.setEasingCurve(QtCore.QEasingCurve.InCubic)
            scale_anim.setStartValue(1.0)
            scale_anim.setEndValue(0.3)
            fade_anim.setDuration(180)
            fade_anim.setStartValue(1.0)
            fade_anim.setEndValue(0.0)

        grp = QtCore.QParallelAnimationGroup(self)
        grp.addAnimation(scale_anim)
        grp.addAnimation(fade_anim)
        if entering:
            grp.finished.connect(lambda: chip.setGraphicsEffect(None))
        else:
            grp.finished.connect(chip.deleteLater)

        # Keep refs so in-flight animations aren't GC'd mid-flight; several
        # chips may be popping at once (e.g. a multi-file drop).
        self._chip_anims = [
            a
            for a in getattr(self, "_chip_anims", [])
            if a.state() == QtCore.QAbstractAnimation.Running
        ]
        self._chip_anims.append(grp)
        grp.start()

    def _make_chip(self, path):
        kind = self._classify(path) or "folder"
        is_zip = kind == "zip"
        name = os.path.basename(path.rstrip("/\\")) or path

        chip = QtWidgets.QFrame()
        chip.setObjectName("sourceChip")
        chip.setStyleSheet("""
            QFrame#sourceChip {
                background: rgba(255, 255, 255, 160);
                border: 1px solid rgba(215, 222, 230, 200);
                border-radius: 8px;
            }
        """)
        row = QtWidgets.QHBoxLayout(chip)
        row.setContentsMargins(10, 6, 6, 6)
        row.setSpacing(8)

        # Both labels are capped at their own sizeHint (never Expanding) so
        # they can't be stretched into filling leftover space - relevant
        # while the chip's own geometry is mid pop-in/out animation.
        label = QtWidgets.QLabel(name)
        label.setToolTip(path)
        label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        label.setStyleSheet(
            "QLabel { color: rgba(28, 40, 52, 230); font-size: 12px; "
            "font-weight: 600; background: transparent; }"
        )
        row.addWidget(label)

        # Colored kind badge stands in for an icon (zip vs. folder).
        kind_lbl = QtWidgets.QLabel("ZIP" if is_zip else "FOLDER")
        kind_lbl.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        badge_color = (
            "background: rgba(214, 90, 110, 60); color: rgba(160, 40, 60, 235);"
            if is_zip
            else "background: rgba(70, 160, 110, 60); color: rgba(30, 110, 70, 235);"
        )
        kind_lbl.setStyleSheet(
            f"QLabel {{ {badge_color} font-size: 9px; font-weight: 700; "
            "padding: 2px 6px; border-radius: 6px; }"
        )
        row.addWidget(kind_lbl)

        btn_x = QtWidgets.QToolButton()
        btn_x.setText("\u00d7")
        btn_x.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_x.setFixedSize(20, 20)
        btn_x.setToolTip("Remove this source")
        btn_x.setStyleSheet("""
            QToolButton {
                background: transparent; border: none; border-radius: 10px;
                color: rgba(110, 120, 130, 200); font-size: 15px; font-weight: bold;
            }
            QToolButton:hover {
                background: rgba(210, 55, 55, 40); color: rgba(200, 45, 45, 240);
            }
        """)
        btn_x.clicked.connect(lambda _=False, p=path: self._remove_source(p))
        row.addWidget(btn_x)
        return chip

    # ------------------------------------------------------------------
    #  Archive preview (one source's branch at a time - never a full rebuild)
    # ------------------------------------------------------------------
    def _append_source_to_tree(self, path):
        """Build and append ONE source's branch to the tree. Sibling sources'
        items, expand state, and checkboxes are never touched."""
        split = os.path.split(path)
        is_zip = self._classify(path) == "zip"
        src_label = (
            (split[1] or path)
            if not os.path.isdir(path)
            else (os.path.basename(path.rstrip("/\\")) or path)
        )
        src_root = self._make_item(src_label, "ZIP" if is_zip else "Folder")
        src_root.setData(0, QtCore.Qt.UserRole, path)

        self._suspend_check_signal = True
        try:
            self.archive_tree.addTopLevelItem(src_root)
            self._source_root_items[path] = src_root
            ok, warn_msg = self._preview_source(path, is_zip, src_root)
            if ok:
                self._mark_run_nodes(src_root)
            else:
                self._flag_source_root(src_root, warn_msg)
        except Exception as e:
            ok = False
            self._flag_source_root(src_root, str(e))
        finally:
            self._suspend_check_signal = False

        src_root.setExpanded(False)  # collapsed by default; drill in to inspect
        self._recount_runs()
        if ok:
            self._set_status_tint("b")
        else:
            self._set_status_tint("r")
            self.status_label.setText(f"“{src_label}” can't be imported - see warning below.")

    def _set_status_tint(self, color):
        fg = {
            "r": "rgba(176, 96, 12, 255)",
            "g": "rgba(46, 120, 70, 220)",
            "b": "rgba(60, 72, 88, 165)",
        }.get(color, "rgba(60, 72, 88, 165)")
        self.status_label.setStyleSheet(
            f"QLabel {{ color: {fg}; font-size: 11px; "
            "background: transparent; border: none; padding: 0px; }"
        )

    def _preview_source(self, path, is_zip, src_root):
        """Populate src_root from the source. Returns (importable, warn_msg)."""
        try:
            if is_zip:
                if not zipfile.is_zipfile(path):
                    return False, "Not a valid ZIP archive."
                with zipfile.ZipFile(path, "r") as f:
                    if f.testzip() is not None:
                        return False, "ZIP archive is corrupt."
                    names = sorted(n for n in f.namelist() if n.split("/")[0] != "__MACOSX")
                    if not names:
                        return False, "Archive is empty - nothing to import."
                    if not any(n.lower().endswith(".xml") for n in names):
                        return False, "No run info (.xml) found in this archive."
                    self._populate_from_paths(names, parent=src_root)
                    return True, None
            else:
                if not os.path.isdir(path):
                    return False, "Folder no longer exists."
                before = src_root.childCount()
                self._walk_folder(path, src_root)
                if src_root.childCount() == before:
                    return False, "Folder is empty - nothing to import."
                if not self._subtree_has_runinfo(src_root):
                    return False, "No run info (.xml) found in this folder."
                return True, None
        except Exception as e:
            return False, f"{e}"

    @staticmethod
    def _subtree_has_runinfo(item):
        """True if any descendant row is a Run Info (.xml) file."""
        stack = [item.child(i) for i in range(item.childCount())]
        while stack:
            node = stack.pop()
            if node.text(1) == "Run Info":
                return True
            stack.extend(node.child(i) for i in range(node.childCount()))
        return False

    def _flag_source_root(self, src_root, message):
        """Mark a source root as un-importable and attach a warning child."""
        src_root.setIcon(0, self._warning_icon())
        warn_fg = QtGui.QColor(176, 96, 12)
        src_root.setForeground(0, warn_fg)
        msg = message or "This source can't be imported."
        src_root.setToolTip(0, msg)
        src_root.addChild(self._make_warning_item(msg))

    # ------------------------------------------------------------------
    #  Run-level checkboxes - a "run" is a folder whose direct children
    #  include capture.zip (same definition the Export pipeline uses).
    # ------------------------------------------------------------------
    def _mark_run_nodes(self, parent_item):
        """Recursively tag any folder-kind descendant containing capture.zip
        as a checkable run, defaulting to checked. Files and non-run folders
        (sources, device folders) are left without a checkbox."""
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            if child.text(1) == "Folder":
                has_capture = any(
                    child.child(j).text(1) != "Folder"
                    and child.child(j).text(0).lower() == "capture.zip"
                    for j in range(child.childCount())
                )
                if has_capture:
                    self._mark_as_run(child)
                self._mark_run_nodes(child)

    @staticmethod
    def _mark_as_run(item):
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(0, QtCore.Qt.Checked)
        item.setData(0, QtCore.Qt.UserRole + 1, "run")

    def _on_tree_item_changed(self, item, column):
        if column != 0 or self._suspend_check_signal:
            return
        if item.data(0, QtCore.Qt.UserRole + 1) != "run":
            return
        self._recount_runs()

    def _recount_runs(self):
        """Update the "N runs selected" label and the Import button to match
        the checked run nodes across the whole tree."""
        selected = 0
        top_n = self.archive_tree.topLevelItemCount()
        stack = [self.archive_tree.topLevelItem(i) for i in range(top_n)]
        while stack:
            item = stack.pop()
            is_run = item.data(0, QtCore.Qt.UserRole + 1) == "run"
            if is_run and item.checkState(0) == QtCore.Qt.Checked:
                selected += 1
            stack.extend(item.child(i) for i in range(item.childCount()))
        noun = "run" if selected == 1 else "runs"
        self.runs_count_label.setText(f"{selected} {noun} selected")
        self.btn_import.setText(f" Import {selected} {noun}")
        self.btn_import.setEnabled(selected > 0)
        self._update_tree_placeholder()

    def _update_tree_placeholder(self):
        """Show the empty-state placeholder until the tree has content."""
        has_content = self.archive_tree.topLevelItemCount() > 0
        self.tree_stack.setCurrentIndex(1 if has_content else 0)
        self.tree_hint.setVisible(has_content)

    @staticmethod
    def _node_relpath(item, sep):
        """Join `item`'s ancestry (excluding the top-level source root) with
        `sep`, e.g. a run two levels under its source -> "DeviceX/RunY"."""
        parts = []
        node = item
        while node.parent() is not None:
            parts.append(node.text(0))
            node = node.parent()
        parts.reverse()
        return sep.join(parts)

    def _excluded_runs_for(self, src_root, sep):
        """Relative paths (joined with `sep`) of unchecked run nodes under
        `src_root`. Must be called on the GUI thread (reads checkState)."""
        excluded = set()

        def walk(item):
            if item.data(0, QtCore.Qt.UserRole + 1) == "run":
                if item.checkState(0) != QtCore.Qt.Checked:
                    excluded.add(self._node_relpath(item, sep))
                return  # runs aren't nested; no need to descend further
            for i in range(item.childCount()):
                walk(item.child(i))

        walk(src_root)
        return excluded

    @staticmethod
    def _relpath_excluded(rel, excluded, sep):
        if not excluded:
            return False
        return any(rel == ex or rel.startswith(ex + sep) for ex in excluded)

    # ---- Tree population --------------------------------------------------
    def _populate_from_paths(self, names, parent=None):
        """Build tree items (folders AND files) from a flat list of archive paths.

        ``parent`` is the per-source root node; paths nest beneath it. Files are
        always included - visibility is governed by tree expansion, not a toggle.
        """
        roots = {}  # path-tuple -> QTreeWidgetItem
        for name in names:
            is_dir = name.endswith("/")
            parts = [p for p in name.split("/") if p]
            if not parts:
                continue
            self._ensure_path(roots, parts, leaf_is_dir=is_dir, parent=parent)

    def _walk_folder(self, path, parent_item):
        try:
            entries = sorted(os.listdir(path))
        except OSError:
            return
        for entry in entries:
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                if entry == "__MACOSX":
                    continue
                node = self._make_item(entry, "Folder")
                parent_item.addChild(node)
                self._walk_folder(full, node)
            else:
                parent_item.addChild(self._make_item(entry, self._kind_for(entry)))

    def _ensure_path(self, roots, parts, leaf_is_dir, parent=None):
        """Create/lookup nested tree items for a path given as a list of parts.

        ``parent`` is the per-source root node the top-level parts attach to.
        """
        src_root = parent
        parent = src_root
        accum = ()
        for i, part in enumerate(parts):
            accum = accum + (part,)
            is_last = i == len(parts) - 1
            if accum in roots:
                parent = roots[accum]
                continue
            if is_last and not leaf_is_dir:
                kind = self._kind_for(part)
            else:
                kind = "Folder"
            item = self._make_item(part, kind)
            if parent is None:
                self.archive_tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            roots[accum] = item
            parent = item

    def _make_item(self, name, kind):
        item = QtWidgets.QTreeWidgetItem([name, kind])
        if kind == "Folder":
            f = item.font(0)
            f.setBold(True)
            item.setFont(0, f)
        return item

    def _warning_icon(self):
        """Warning icon for failed sources. Uses warning.svg if present in the
        icon set; otherwise paints a small amber triangle so the indicator is
        always visible regardless of which assets shipped."""
        ic = self._icon("warning.svg")
        if not ic.isNull():
            return ic
        pix = QtGui.QPixmap(16, 16)
        pix.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        amber = QtGui.QColor(214, 158, 46)
        p.setBrush(amber)
        p.setPen(QtCore.Qt.NoPen)
        tri = QtGui.QPolygonF([QtCore.QPointF(8, 1), QtCore.QPointF(15, 14), QtCore.QPointF(1, 14)])
        p.drawPolygon(tri)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1.6))
        p.drawLine(QtCore.QPointF(8, 5), QtCore.QPointF(8, 10))
        p.drawPoint(QtCore.QPointF(8, 12))
        p.end()
        return QtGui.QIcon(pix)

    def _make_warning_item(self, message, kind="Warning"):
        """A tree row flagging that a source/run can't be imported."""
        item = QtWidgets.QTreeWidgetItem([message, kind])
        item.setIcon(0, self._warning_icon())
        warn_fg = QtGui.QColor(176, 96, 12)
        item.setForeground(0, warn_fg)
        item.setForeground(1, warn_fg)
        f = item.font(0)
        f.setItalic(True)
        item.setFont(0, f)
        item.setToolTip(0, message)
        return item

    @staticmethod
    def _kind_for(filename):
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if not ext:
            return "File"
        if ext == "xml":
            return "Run Info"
        if ext == "csv":
            return "Data"
        return ext.upper()

    # ------------------------------------------------------------------
    #  Import
    # ------------------------------------------------------------------
    def _do_import(self):
        if not self._sources:
            return
        sources = list(self._sources)  # snapshot for the worker thread
        # Unchecked-run exclusions must be read on the GUI thread (checkState
        # access) before handing off to the worker.
        excluded_map = {}
        for path in sources:
            src_root = self._source_root_items.get(path)
            if src_root is None:
                continue
            sep = "/" if os.path.isfile(path) else Constants.slash
            excluded_map[path] = self._excluded_runs_for(src_root, sep)
        self.services.run_task(lambda abort: self._import_task(abort, sources, excluded_map))

    def _set_running(self, running):
        # Mirror the original importNow/importCancel toggling.
        QtCore.QMetaObject.invokeMethod(
            self.btn_import,
            "setEnabled",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, not running),
        )
        QtCore.QMetaObject.invokeMethod(
            self.btn_cancel,
            "setEnabled",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, running),
        )
        # Reset to 0 on start, then show/hide the slim bar (worker-thread safe).
        if running:
            QtCore.QMetaObject.invokeMethod(
                self.import_progress,
                "setValue",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, 0),
            )
        QtCore.QMetaObject.invokeMethod(
            self.import_progress,
            "setVisible",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, running),
        )

    def _policy_id(self):
        return self.policy_group.checkedId()

    def _import_task(self, abort, sources, excluded_map):
        self._set_running(True)
        total_copied = total_skipped = 0
        try:
            n = len(sources)
            self.services.emit_progress(
                self.MODE_KEY, f"Importing {n} source(s)... please wait...", 0, "g"
            )
            time.sleep(2)  # give the user a moment to bail
            if abort.is_set():
                self.services.emit_progress(
                    self.MODE_KEY,
                    "Import to local data: Operation cancelled. No import was performed.",
                    100,
                    "b",
                )
                Log.w(f"{TAG} Import thread killed prematurely!")
                return

            for idx, path in enumerate(sources, start=1):
                if abort.is_set():
                    self.services.emit_progress(
                        self.MODE_KEY,
                        f"Import cancelled after {idx - 1} of {n} source(s). Partial import performed.",
                        100,
                        "b",
                    )
                    Log.w(f"{TAG} Import cancelled between sources.")
                    break

                label = os.path.split(path)[1] or path
                self.services.emit_progress(
                    self.MODE_KEY, f"Importing source {idx} of {n}: {label}...", 0, "g"
                )

                excluded = excluded_map.get(path, set())
                if os.path.isfile(path):
                    copied, skipped = self._import_zip(abort, path, excluded)
                    if copied is None:  # invalid/corrupt zip already reported
                        continue  # skip this source, keep going with the rest
                else:
                    copied, skipped = self._import_folder(path, excluded)

                total_copied += copied
                total_skipped += skipped
                self._write_history(path, copied, skipped)

            msg = f"Imported {total_copied} run(s) from {n} source(s)! Import process complete."
            if total_skipped > 0:
                msg += f" {total_skipped} run(s) were skipped."
            self.services.emit_progress(self.MODE_KEY, msg, 100, "g")
        except Exception as e:
            Log.e(TAG, f"Import error: {e}")
            self.services.emit_progress(self.MODE_KEY, "Error importing local data!", 100, "r")
        finally:
            self._set_running(False)

    def _import_zip(self, abort, path, excluded=None):
        policy = self._policy_id()
        excluded = excluded or set()
        if not zipfile.is_zipfile(path):
            self.services.emit_progress(
                self.MODE_KEY,
                "Import Error: Invalid ZIP file selected! Please try again...",
                100,
                "r",
            )
            Log.e(f"{TAG} Selected file is not a valid ZIP.")
            return None, None

        copied = skipped = 0
        with zipfile.ZipFile(path, "r") as f:
            zip_filename = os.path.split(path)[1][:-4]
            self.services.emit_progress(
                self.MODE_KEY, "Verifying ZIP file integrity... please wait...", 0, "b"
            )
            if f.testzip() is not None:
                self.services.emit_progress(
                    self.MODE_KEY,
                    "Import Error: Corrupt ZIP file selected! Please try again...",
                    100,
                    "r",
                )
                Log.e(f"{TAG} ZIP archive contains a bad file.")
                return None, None

            local_data = os.path.join(Constants.log_prefer_path)
            Log.i(f"{TAG} Import from {path} to {local_data}")

            zippedFiles, zippedXMLs = [], []
            for info in f.infolist():
                if info.filename.split("/")[0] == "__MACOSX":
                    continue
                if info.filename[-1] == "/":
                    continue
                zippedFiles.append(info)
                if info.filename[-4:] == ".xml":
                    zippedXMLs.append(info)

            export_to = self._zip_export_map(f, zippedFiles, zippedXMLs, zip_filename, local_data)

            num = len(zippedFiles)
            zip_src = os.path.split(path)[1]
            for x, zf in enumerate(zippedFiles):
                pct = min(99, max(1, int(100 * (x + 1) / max(num, 1))))
                if abort.is_set():
                    self.services.emit_progress(
                        self.MODE_KEY,
                        f"Import {zip_src}: Operation cancelled. Partial import was performed.",
                        pct,
                        "b",
                    )
                    Log.w(f"{TAG} Import thread killed prematurely!")
                    return copied, skipped
                sp0 = os.path.split(zf.filename)
                if self._relpath_excluded(sp0[0], excluded, "/"):
                    continue  # source run was unchecked in the preview tree
                run = os.path.split(sp0[0])[1]
                try:
                    if run == "_unnamed":
                        run = sp0[1][0:-4]
                except Exception:
                    pass
                self.services.emit_progress(
                    self.MODE_KEY,
                    f"Importing {zip_src}... please wait... Importing '{run}'",
                    pct,
                    "g",
                )
                d = os.path.join(os.getcwd(), zf.filename)
                allow = self._allow_zip_copy(policy, zf, d)
                if allow:
                    if zf.filename.endswith(".xml"):
                        copied += 1
                    out = f.extract(zf, export_to.get(sp0[0]))
                    last_mod = datetime.datetime(*zf.date_time).astimezone()
                    epoch = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
                    ftime = (last_mod - epoch).total_seconds()
                    os.utime(out, (ftime, ftime))
                else:
                    if zf.filename.endswith(".xml"):
                        skipped += 1
        return copied, skipped

    @staticmethod
    def _allow_zip_copy(policy, zf, dest):
        if policy == POLICY_REPLACE:
            return True
        if not os.path.exists(dest):
            return True
        if policy == POLICY_MERGE:
            last_mod = datetime.datetime(*zf.date_time).astimezone()
            exist_mod = datetime.datetime.fromtimestamp(
                os.stat(dest).st_mtime, tz=datetime.timezone.utc
            )
            return last_mod - exist_mod > datetime.timedelta(seconds=2)
        return False  # POLICY_SKIP

    def _zip_export_map(self, f, zippedFiles, zippedXMLs, zip_filename, local_data):
        export_to = {}
        for xf in zippedXMLs:
            xfp = os.path.split(xf.filename)[0]
            xml_str = f.read(xf).decode()
            doc = minidom.parseString(xml_str)
            device = zip_filename
            for m in doc.getElementsByTagName("run_info"):
                device = m.getAttribute("device") if m.hasAttribute("device") else zip_filename
            if xf.filename.find(Constants.log_export_path) >= 0:
                exp = os.getcwd()
            elif xfp.count("/") == 0:
                exp = os.path.join(local_data, device)
            else:
                exp = local_data
            export_to[xfp] = exp
        for xf in zippedFiles:
            xfp = os.path.split(xf.filename)[0]
            if export_to.get(xfp) is None:
                if xf.filename.find(Constants.log_export_path) >= 0:
                    exp = os.getcwd()
                elif xfp.count("/") == 0:
                    exp = os.path.join(local_data, zip_filename)
                else:
                    exp = local_data
                export_to[xfp] = exp
        return export_to

    def _import_folder(self, path, excluded=None):
        policy = self._policy_id()
        excluded = excluded or set()

        def find_xml_files(directory):
            xmls, allf = [], []
            for root, dirs, files in os.walk(directory):
                dirs[:] = [d for d in dirs if d != "__MACOSX"]
                for file in files:
                    if file.endswith(".xml"):
                        xmls.append(os.path.join(root, file))
                    allf.append(os.path.join(root, file))
            return xmls, allf

        local_data = os.path.join(Constants.log_prefer_path)
        archive_filename = os.path.split(path)[1]
        xml_files, all_files = find_xml_files(path)
        export_to = {}

        for xf in xml_files:
            xfp = os.path.split(xf)[0]
            relative = self._relative(xfp, path)
            doc = minidom.parse(xf)
            found_dev = False
            device = archive_filename
            name = os.path.split(xfp)[1]
            for m in doc.getElementsByTagName("run_info"):
                found_dev = m.hasAttribute("device")
                device = m.getAttribute("device") if found_dev else archive_filename
                name = m.getAttribute("name") if m.hasAttribute("name") else os.path.split(xfp)[1]
            if relative.find(Constants.log_export_path) >= 0:
                exp = os.path.join(os.getcwd(), relative)
            elif relative.count(Constants.slash) == 0:
                if not found_dev and name == archive_filename:
                    if not os.path.exists(os.path.join(path, name)):
                        archive_filename = os.path.split(os.path.split(path)[0])[1]
                        device = archive_filename
                exp = os.path.join(local_data, device, name)
            else:
                exp = os.path.join(local_data, relative)
            export_to[xfp] = exp

        for xf in all_files:
            xfp = os.path.split(xf)[0]
            relative = self._relative(xfp, path)
            if export_to.get(xfp) is None:
                if relative.find(Constants.log_export_path) >= 0:
                    exp = os.path.join(os.getcwd(), relative)
                elif relative.count(Constants.slash) == 0:
                    device = archive_filename
                    name = os.path.split(xfp)[1]
                    if name == archive_filename and not os.path.exists(os.path.join(path, name)):
                        archive_filename = os.path.split(os.path.split(path)[0])[1]
                        device = archive_filename
                    exp = os.path.join(local_data, device, name)
                else:
                    exp = os.path.join(local_data, relative)
                export_to[xfp] = exp

        Log.i(f"{TAG} Import from {path} to {local_data}")
        copied = skipped = 0
        for src, dst in export_to.items():
            rel = self._relative(src, path)
            if self._relpath_excluded(rel, excluded, Constants.slash):
                continue  # source run was unchecked in the preview tree
            copied, skipped = self._copytree(src, dst, policy, copied, skipped)
        return copied, skipped

    @staticmethod
    def _relative(xfp, path):
        relative = xfp.replace(path, "")
        if len(relative) > 0:
            if relative[0] == Constants.slash:
                relative = relative[1:]
            if relative and relative[-1] == Constants.slash:
                relative = relative[:-1]
        return relative

    def _copytree(self, src, dst, policy, copied=0, skipped=0, date_filter=0):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                copied, skipped = self._copytree(s, d, policy, copied, skipped, date_filter)
                continue
            allow = False
            if policy == POLICY_REPLACE:
                allow = True
            elif not os.path.exists(d):
                allow = True
            elif policy == POLICY_MERGE:
                last_mod = datetime.datetime.fromtimestamp(os.stat(s).st_mtime, tz=tz.utc)
                exist_mod = datetime.datetime.fromtimestamp(os.stat(d).st_mtime, tz=tz.utc)
                if last_mod - exist_mod > datetime.timedelta(seconds=2):
                    allow = True
            if allow:
                if not os.path.exists(dst):
                    os.makedirs(dst)
                if item.endswith(".xml"):
                    copied += 1
                shutil.copy2(s, d)
            else:
                if item.endswith(".xml"):
                    skipped += 1
        return copied, skipped

    # ------------------------------------------------------------------
    #  History
    # ------------------------------------------------------------------
    def _write_history(self, path, copied, skipped):
        history_path = os.path.join(os.getcwd(), Constants.log_export_path, "export_history.log")
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    log_lines = f.read()
            else:
                log_lines = ""
            local_data = os.path.join(Constants.log_prefer_path)
            policy_btn = self.policy_group.checkedButton()
            policy_text = policy_btn.text() if policy_btn else "Merge"
            ts = str(datetime.datetime.now()).split(".")[0]
            with open(history_path, "w") as f:
                f.write(f"<b>Imported {copied} run(s) at {ts}</b><br/>\n")
                f.write(f'<small>from "{path}" <br/>\n')
                f.write(f'to "{local_data}"</small><br/>\n')
                f.write("<small>Settings: ")
                f.write("Import from {}, ".format("ZIP" if ".zip" in path else "Folder"))
                f.write(f"{policy_text} existing files</small><br/>\n")
                if skipped > 0:
                    f.write(
                        f"<small>Skipped {skipped} run(s) since overwrites were disabled.</small><br/>\n"
                    )
                f.write("<br/>\n")
                f.write(log_lines)
        except Exception as e:
            Log.e(TAG, f"Failed writing import history: {e}")

    # ------------------------------------------------------------------
    #  Styling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _desc_qss():
        return (
            "QLabel { color: rgba(60, 72, 88, 190); font-size: 12px; " "background: transparent; }"
        )

    @staticmethod
    def _caption_qss():
        return (
            "QLabel { color: rgba(60, 72, 88, 160); font-size: 10px; "
            "font-weight: 600; text-transform: uppercase; "
            "letter-spacing: 0.5px; background: transparent; }"
        )

    @staticmethod
    def _radio_qss():
        return (
            "QRadioButton, QCheckBox { color: rgba(40, 50, 65, 200); "
            "font-size: 12px; background: transparent; }"
        )

    def _icon(self, name):
        try:
            from QATCH.common.architecture import Architecture

            path = os.path.join(Architecture.get_path(), "QATCH", "icons", name)
            if os.path.exists(path):
                return QtGui.QIcon(path)
        except Exception:
            pass
        return QtGui.QIcon()

    def _card(self, title, subtitle=""):
        card = QtWidgets.QFrame()
        card.setObjectName("glassPanel")
        card.setStyleSheet("""
            QFrame#glassPanel {
                background: rgba(255, 255, 255, 110);
                border: 1px solid rgba(218, 224, 232, 170);
                border-radius: 10px;
            }
        """)
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        header = QtWidgets.QLabel(title)
        header.setStyleSheet(
            "QLabel { color: #333; font-size: 12px; font-weight: bold; "
            "background: transparent; }"
        )
        lay.addWidget(header)
        if subtitle:
            sub = QtWidgets.QLabel(subtitle)
            sub.setStyleSheet(self._desc_qss())
            sub.setWordWrap(True)
            lay.addWidget(sub)
        return card

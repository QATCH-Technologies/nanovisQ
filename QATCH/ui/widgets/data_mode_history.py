"""History mode — view and clear the import/export history log.

PORT FROM export_widget.Ui_Export:
    build()            <- tab3 construction (history view, clearAllHistory)
    tabChanged(idx==4) -> the history-loading branch becomes on_enter()
    do_clearAllHistory -> _clear_all()

Functionally identical to the original (same log path, same clear-via-trash +
reload, same empty-state handling), but instead of dumping the raw HTML log into
a flat QTextEdit, each entry is parsed into a styled glass "card": colour-coded
by action (Import = blue, Export = green), with the run count + timestamp as a
header and the from/to/settings lines as structured detail rows.
"""

import os
import re
import html

from PyQt5 import QtCore, QtWidgets

from QATCH.core.constants import Constants
from QATCH.common.logger import Logger as Log

from QATCH.ui.widgets.data_mode_base import DataModeWidget
from QATCH.ui.components.glass_push_button import GlassPushButton

try:
    import send2trash
except Exception:  # pragma: no cover - optional dependency
    send2trash = None

TAG = "[DataHistory]"


class HistoryMode(DataModeWidget):
    MODE_KEY = "history"
    MODE_LABEL = "History"

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------
    def build(self):
        self._entries = []  # cached parsed entries (file order = newest first)
        self._sort_desc = True  # True: newest first; False: oldest first

        # ---- Glass panel matching the import widget --------------------
        card = self._card("Import / Export History")
        card_lay = card.layout()

        # Header controls row: caption + sort segment.
        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        sort_caption = QtWidgets.QLabel("Sort")
        sort_caption.setStyleSheet(self._caption_qss())
        controls.addWidget(sort_caption)

        # Frosted segmented sort control (Newest / Oldest).
        self.sort_segment = QtWidgets.QFrame()
        self.sort_segment.setObjectName("sortSegment")
        self.sort_segment.setStyleSheet("""
            QFrame#sortSegment {
                background: rgba(255, 255, 255, 60);
                border: 1px solid rgba(255, 255, 255, 150);
                border-radius: 8px;
            }
        """)
        seg_lay = QtWidgets.QHBoxLayout(self.sort_segment)
        seg_lay.setContentsMargins(4, 4, 4, 4)
        seg_lay.setSpacing(4)
        self.sort_group = QtWidgets.QButtonGroup(self)
        self.btn_newest = self._segment_button("Newest first", "Most recent at the top")
        self.btn_oldest = self._segment_button("Oldest first", "Earliest at the top")
        self.sort_group.addButton(self.btn_newest, 0)
        self.sort_group.addButton(self.btn_oldest, 1)
        self.btn_newest.setChecked(True)
        seg_lay.addWidget(self.btn_newest)
        seg_lay.addWidget(self.btn_oldest)
        self.sort_group.buttonToggled.connect(self._on_sort_changed)
        controls.addWidget(self.sort_segment)
        controls.addStretch(1)
        card_lay.addLayout(controls)

        # Scrollable column of entry cards (frosted scrollbar, transparent body).
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
            QScrollBar::handle:vertical {
                background: rgba(120, 130, 145, 90);
                border-radius: 4px; min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background: rgba(120, 130, 145, 140); }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)

        self._list_host = QtWidgets.QWidget()
        self._list_host.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._list_layout = QtWidgets.QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(2, 2, 2, 2)
        self._list_layout.setSpacing(10)
        self._list_layout.addStretch(1)
        self.scroll.setWidget(self._list_host)

        # Empty-state label (shown when there's no history).
        self.empty_label = QtWidgets.QLabel("No import/export history to show.")
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 150); font-size: 13px; "
            "font-style: italic; background: transparent; padding: 24px; }"
        )

        card_lay.addWidget(self.empty_label)
        card_lay.addWidget(self.scroll, 1)

        # Footer row, right-aligned clear button.
        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch(1)
        self.btn_clear = GlassPushButton(" Clear All History", variant="danger")
        self.btn_clear.setFixedHeight(34)
        self.btn_clear.clicked.connect(self._clear_all)
        footer.addWidget(self.btn_clear)

        self.root.addWidget(card, 1)
        self.root.addLayout(footer)

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------
    def on_enter(self):
        """Load + render the history log when this mode is shown."""
        path = self._history_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    raw = f.read()
            except Exception as e:
                Log.e(TAG, f"Failed reading history log: {e}")
                raw = ""
        else:
            raw = ""

        self._entries = self._parse_entries(raw)  # newest-first (file order)
        self._render(self._sorted_entries())

        has_entries = bool(self._entries)
        self.btn_clear.setEnabled(has_entries)
        self.empty_label.setVisible(not has_entries)
        self.scroll.setVisible(has_entries)
        self.sort_segment.setEnabled(has_entries)

    # ------------------------------------------------------------------
    #  Sorting
    # ------------------------------------------------------------------
    def _sorted_entries(self):
        """Return the cached entries in the current sort order.

        The parser yields newest-first (file order). Descending == newest first
        (as parsed); ascending == oldest first (reversed)."""
        if self._sort_desc:
            return self._entries
        return list(reversed(self._entries))

    def _on_sort_changed(self, button, checked):
        if not checked:
            return
        self._sort_desc = button is self.btn_newest
        self._render(self._sorted_entries())

    # ------------------------------------------------------------------
    #  Clear
    # ------------------------------------------------------------------
    def _clear_all(self):
        """Trash the log, then reload (identical behavior to do_clearAllHistory)."""
        path = self._history_path()
        try:
            if os.path.exists(path):
                if send2trash is not None:
                    send2trash.send2trash(path)
                else:
                    # Fallback if send2trash is unavailable in this environment.
                    os.remove(path)
        except Exception as e:
            Log.e(TAG, f"Failed clearing history: {e}")
        self.on_enter()

    # ------------------------------------------------------------------
    #  Parsing — turn the raw HTML log into structured records
    # ------------------------------------------------------------------
    def _parse_entries(self, raw):
        """Split the log into entries and pull out structured fields.

        The writer emits, newest first, blocks like:
            <b>Imported N run(s) at YYYY-MM-DD HH:MM:SS</b><br/>
            <small>from "SRC" <br/>
            to "DST"</small><br/>
            <small>Settings: ...</small><br/>
            [<small>Skipped K run(s) ...</small><br/>]
            <br/>
        Entries are separated by a standalone `<br/>`.
        """
        if not raw or not raw.strip():
            return []

        # Each entry begins at a "<b>...run(s) at...</b>" header. Split on that
        # boundary so trailing skipped/settings lines stay with their entry.
        chunks = re.split(r"(?=<b>(?:Imported|Exported)\b)", raw)
        entries = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            rec = self._parse_one(chunk)
            if rec is not None:
                entries.append(rec)
        return entries

    def _parse_one(self, chunk):
        header = re.search(
            r"<b>\s*(Imported|Exported)\s+(\d+)\s+run\(s\)\s+at\s+(.*?)\s*</b>",
            chunk,
        )
        if not header:
            return None
        action = header.group(1)
        count = header.group(2)
        timestamp = header.group(3)

        src = self._search(r'from\s+"(.*?)"', chunk)
        dst = self._search(r'to\s+"(.*?)"', chunk)
        settings = self._search(r"Settings:\s*(.*?)\s*</small>", chunk)
        skipped = self._search(r"(Skipped\s+\d+\s+run\(s\).*?)\s*</small>", chunk)

        return {
            "action": action,  # "Imported" | "Exported"
            "count": count,
            "timestamp": timestamp,
            "src": src,
            "dst": dst,
            "settings": settings,
            "skipped": skipped,
        }

    @staticmethod
    def _search(pattern, text):
        m = re.search(pattern, text, flags=re.DOTALL)
        return m.group(1).strip() if m else None

    # ------------------------------------------------------------------
    #  Rendering — one glass card per entry
    # ------------------------------------------------------------------
    MAX_CARDS = 50  # cap rendered cards; long logs stay responsive

    def _render(self, entries):
        # Clear existing cards (keep the trailing stretch).
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        # Entries are newest-first; render at most MAX_CARDS of them.
        shown = entries[: self.MAX_CARDS]
        for rec in shown:
            self._list_layout.insertWidget(self._list_layout.count() - 1, self._make_card(rec))

        if len(entries) > len(shown):
            more = QtWidgets.QLabel(
                f"Showing the {len(shown)} most recent of {len(entries)} entries."
            )
            more.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            more.setStyleSheet(
                "QLabel { color: rgba(60, 72, 88, 150); font-size: 11px; "
                "font-style: italic; background: transparent; padding: 6px; }"
            )
            self._list_layout.insertWidget(self._list_layout.count() - 1, more)

    def _make_card(self, rec):
        is_import = rec["action"] == "Imported"
        if is_import:
            accent = "rgba(0, 118, 174, 230)"
            tint = "rgba(10, 163, 230, 22)"
            badge = "Import"
        else:
            accent = "rgba(46, 140, 90, 230)"
            tint = "rgba(46, 155, 110, 22)"
            badge = "Export"

        card = QtWidgets.QFrame()
        card.setObjectName("historyCard")
        card.setStyleSheet(f"""
            QFrame#historyCard {{
                background: {tint};
                border: 1px solid rgba(255, 255, 255, 150);
                border-left: 3px solid {accent};
                border-radius: 10px;
            }}
        """)
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(4)

        # Header row: badge + count + timestamp.
        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        badge_lbl = QtWidgets.QLabel(badge)
        badge_lbl.setStyleSheet(
            f"QLabel {{ color: #fff; background: {accent}; border-radius: 8px; "
            f"padding: 1px 8px; font-size: 10px; font-weight: bold; }}"
        )
        count_lbl = QtWidgets.QLabel(f"{rec['count']} run(s)")
        count_lbl.setStyleSheet(
            "QLabel { color: rgba(28, 40, 52, 220); font-size: 13px; "
            "font-weight: 600; background: transparent; }"
        )
        time_lbl = QtWidgets.QLabel(rec["timestamp"] or "")
        time_lbl.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 150); font-size: 11px; background: transparent; }"
        )

        head.addWidget(badge_lbl)
        head.addWidget(count_lbl)
        head.addStretch(1)
        head.addWidget(time_lbl)
        lay.addLayout(head)

        # Detail rows.
        if rec["src"]:
            lay.addWidget(self._detail_row("From", rec["src"]))
        if rec["dst"]:
            lay.addWidget(self._detail_row("To", rec["dst"]))
        if rec["settings"]:
            lay.addWidget(self._detail_row("Settings", rec["settings"]))
        if rec["skipped"]:
            warn = QtWidgets.QLabel(html.unescape(rec["skipped"]))
            warn.setWordWrap(True)
            warn.setStyleSheet(
                "QLabel { color: rgba(180, 95, 20, 220); font-size: 11px; "
                "background: transparent; }"
            )
            lay.addWidget(warn)

        return card

    @staticmethod
    def _detail_row(label, value):
        row = QtWidgets.QWidget()
        row.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        h.setAlignment(QtCore.Qt.AlignTop)

        key = QtWidgets.QLabel(label)
        key.setFixedWidth(58)
        key.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 160); font-size: 11px; "
            "font-weight: 600; background: transparent; }"
        )
        val = QtWidgets.QLabel(html.unescape(value))
        val.setWordWrap(True)
        val.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        val.setStyleSheet(
            "QLabel { color: rgba(30, 42, 56, 210); font-size: 11px; background: transparent; }"
        )
        h.addWidget(key)
        h.addWidget(val, 1)
        return row

    # ------------------------------------------------------------------
    @staticmethod
    def _history_path():
        return os.path.join(os.getcwd(), Constants.log_export_path, "export_history.log")

    # ------------------------------------------------------------------
    #  Shared glass styling helpers (mirrors data_mode_import)
    # ------------------------------------------------------------------
    def _card(self, title):
        card = QtWidgets.QFrame()
        card.setObjectName("glassPanel")
        card.setStyleSheet("""
            QFrame#glassPanel {
                background: rgba(255, 255, 255, 30);
                border: 1px solid rgba(200, 210, 220, 110);
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
        return card

    @staticmethod
    def _caption_qss():
        return (
            "QLabel { color: rgba(60, 72, 88, 160); font-size: 10px; "
            "font-weight: 600; text-transform: uppercase; "
            "letter-spacing: 0.5px; background: transparent; }"
        )

    def _segment_button(self, text, tooltip=""):
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.setCheckable(True)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(28)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        if tooltip:
            btn.setToolTip(tooltip)
        btn.setStyleSheet(self._segment_qss())
        return btn

    @staticmethod
    def _segment_qss():
        return """
            QToolButton {
                background: transparent; border: none; border-radius: 6px;
                color: rgba(40, 50, 65, 190); font-size: 12px; font-weight: 600;
                padding: 0px 12px;
            }
            QToolButton:hover   { background: rgba(255, 255, 255, 80); }
            QToolButton:checked {
                background: rgba(255, 255, 255, 235);
                color: rgba(0, 118, 174, 230);
            }
        """

"""History mode - view and clear the import/export history log.

PORT FROM export_widget.Ui_Export:
    build()            <- tab3 construction (history view, clearAllHistory)
    tabChanged(idx==4) -> the history-loading branch becomes on_enter()
    do_clearAllHistory -> _clear_all()

Functionally identical to the original (same log path, same clear-via-trash +
reload, same empty-state handling), but instead of dumping the raw HTML log
into a flat QTextEdit, each entry renders as a compact, expandable row:
direction chip + action/count + an inline "N skipped" pill, a right-aligned
path preview and timestamp, and a chevron. Clicking a row expands it in place
to show the full from/to/settings detail plus per-entry actions (Repeat
export, Open folder). Rows are grouped under date headers (Today/Yesterday/...)
and can be filtered to All/Export/Import and sorted Newest/Oldest first.
"""

import datetime
import html
import os
import re

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.components import AnimatedComboBox, GlassPushButton
from QATCH.ui.widgets.data_mode_base import DataModeWidget

try:
    import send2trash
except Exception:  # pragma: no cover - optional dependency
    send2trash = None

TAG = "[DataHistory]"


class _ClickableRow(QtWidgets.QFrame):
    """A transparent row container that emits ``clicked`` on a left click -
    used as the always-visible header of an expandable history row."""

    clicked = QtCore.pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class HistoryMode(DataModeWidget):
    MODE_KEY = "history"
    MODE_LABEL = "History"

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------
    def build(self):
        self._entries = []  # cached parsed entries (file order = newest first)
        self._sort_desc = True  # True: newest first; False: oldest first
        self._filter = "all"  # "all" | "Exported" | "Imported"

        # Pagination state for the infinite-scroll list (see _render /
        # _append_next_page / _on_scroll_changed).
        self._page_entries = []  # full filtered/sorted list backing the list
        self._visible_count = 0  # how many of _page_entries are rendered
        self._last_group = None  # date-group header most recently rendered
        self._end_label = None  # "- End of history -" marker, once shown
        self._loading_more = False  # reentrancy guard for the scroll handler

        heading = QtWidgets.QLabel("Import / Export History")
        heading.setStyleSheet(
            "QLabel { color: #333; font-size: 14px; font-weight: bold; background: transparent; }"
        )
        self.root.addWidget(heading)
        subtitle = QtWidgets.QLabel("Every transfer, most recent first.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(self._desc_qss())
        self.root.addWidget(subtitle)

        # Controls row: All/Export/Import filter chips (left) + sort dropdown (right).
        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 4, 0, 0)
        controls.setSpacing(8)

        self.filter_group = QtWidgets.QButtonGroup(self)
        self.btn_filter_all = self._segment_button("All")
        self.btn_filter_export = self._segment_button("Export")
        self.btn_filter_import = self._segment_button("Import")
        self.filter_group.addButton(self.btn_filter_all, 0)
        self.filter_group.addButton(self.btn_filter_export, 1)
        self.filter_group.addButton(self.btn_filter_import, 2)
        self.btn_filter_all.setChecked(True)
        self.filter_group.buttonToggled.connect(self._on_filter_changed)

        filter_box = QtWidgets.QFrame()
        filter_box.setObjectName("filterSegment")
        filter_box.setStyleSheet("""
            QFrame#filterSegment {
                background: rgba(255, 255, 255, 60);
                border: 1px solid rgba(255, 255, 255, 150);
                border-radius: 8px;
            }
        """)
        filter_lay = QtWidgets.QHBoxLayout(filter_box)
        filter_lay.setContentsMargins(4, 4, 4, 4)
        filter_lay.setSpacing(4)
        filter_lay.addWidget(self.btn_filter_all)
        filter_lay.addWidget(self.btn_filter_export)
        filter_lay.addWidget(self.btn_filter_import)
        controls.addWidget(filter_box)
        controls.addStretch(1)

        self.sort_combo = AnimatedComboBox(icon_path=self._icon_file_path("down-chevron.svg"))
        self.sort_combo.addItem("Newest first", True)
        self.sort_combo.addItem("Oldest first", False)
        self.sort_combo.setCurrentIndex(0)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        controls.addWidget(self.sort_combo)

        self.root.addLayout(controls)

        # Scrollable column of date groups + entry rows.
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
        self._list_layout.setContentsMargins(2, 4, 2, 2)
        self._list_layout.setSpacing(8)
        self._list_layout.addStretch(1)
        self.scroll.setWidget(self._list_host)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)

        # Empty-state label (shown when there's no history).
        self.empty_label = QtWidgets.QLabel("No import/export history to show.")
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 150); font-size: 13px; "
            "font-style: italic; background: transparent; padding: 24px; }"
        )

        self.root.addWidget(self.empty_label)
        self.root.addWidget(self.scroll, 1)

        # Footer row: total count (left), Clear All History (right).
        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        self.count_label = QtWidgets.QLabel()
        self.count_label.setStyleSheet(self._caption_qss())
        footer.addWidget(self.count_label)
        footer.addStretch(1)
        self.btn_clear = GlassPushButton(" Clear history", variant="danger")
        self.btn_clear.setFixedHeight(32)
        self.btn_clear.clicked.connect(self._clear_all)
        footer.addWidget(self.btn_clear)

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
        self._render(self._visible_entries())

        has_entries = bool(self._entries)
        self.btn_clear.setEnabled(has_entries)
        self.empty_label.setVisible(not has_entries)
        self.scroll.setVisible(has_entries)
        total = sum(int(e["count"]) for e in self._entries if str(e["count"]).isdigit())
        noun = "transfer" if len(self._entries) == 1 else "transfers"
        self.count_label.setText(f"{len(self._entries)} {noun} logged ({total} runs)")

    # ------------------------------------------------------------------
    #  Filtering / sorting
    # ------------------------------------------------------------------
    def _visible_entries(self):
        entries = self._entries if self._sort_desc else list(reversed(self._entries))
        if self._filter != "all":
            entries = [e for e in entries if e["action"] == self._filter]
        return entries

    def _on_filter_changed(self, button, checked):
        if not checked:
            return
        self._filter = {0: "all", 1: "Exported", 2: "Imported"}[self.filter_group.id(button)]
        self._render(self._visible_entries())

    def _on_sort_changed(self, index):
        self._sort_desc = bool(self.sort_combo.itemData(index))
        self._render(self._visible_entries())

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
    #  Parsing - turn the raw HTML log into structured records
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
        skip_match = re.search(r"Skipped\s+(\d+)\s+run", chunk)
        skipped_count = int(skip_match.group(1)) if skip_match else None

        return {
            "action": action,  # "Imported" | "Exported"
            "count": count,
            "timestamp": timestamp,
            "src": src,
            "dst": dst,
            "settings": settings,
            "skipped": skipped,
            "skipped_count": skipped_count,
        }

    @staticmethod
    def _search(pattern, text):
        m = re.search(pattern, text, flags=re.DOTALL)
        return m.group(1).strip() if m else None

    # ------------------------------------------------------------------
    #  Rendering - date-grouped, compact expandable rows
    # ------------------------------------------------------------------
    PAGE_SIZE = 50  # rows rendered per page; long logs stay responsive

    def _render(self, entries):
        """Reset the list to ``entries`` and render just the first page.

        Further pages are appended by ``_maybe_load_more`` as the user
        scrolls toward the bottom (see ``_on_scroll_changed``).
        """
        # Clear existing rows/headers (keep the trailing stretch).
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self._page_entries = entries
        self._visible_count = 0
        self._last_group = None
        self._end_label = None
        self._append_next_page()

        # Filter/sort changes should read from the top, not wherever the
        # scrollbar happened to be left.
        self.scroll.verticalScrollBar().setValue(0)

    def _append_next_page(self):
        """Render the next PAGE_SIZE entries onto the end of the list."""
        remaining = self._page_entries[self._visible_count :]
        batch = remaining[: self.PAGE_SIZE]
        if not batch:
            self._show_end_label()
            return

        for rec in batch:
            group = self._group_label(rec["timestamp"])
            if group != self._last_group:
                self._list_layout.insertWidget(
                    self._list_layout.count() - 1, self._group_header(group)
                )
                self._last_group = group
            self._list_layout.insertWidget(self._list_layout.count() - 1, self._make_row(rec))
        self._visible_count += len(batch)

        if self._visible_count >= len(self._page_entries):
            self._show_end_label()

    def _show_end_label(self):
        """Append an "end of history" cue - only once pagination actually
        happened (a single page's worth of entries needs no such marker)."""
        if self._end_label is not None or len(self._page_entries) <= self.PAGE_SIZE:
            return
        label = QtWidgets.QLabel("- End of history -")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 150); font-size: 11px; "
            "font-style: italic; background: transparent; padding: 6px; }"
        )
        self._list_layout.insertWidget(self._list_layout.count() - 1, label)
        self._end_label = label

    def _on_scroll_changed(self, _value):
        """Load the next page once the user scrolls near the bottom."""
        if self._loading_more:
            return
        bar = self.scroll.verticalScrollBar()
        near_bottom = bar.value() >= bar.maximum() - 48
        if not near_bottom or self._visible_count >= len(self._page_entries):
            return
        self._loading_more = True
        try:
            self._append_next_page()
        finally:
            self._loading_more = False

    @staticmethod
    def _group_label(ts_str):
        dt = HistoryMode._parse_ts(ts_str)
        if dt is None:
            return "EARLIER"
        d = dt.date()
        today = datetime.date.today()
        if d == today:
            return f"TODAY · {d.strftime('%B %d').upper()}"
        if d == today - datetime.timedelta(days=1):
            return f"YESTERDAY · {d.strftime('%B %d').upper()}"
        if d.year != today.year:
            return f"{d.strftime('%B %d').upper()}, {d.year}"
        return d.strftime("%B %d").upper()

    @staticmethod
    def _group_header(text):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 150); font-size: 10px; font-weight: 700; "
            "letter-spacing: 0.5px; background: transparent; padding: 8px 2px 2px 4px; }"
        )
        return lbl

    def _make_row(self, rec):
        is_import = rec["action"] == "Imported"
        accent = QtGui.QColor(0, 118, 174, 235) if is_import else QtGui.QColor(46, 140, 90, 230)
        tint = "rgba(10, 163, 230, 18)" if is_import else "rgba(46, 155, 110, 18)"
        icon_name = "import.svg" if is_import else "export.svg"
        label = "Import" if is_import else "Export"

        card = QtWidgets.QFrame()
        card.setObjectName("historyRow")
        card.setStyleSheet(f"""
            QFrame#historyRow {{
                background: {tint};
                border: 1px solid rgba(255, 255, 255, 150);
                border-radius: 10px;
            }}
        """)
        outer = QtWidgets.QVBoxLayout(card)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = _ClickableRow()
        header.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        header.setStyleSheet("QFrame { background: transparent; }")
        hlay = QtWidgets.QHBoxLayout(header)
        hlay.setContentsMargins(12, 9, 12, 9)
        hlay.setSpacing(10)

        hlay.addWidget(self._direction_chip(icon_name, accent))

        title_lbl = QtWidgets.QLabel(f"{label} · {rec['count']} runs")
        title_lbl.setStyleSheet(
            "QLabel { color: rgba(28, 40, 52, 230); font-size: 12.5px; font-weight: 700; "
            "background: transparent; }"
        )
        hlay.addWidget(title_lbl)

        if rec["skipped_count"]:
            hlay.addWidget(self._skip_pill(rec["skipped_count"]))

        hlay.addStretch(1)

        preview_path = rec["src"] if is_import else rec["dst"]
        arrow = "←" if is_import else "→"
        preview_lbl = QtWidgets.QLabel(f"{arrow} {self._short_path(preview_path)}")
        preview_lbl.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 165); font-size: 11px; background: transparent; }"
        )
        hlay.addWidget(preview_lbl)

        time_lbl = QtWidgets.QLabel(self._fmt_time(rec["timestamp"]))
        time_lbl.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 150); font-size: 11px; background: transparent; }"
        )
        hlay.addWidget(time_lbl)

        chevron_lbl = QtWidgets.QLabel("▾")
        chevron_lbl.setStyleSheet(
            "QLabel { color: rgba(120, 130, 145, 200); font-size: 11px; background: transparent; }"
        )
        hlay.addWidget(chevron_lbl)
        outer.addWidget(header)

        detail = QtWidgets.QWidget()
        detail.setVisible(False)
        dlay = QtWidgets.QVBoxLayout(detail)
        dlay.setContentsMargins(12, 0, 12, 10)
        dlay.setSpacing(6)
        if rec["src"]:
            dlay.addWidget(self._detail_row("From", rec["src"]))
        if rec["dst"]:
            dlay.addWidget(self._detail_row("To", rec["dst"]))
        if rec["settings"]:
            dlay.addWidget(self._detail_row("Settings", rec["settings"]))
        if rec["skipped"]:
            warn = QtWidgets.QLabel(html.unescape(rec["skipped"]))
            warn.setWordWrap(True)
            warn.setStyleSheet(
                "QLabel { color: rgba(180, 95, 20, 220); font-size: 11px; "
                "background: transparent; }"
            )
            dlay.addWidget(warn)

        actions_row = QtWidgets.QHBoxLayout()
        actions_row.setContentsMargins(0, 4, 0, 0)
        actions_row.setSpacing(8)
        if not is_import:
            btn_repeat = GlassPushButton(" Repeat export", variant="primary")
            btn_repeat.setFixedHeight(26)
            btn_repeat.setIcon(self._icon("refresh-cw.svg"))
            btn_repeat.clicked.connect(self._go_to_export)
            actions_row.addWidget(btn_repeat)
        open_target = rec["src"] if is_import else rec["dst"]
        btn_open = GlassPushButton(" Open folder", variant="default")
        btn_open.setFixedHeight(26)
        btn_open.clicked.connect(lambda _=False, p=open_target: self._open_folder(p))
        actions_row.addWidget(btn_open)
        actions_row.addStretch(1)
        dlay.addLayout(actions_row)
        outer.addWidget(detail)

        header.clicked.connect(lambda: self._toggle_row(detail, chevron_lbl))
        return card

    @staticmethod
    def _toggle_row(detail, chevron_lbl):
        visible = not detail.isVisible()
        detail.setVisible(visible)
        chevron_lbl.setText("▴" if visible else "▾")

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
    #  Per-entry actions
    # ------------------------------------------------------------------
    def _go_to_export(self):
        """Jump to the Export step so the user can set up a similar export.

        Settings aren't replayed automatically - the logged text ("All Runs,
        CSV Report, Append existing") is free-form and lossy to re-parse
        reliably, so this is a shortcut to the wizard, not an auto-repeat.
        """
        w = self.parentWidget()
        while w is not None and not hasattr(w, "segmented"):
            w = w.parentWidget()
        if w is not None:
            w.segmented.set_active("export")
        else:
            Log.d(f"{TAG} could not locate container to switch to Export mode")

    def _open_folder(self, path):
        if not path:
            return
        target = path if os.path.isdir(path) else os.path.dirname(path)
        if not target or not os.path.exists(target):
            Log.w(TAG, f"Cannot open folder, path does not exist: {target}")
            return
        try:
            if os.name == "nt":
                os.startfile(target)
            else:
                import subprocess

                subprocess.call(["xdg-open", target])
        except OSError as e:
            Log.e(TAG, f"Failed to open folder {target}: {e}")

    # ------------------------------------------------------------------
    #  Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_ts(ts_str):
        try:
            return datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _fmt_time(ts_str):
        dt = HistoryMode._parse_ts(ts_str)
        return dt.strftime("%H:%M") if dt else (ts_str or "")

    @staticmethod
    def _short_path(path, max_len=42):
        if not path:
            return ""
        if len(path) <= max_len:
            return path
        return "…" + path[-(max_len - 1) :]

    # ------------------------------------------------------------------
    @staticmethod
    def _history_path():
        return os.path.join(os.getcwd(), Constants.log_export_path, "export_history.log")

    # ------------------------------------------------------------------
    #  Shared glass styling helpers (mirrors data_mode_import / data_mode_advanced)
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
    def _skip_pill(count):
        pill = QtWidgets.QLabel(f"{count} skipped")
        pill.setStyleSheet(
            "QLabel { color: rgba(150, 95, 10, 240); background: rgba(235, 175, 60, 75); "
            "border-radius: 8px; font-size: 10px; font-weight: 700; padding: 1px 7px; }"
        )
        return pill

    def _direction_chip(self, icon_name, color, size=26):
        chip = QtWidgets.QLabel()
        chip.setFixedSize(size, size)
        chip.setAlignment(QtCore.Qt.AlignCenter)
        chip.setStyleSheet(
            f"QLabel {{ background: rgba({color.red()}, {color.green()}, {color.blue()}, 35); "
            f"border-radius: {size // 2}px; }}"
        )
        icon = self._tinted_icon(icon_name, color, icon_size=int(size * 0.55))
        if icon is not None:
            chip.setPixmap(icon)
        return chip

    def _icon(self, name):
        path = self._icon_file_path(name)
        return QtGui.QIcon(path) if path else QtGui.QIcon()

    def _tinted_icon(self, name, color, icon_size=18):
        path = self._icon_file_path(name)
        if not path:
            return None
        src = QtGui.QIcon(path).pixmap(icon_size, icon_size)
        dst = QtGui.QPixmap(src.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(dst)
        p.drawPixmap(0, 0, src)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()
        return dst

    @staticmethod
    def _icon_file_path(name):
        try:
            path = os.path.join(Architecture.get_path(), "QATCH", "icons", name)
            if os.path.exists(path):
                return path.replace("\\", "/")
        except Exception:
            pass
        return ""

    def _segment_button(self, text, tooltip=""):
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.setCheckable(True)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(26)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
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

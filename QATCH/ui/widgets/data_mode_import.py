"""Import mode — select a folder/ZIP source and import runs into local data.

PORT FROM export_widget.Ui_Export:
    build()                  <- tab1 (source folder/ZIP, existing-files policy,
                                archive preview, progress, Import/Cancel)
    select_import_folder     -> _select_folder
    select_file_source       -> _select_zip
    select_folder/select_file-> _pick_folder / _pick_file
    generateImportDescription-> _generate_description
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
from QATCH.ui.components.glass_push_button import GlassPushButton

TAG = "[DataImport]"

# Existing-files policy ids — preserved from the original btnGroup4.
POLICY_REPLACE = 1
POLICY_MERGE = 2
POLICY_SKIP = 3


class ImportMode(DataModeWidget):
    MODE_KEY = "import"
    MODE_LABEL = "Import"

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------
    def build(self):
        self._sources = []  # list of selected source paths (folders/ZIPs)
        self._chip_widgets = {}  # path -> chip QFrame

        # ---- Source card ----------------------------------------------
        src_card = self._card("Import Source")
        src_lay = src_card.layout()
        # Tighten this section so it doesn't dominate the widget height.
        src_lay.setContentsMargins(14, 10, 14, 10)
        src_lay.setSpacing(6)

        # Source row: caption + single "Add Source" action.
        head_row = QtWidgets.QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        src_caption = QtWidgets.QLabel("Sources")
        src_caption.setStyleSheet(self._caption_qss())
        head_row.addWidget(src_caption)
        head_row.addStretch(1)

        self.btn_add_source = GlassPushButton(" Add Source", variant="default")
        self.btn_add_source.setFixedHeight(28)
        self.btn_add_source.setIcon(self._icon("add.svg"))
        self.btn_add_source.setToolTip("Add a folder or ZIP archive to the import set")
        self.btn_add_source.clicked.connect(self._select_source)
        self.btn_clear_sources = GlassPushButton(" Clear", variant="default")
        self.btn_clear_sources.setFixedHeight(28)
        self.btn_clear_sources.setIcon(self._icon("clear.svg"))
        self.btn_clear_sources.setToolTip("Remove all selected sources")
        self.btn_clear_sources.setEnabled(False)
        self.btn_clear_sources.clicked.connect(self._clear_sources)
        head_row.addWidget(self.btn_add_source)
        head_row.addWidget(self.btn_clear_sources)
        src_lay.addLayout(head_row)

        # Sources appear as removable "chips" inside a frosted well. When empty,
        # a helpful placeholder invites the user to add something. The well has
        # a fixed visible height; once chips overflow it, it scrolls rather than
        # growing the card.
        self.sources_well = QtWidgets.QFrame()
        self.sources_well.setObjectName("sourcesWell")
        self.sources_well.setStyleSheet("""
            QFrame#sourcesWell {
                background: rgba(255, 255, 255, 40);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 8px;
            }
        """)
        self.sources_well.setFixedHeight(64)  # scroll past this; do not grow
        well_outer = QtWidgets.QVBoxLayout(self.sources_well)
        well_outer.setContentsMargins(4, 4, 4, 4)
        well_outer.setSpacing(0)

        self.sources_scroll = QtWidgets.QScrollArea()
        self.sources_scroll.setWidgetResizable(True)
        self.sources_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.sources_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.sources_scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
            QScrollBar::handle:vertical {
                background: rgba(120, 130, 145, 90);
                border-radius: 4px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: rgba(120, 130, 145, 140); }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)
        self._sources_host = QtWidgets.QWidget()
        self._sources_host.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._sources_flow = QtWidgets.QVBoxLayout(self._sources_host)
        self._sources_flow.setContentsMargins(4, 4, 4, 4)
        self._sources_flow.setSpacing(6)
        self._sources_flow.addStretch(1)  # keeps chips top-aligned

        self.sources_placeholder = QtWidgets.QLabel(
            "No sources yet — click “Add Source” to choose a folder or ZIP "
            "archive of runs to import."
        )
        self.sources_placeholder.setWordWrap(True)
        self.sources_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sources_placeholder.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 140); font-size: 12px; "
            "font-style: italic; background: transparent; }"
        )
        self._sources_flow.insertWidget(0, self.sources_placeholder)
        self.sources_scroll.setWidget(self._sources_host)
        well_outer.addWidget(self.sources_scroll)
        src_lay.addWidget(self.sources_well)

        # Existing-files policy: caption + frosted segmented control.
        src_lay.addSpacing(4)
        policy_caption = QtWidgets.QLabel("When a run already exists")
        policy_caption.setStyleSheet(self._caption_qss())
        src_lay.addWidget(policy_caption)

        self.policy_group = QtWidgets.QButtonGroup(self)
        self.policy_segment = QtWidgets.QFrame()
        self.policy_segment.setObjectName("policySegment")
        self.policy_segment.setStyleSheet("""
            QFrame#policySegment {
                background: rgba(255, 255, 255, 60);
                border: 1px solid rgba(255, 255, 255, 150);
                border-radius: 8px;
            }
        """)
        seg = self.policy_segment
        seg.setFixedHeight(36)

        # Sliding highlight pill sits BEHIND the buttons and animates to the
        # active one. Created as a child of the segment, positioned manually.
        self._policy_pill = QtWidgets.QFrame(seg)
        self._policy_pill.setObjectName("policyPill")
        self._policy_pill.setStyleSheet("""
            QFrame#policyPill {
                background: rgba(255, 255, 255, 235);
                border: none; border-radius: 6px;
            }
        """)
        self._policy_pill.lower()

        seg_lay = QtWidgets.QHBoxLayout(seg)
        seg_lay.setContentsMargins(4, 4, 4, 4)
        seg_lay.setSpacing(4)
        self.rb_merge = self._segment_button("Merge", "Keep newer versions")
        self.rb_replace = self._segment_button("Replace", "Overwrite existing")
        self.rb_skip = self._segment_button("Skip", "Leave existing untouched")
        self.policy_group.addButton(self.rb_replace, POLICY_REPLACE)
        self.policy_group.addButton(self.rb_merge, POLICY_MERGE)
        self.policy_group.addButton(self.rb_skip, POLICY_SKIP)
        self._policy_order = [self.rb_merge, self.rb_replace, self.rb_skip]
        for rb in self._policy_order:
            # The pill provides the "checked" background, so the buttons keep a
            # transparent checked state to avoid a double-fill.
            rb.setStyleSheet(self._segment_qss(checked_bg=False))
            seg_lay.addWidget(rb)
        self.rb_merge.setChecked(True)  # original default
        self.policy_group.buttonToggled.connect(self._on_policy_toggled)
        seg.installEventFilter(self)
        src_lay.addWidget(seg)

        # ---- Preview card ---------------------------------------------
        prev_card = self._card("Data to Import")
        prev_lay = prev_card.layout()

        hint = QtWidgets.QLabel("Expand a source to see its runs and files.")
        hint.setStyleSheet(self._desc_qss())
        prev_lay.addWidget(hint)

        self.archive_tree = QtWidgets.QTreeWidget()
        self.archive_tree.setObjectName("archiveTree")
        self.archive_tree.setColumnCount(2)
        self.archive_tree.setHeaderLabels(["Name", "Type"])
        self.archive_tree.setRootIsDecorated(True)
        self.archive_tree.setAlternatingRowColors(True)
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
                border-bottom: 1px solid rgba(255, 255, 255, 90);
                color: rgba(30, 42, 56, 220);
            }
            QTreeWidget#archiveTree::item:alternate {
                background-color: rgba(255, 255, 255, 50);
            }
            QHeaderView::section {
                background-color: rgba(255, 255, 255, 120);
                padding: 8px;
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 220);
                border-right: 1px solid rgba(255, 255, 255, 150);
                font-weight: bold;
                color: #333;
            }
            QHeaderView::section:last { border-right: none; }
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
        prev_lay.addWidget(self.archive_tree)

        # Thin indeterminate progress bar that sits flush on the bottom edge of
        # the preview card; visible only while an import is running. The card's
        # bottom margin is removed (below) so the bar reaches the very edge.
        self.import_progress = QtWidgets.QProgressBar()
        self.import_progress.setObjectName("importProgress")
        self.import_progress.setRange(0, 0)  # indeterminate / busy
        self.import_progress.setTextVisible(False)
        self.import_progress.setFixedHeight(3)
        self.import_progress.setVisible(False)
        self.import_progress.setStyleSheet("""
            QProgressBar#importProgress {
                background: rgba(255, 255, 255, 35);
                border: none;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
            }
            QProgressBar#importProgress::chunk {
                background: rgba(0, 118, 174, 120);
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
            }
        """)
        prev_lay.addWidget(self.import_progress)
        # Pull the bar flush to the card's bottom edge (card uses 12px bottom pad).
        prev_lay.setContentsMargins(14, 12, 14, 0)

        # ---- Status + actions -----------------------------------------
        # The status readout is implicit: a slim, borderless line that shares
        # the action row. Transient progress shows here; per-source import
        # failures surface as warning rows inside the Data to Import tree.
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
        self.btn_import = GlassPushButton(" Import", variant="default")
        self.btn_import.setFixedHeight(34)
        self.btn_import.setEnabled(False)
        self.btn_import.clicked.connect(self._do_import)
        action_row.addWidget(self.btn_cancel)
        action_row.addWidget(self.btn_import)

        self.root.addWidget(src_card)
        self.root.addWidget(prev_card, 1)
        self.root.addLayout(action_row)

    # ------------------------------------------------------------------
    #  Shared hooks
    # ------------------------------------------------------------------
    def on_freeze(self, frozen: bool):
        # During a running import the worker drives enable-state via _set_running;
        # this guards against cross-mode freezes touching our controls.
        for w in (self.btn_add_source, self.btn_clear_sources, self.btn_import):
            w.setDisabled(frozen)

    def on_progress(self, label, pct, color):
        if not label:
            return
        # Tint the implicit status line to match the message severity.
        self._set_status_tint(color)
        self.status_label.setText(label)

    # ------------------------------------------------------------------
    #  Source selection — one "Add Source" picker; backend detects type
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
        """Append new sources (folder or ZIP), de-dup, refresh chips + preview."""
        added = False
        for p in paths:
            p = p.rstrip("/\\") if os.path.isdir(p) else p
            if not p or p in self._sources:
                continue
            kind = self._classify(p)
            if kind is None:
                Log.w(TAG, f"Ignoring unsupported source (not a folder or ZIP): {p}")
                self.status_label.setText(
                    f"Skipped “{os.path.basename(p)}” — only folders and ZIP archives are supported."
                )
                continue
            self._sources.append(p)
            Log.i(TAG, f"Added import source ({kind}): {p}")
            added = True
        if added:
            self._rebuild_chips()
            self._generate_description()

    def _clear_sources(self):
        if not self._sources:
            return
        self._sources = []
        self._rebuild_chips()
        self.archive_tree.clear()
        self.btn_import.setEnabled(False)
        self.status_label.setText("")

    def _remove_source(self, path):
        if path not in self._sources:
            return
        Log.i(TAG, f"Removed import source: {path}")
        chip = self._chip_widgets.get(path)

        def _finalize():
            if path in self._sources:
                self._sources.remove(path)
            self._chip_widgets.pop(path, None)
            self._rebuild_chips()

            def _after_exit():
                if self._sources:
                    self._generate_description()
                else:
                    self.archive_tree.clear()
                    self.btn_import.setEnabled(False)
                    self.status_label.setText("")

            # Fade the current preview out, then rebuild (which fades back in).
            self._animate_tree_exit(_after_exit)

        if chip is None:
            _finalize()
            return

        # Collapse the chip (height -> 0) and fade it, then rebuild.
        eff = QtWidgets.QGraphicsOpacityEffect(chip)
        chip.setGraphicsEffect(eff)
        start_h = chip.sizeHint().height() or chip.height() or 30

        h_anim = QtCore.QPropertyAnimation(chip, b"maximumHeight", self)
        h_anim.setDuration(180)
        h_anim.setEasingCurve(QtCore.QEasingCurve.InCubic)
        h_anim.setStartValue(start_h)
        h_anim.setEndValue(0)

        o_anim = QtCore.QPropertyAnimation(eff, b"opacity", self)
        o_anim.setDuration(180)
        o_anim.setStartValue(1.0)
        o_anim.setEndValue(0.0)

        grp = QtCore.QParallelAnimationGroup(self)
        grp.addAnimation(h_anim)
        grp.addAnimation(o_anim)
        grp.finished.connect(_finalize)
        # Keep a ref so the group isn't garbage-collected mid-animation.
        self._chip_anim = grp
        grp.start()

    # ------------------------------------------------------------------
    #  Source chips
    # ------------------------------------------------------------------
    def _rebuild_chips(self):
        # Remove all chip widgets (keep the persistent placeholder + stretch).
        for w in list(getattr(self, "_chip_widgets", {}).values()):
            self._sources_flow.removeWidget(w)
            w.deleteLater()
        self._chip_widgets = {}

        if not self._sources:
            self.sources_placeholder.setVisible(True)
            self.btn_clear_sources.setEnabled(False)
            return

        self.sources_placeholder.setVisible(False)
        # Insert each chip before the trailing stretch (last layout item).
        for path in self._sources:
            chip = self._make_chip(path)
            self._chip_widgets[path] = chip
            self._sources_flow.insertWidget(self._sources_flow.count() - 1, chip)
        self.btn_clear_sources.setEnabled(True)

    def _make_chip(self, path):
        kind = self._classify(path) or "folder"
        name = os.path.basename(path.rstrip("/\\")) or path

        chip = QtWidgets.QFrame()
        chip.setObjectName("sourceChip")
        chip.setStyleSheet("""
            QFrame#sourceChip {
                background: rgba(255, 255, 255, 150);
                border: 1px solid rgba(255, 255, 255, 200);
                border-radius: 7px;
            }
        """)
        row = QtWidgets.QHBoxLayout(chip)
        row.setContentsMargins(8, 4, 4, 4)
        row.setSpacing(8)

        icon = QtWidgets.QLabel()
        ic = self._icon("archive.svg" if kind == "zip" else "folder.svg")
        if not ic.isNull():
            icon.setPixmap(ic.pixmap(14, 14))
            row.addWidget(icon)

        label = QtWidgets.QLabel(name)
        label.setToolTip(path)
        label.setStyleSheet(
            "QLabel { color: rgba(28, 40, 52, 230); font-size: 12px; "
            "font-weight: 600; background: transparent; }"
        )
        kind_lbl = QtWidgets.QLabel("ZIP" if kind == "zip" else "Folder")
        kind_lbl.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 150); font-size: 10px; background: transparent; }"
        )
        row.addWidget(label)
        row.addWidget(kind_lbl)
        row.addStretch(1)

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
    #  Archive preview (renders every selected source)
    # ------------------------------------------------------------------
    def _generate_description(self):
        if not self._sources:
            self.archive_tree.clear()
            self.btn_import.setEnabled(False)
            return
        try:
            self.btn_import.setEnabled(False)
            self.archive_tree.clear()

            importable = 0
            warned = 0
            for path in self._sources:
                # Each source gets a top-level node labeled by its kind, so a
                # multi-source import reads as distinct trees under one preview.
                split = os.path.split(path)
                is_zip = self._classify(path) == "zip"
                src_label = (
                    (split[1] or path)
                    if not os.path.isdir(path)
                    else (os.path.basename(path.rstrip("/\\")) or path)
                )
                src_root = self._make_item(src_label, "ZIP" if is_zip else "Folder")
                src_root.setData(0, QtCore.Qt.UserRole, path)
                self.archive_tree.addTopLevelItem(src_root)

                ok, warn_msg = self._preview_source(path, is_zip, src_root)
                if ok:
                    importable += 1
                else:
                    warned += 1
                    self._flag_source_root(src_root, warn_msg)

            # Show only the top-level source nodes, fully collapsed. Users can
            # expand a source to drill into its runs and files from there.
            self.archive_tree.collapseAll()

            # Animate the newly rendered source roots into view.
            self._animate_tree_entrance()

            # Implicit status: only summarize when something needs attention.
            if warned:
                self._set_status_tint("r")
                if importable:
                    self.status_label.setText(
                        f"{warned} source(s) can't be imported — see warnings below."
                    )
                else:
                    self.status_label.setText("No importable sources — see warnings below.")
            else:
                self._set_status_tint("b")
                self.status_label.setText("")
            self.btn_import.setEnabled(importable > 0)
        except Exception as e:
            self.archive_tree.clear()
            err = self._make_warning_item(f"Error reading sources: {e}", "Error")
            self.archive_tree.addTopLevelItem(err)
            self._set_status_tint("r")
            self.status_label.setText("Couldn't read the selected sources.")
            self.btn_import.setEnabled(False)

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
                        return False, "Archive is empty — nothing to import."
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
                    return False, "Folder is empty — nothing to import."
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

    def _animate_tree_entrance(self):
        """Fade + lift the Data to Import tree as freshly rendered source roots
        appear. The tree is a plain widget, so a QGraphicsOpacityEffect composes
        cleanly here (unlike on the custom-painted glass frame)."""
        # Stop any in-flight fade so its callback can't touch a stale effect.
        for attr in ("_tree_anim", "_tree_exit_anim"):
            prev = getattr(self, attr, None)
            if prev is not None:
                try:
                    prev.stop()
                except RuntimeError:
                    pass
                setattr(self, attr, None)

        eff = QtWidgets.QGraphicsOpacityEffect(self.archive_tree)
        self.archive_tree.setGraphicsEffect(eff)

        fade = QtCore.QPropertyAnimation(eff, b"opacity", self)
        fade.setDuration(220)
        fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)

        def _clear_effect():
            # Drop the effect once settled so it never interferes with painting.
            self.archive_tree.setGraphicsEffect(None)

        fade.finished.connect(_clear_effect)
        self._tree_anim = fade  # keep a ref so it isn't GC'd mid-flight
        fade.start()

    def _animate_tree_exit(self, on_done):
        """Fade the Data to Import tree out, then invoke on_done (which rebuilds
        the preview and fades it back in). Gives source removal a visible exit."""
        if self.archive_tree.topLevelItemCount() == 0:
            on_done()
            return
        eff = QtWidgets.QGraphicsOpacityEffect(self.archive_tree)
        self.archive_tree.setGraphicsEffect(eff)

        fade = QtCore.QPropertyAnimation(eff, b"opacity", self)
        fade.setDuration(160)
        fade.setEasingCurve(QtCore.QEasingCurve.InCubic)
        fade.setStartValue(1.0)
        fade.setEndValue(0.0)

        def _done():
            self.archive_tree.setGraphicsEffect(None)
            on_done()

        fade.finished.connect(_done)
        self._tree_exit_anim = fade  # keep a ref
        fade.start()

    # ---- Tree population --------------------------------------------------
    def _populate_from_paths(self, names, parent=None):
        """Build tree items (folders AND files) from a flat list of archive paths.

        ``parent`` is the per-source root node; paths nest beneath it. Files are
        always included — visibility is governed by tree expansion, not a toggle.
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
        self.services.run_task(lambda abort: self._import_task(abort, sources))

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
        # Show/hide the thin progress bar (called from the worker thread).
        QtCore.QMetaObject.invokeMethod(
            self.import_progress,
            "setVisible",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, running),
        )

    def _policy_id(self):
        return self.policy_group.checkedId()

    def _import_task(self, abort, sources):
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

                if os.path.isfile(path):
                    copied, skipped = self._import_zip(abort, path)
                    if copied is None:  # invalid/corrupt zip already reported
                        continue  # skip this source, keep going with the rest
                else:
                    copied, skipped = self._import_folder(path)

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

    def _import_zip(self, abort, path):
        policy = self._policy_id()
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

    def _import_folder(self, path):
        policy = self._policy_id()

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

    def _segment_button(self, text, tooltip=""):
        """A checkable, frosted segment for the existing-files policy control —
        matches the sidebar segmented-control language."""
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
    def _segment_qss(checked_bg=True):
        """QSS for a policy segment. When checked_bg is False the checked state
        keeps a transparent background (the sliding pill supplies the fill)."""
        checked = (
            "background: rgba(255, 255, 255, 235); color: rgba(0, 118, 174, 230);"
            if checked_bg
            else "background: transparent; color: rgba(0, 118, 174, 230);"
        )
        return f"""
            QToolButton {{
                background: transparent; border: none; border-radius: 6px;
                color: rgba(40, 50, 65, 190); font-size: 12px; font-weight: 600;
                padding: 0px 10px;
            }}
            QToolButton:hover   {{ background: rgba(255, 255, 255, 80); }}
            QToolButton:checked {{ {checked} }}
        """

    # ---- Policy pill animation -------------------------------------------
    def _on_policy_toggled(self, button, checked):
        if checked:
            self._animate_policy_pill(button)

    def _policy_pill_target(self, button):
        g = button.geometry()
        return QtCore.QRect(g.x(), g.y(), g.width(), g.height())

    def _place_policy_pill(self, button, animate):
        if not hasattr(self, "_policy_pill"):
            return
        target = self._policy_pill_target(button)
        if not animate or not self.policy_segment.isVisible():
            self._policy_pill.setGeometry(target)
            return
        anim = QtCore.QPropertyAnimation(self._policy_pill, b"geometry", self)
        anim.setDuration(200)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.setStartValue(self._policy_pill.geometry())
        anim.setEndValue(target)
        self._policy_pill_anim = anim  # keep ref
        anim.start()

    def _animate_policy_pill(self, button):
        # Geometry is only valid after layout; defer one tick if needed.
        if button.width() <= 1:
            QtCore.QTimer.singleShot(0, lambda: self._place_policy_pill(button, animate=True))
        else:
            self._place_policy_pill(button, animate=True)

    def on_enter(self):
        # Position the pill under the current selection without animating.
        checked = self.policy_group.checkedButton() or self.rb_merge
        QtCore.QTimer.singleShot(0, lambda: self._place_policy_pill(checked, animate=False))

    def eventFilter(self, obj, event):
        if (
            obj is getattr(self, "policy_segment", None)
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            checked = self.policy_group.checkedButton() or self.rb_merge
            self._place_policy_pill(checked, animate=False)
        return super().eventFilter(obj, event)

    def _icon(self, name):
        try:
            from QATCH.common.architecture import Architecture

            path = os.path.join(Architecture.get_path(), "QATCH", "icons", name)
            if os.path.exists(path):
                return QtGui.QIcon(path)
        except Exception:
            pass
        return QtGui.QIcon()

    def _card(self, title):
        card = QtWidgets.QFrame()
        card.setObjectName("glassPanel")
        card.setStyleSheet("""
            QFrame#glassPanel {
                background: rgba(255, 255, 255, 90);
                border: 1px solid rgba(255, 255, 255, 230);
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

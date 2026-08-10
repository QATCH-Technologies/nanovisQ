"""
QATCH.ui.widgets.run_info_overlay

Glassmorphic "Run Info" overlay - same overlay shell convention as
`QATCH.ui.widgets.data_management_widget.DataManagementWidget`,
`QATCH.ui.widgets.user_profiles_manager_widget.UserProfilesManagerWidget`,
and `QATCH.ui.widgets.user_preferences_widget.UserPreferencesWidget`: a child
widget reparented over the app's central widget, dimmed scrim + fade-in/out
glass panel, click-outside-to-dismiss, and a fullscreen toggle. The overlay
lifecycle itself (init scaffolding, fade animations, parent-tracking
geometry fit, scrim paint, click-outside dismiss) is shared with those three
widgets via `QATCH.ui.components.overlay_shell.OverlayLifecycleMixin` - only
this widget's own content (one or more `QueryRunInfoWidget` forms, plus a
"common fields" row shared across ports when saving a multi-port capture)
lives here.

Replaces the old `query_run_info_widget.QueryRunInfoWidget.show()` position-
override and `run_info_widget.RunInfoWindow` (a standalone, desktop-centered,
`QDockWidget`-based top-level window) with a single overlay that handles both
the single-port and multi-port cases uniformly via `open_runs()`.
"""

from __future__ import annotations

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.components import LabeledToggle, QATCHLineEdit, QATCHPanel, QATCHPushButton
from QATCH.ui.components.overlay_shell import (
    FULLSCREEN_ANIM_EASING,
    OverlayLifecycleMixin,
    rebuild_fullscreen_icons,
    run_variant_animation,
)
from QATCH.ui.dialogs.pop_up_dialog import PopUp
from QATCH.ui.styles.theme_manager import (
    ThemeManager,
    caption_label_qss,
    field_label_qss,
    glass_panel_qss,
    tok_css,
)
from QATCH.ui.widgets.query_run_info_widget import QueryRunInfoWidget

TAG = "[RunInfoOverlay]"

_ICONS_DIR = os.path.join(Architecture.get_path(), "QATCH", "icons")


class RunInfoOverlay(OverlayLifecycleMixin, QtWidgets.QWidget):
    """Container overlay for one or more `QueryRunInfoWidget` forms.

    A singleton per app window (lazily created/cached by the caller, see
    `ControlsWindow._ensure_run_info_overlay`), reused across invocations -
    only its port content is torn down and rebuilt on each `open_runs()`
    call, the chrome (scrim/header/close/fullscreen) persists.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ICON_MAIN = os.path.join(_ICONS_DIR, "info-circle.svg")
        self.ICON_EXPAND = os.path.join(_ICONS_DIR, "expand.svg")
        self.ICON_COLLAPSE = os.path.join(_ICONS_DIR, "collapse.svg")

        self._forms: list[QueryRunInfoWidget] = []
        self._port_cards: list[QtWidgets.QWidget] = []
        self._port_card_layouts: list[QtWidgets.QVBoxLayout] = []
        self._post_run = False
        self._unsaved_changes = False
        self._forms_confirmed_done = False
        self._portIDfromIndex = lambda pid: hex(pid)[2:].upper()

        self._captions: list[QtWidgets.QLabel] = []
        self._field_labels: list[QtWidgets.QLabel] = []

        self._init_overlay_shell(
            parent,
            "runinfoview",
            panel_alpha=215,
            margin_pct=0.175,
            content_margins=(20, 14, 20, 20),
            content_spacing=14,
        )

        self.header_layout = self._build_overlay_header(self.ICON_MAIN, "Run Info", fullscreen=True)
        self.main_layout.addLayout(self.header_layout)

        self._build_common_row()
        self.main_layout.addWidget(self.common_row)

        self._build_ports_area()
        self.main_layout.addWidget(self.ports_scroll, 1)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

        self._finish_overlay_shell()

    # ------------------------------------------------------------------
    #  Build: common-fields row (shared run name/batch/notes for multi-port)
    # ------------------------------------------------------------------
    def _build_common_row(self) -> None:
        self.common_row = QATCHPanel()
        outer = QtWidgets.QVBoxLayout(self.common_row)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        self.cap_common = self._caption("Enter Run Info (All Ports)")
        outer.addWidget(self.cap_common)

        path_row = QtWidgets.QHBoxLayout()
        self.l_runpath = self._field_label("Saved Run To")
        path_row.addWidget(self.l_runpath)
        path_row.addStretch()
        self.cpy_runpath = QtWidgets.QLabel("Copied!")
        self.cpy_runpath.setVisible(False)
        path_row.addWidget(self.cpy_runpath, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.cb_runpath = QtWidgets.QLabel("&#x1F4CB;")  # clipboard icon
        self.cb_runpath.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.cb_runpath.setToolTip("Copy path to clipboard")
        self.cb_runpath.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.cb_runpath.mousePressEvent = self.copyText
        path_row.addWidget(self.cb_runpath, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        outer.addLayout(path_row)
        self.t_runpath = QtWidgets.QPlainTextEdit()
        self.t_runpath.setObjectName("runInfoOverlayPath")
        self.t_runpath.setReadOnly(True)
        self.t_runpath.setFixedHeight(50)
        outer.addWidget(self.t_runpath)

        fields_row = QtWidgets.QHBoxLayout()

        runname_col = QtWidgets.QVBoxLayout()
        rn_row = QtWidgets.QHBoxLayout()
        self.l_runname = self._field_label("Run Name")
        rn_row.addWidget(self.l_runname)
        self.t_runname = QATCHLineEdit()
        self.t_runname.textChanged.connect(self._detect_change)
        self.t_runname.editingFinished.connect(self._update_hidden_child_fields)
        rn_row.addWidget(self.t_runname)
        runname_col.addLayout(rn_row)

        batch_row = QtWidgets.QHBoxLayout()
        self.l_batch = self._field_label("Batch Number")
        batch_row.addWidget(self.l_batch)
        self.t_batch = QATCHLineEdit()
        self.t_batch.textChanged.connect(self._detect_change)
        self.t_batch.textEdited.connect(self.prevent_duplicate_scans)
        self.t_batch.editingFinished.connect(self._update_hidden_child_fields)
        batch_row.addWidget(self.t_batch)
        runname_col.addLayout(batch_row)

        self.blankIcon = QtGui.QIcon()
        self.foundIcon = QtGui.QIcon(os.path.join(_ICONS_DIR, "checkmark-circle.svg"))
        self.missingIcon = QtGui.QIcon(os.path.join(_ICONS_DIR, "warning.svg"))
        self.t_batchAction = self.t_batch.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.t_batchAction.triggered.connect(self.find_batch_num)
        self.t_batch.textChanged.connect(self.find_batch_num)
        self.t_batch.editingFinished.connect(self.find_batch_num)
        self.batch_found = False

        fields_row.addLayout(runname_col, 1)

        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setObjectName("runInfoOverlayNotes")
        self.notes.setPlaceholderText("Notes")
        self.notes.textChanged.connect(self._detect_change)
        fields_row.addWidget(self.notes, 1)
        outer.addLayout(fields_row)

        self.q_recall = LabeledToggle("Remember for next run", compact=True)
        self.q_recall.setChecked(True)
        self.q_recall.toggled.connect(self._detect_change)
        self.q_recall.toggled.connect(self._update_hidden_child_fields)
        outer.addWidget(self.q_recall, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        self.btn_save_all = QATCHPushButton("Save All", variant="primary")
        self.btn_save_all.pressed.connect(self._confirm_all)
        outer.addWidget(self.btn_save_all)

        # scannow highlight widget for the common batch field
        self.l_scannow = QtWidgets.QWidget(self.common_row)
        self.l_scannow.setVisible(False)

        QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Enter), self.common_row, activated=self._confirm_all
        )
        QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Return), self.common_row, activated=self._confirm_all
        )

    # ------------------------------------------------------------------
    #  Build: ports area (grid of port cards, or a single embedded form)
    # ------------------------------------------------------------------
    def _build_ports_area(self) -> None:
        self.ports_scroll = QtWidgets.QScrollArea()
        self.ports_scroll.setObjectName("runInfoPortsScroll")
        self.ports_scroll.setWidgetResizable(True)
        self.ports_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        # A QScrollArea and its viewport otherwise paint Qt's opaque default
        # widget background (light gray, independent of the app theme) -
        # make both transparent so the glass_frame's own themed background
        # (see _apply_panel_appearance) shows through instead of a stale
        # untinted rectangle behind the port cards.
        self.ports_scroll.setStyleSheet(
            "QScrollArea#runInfoPortsScroll { background: transparent; border: none; }"
        )
        self.ports_scroll.viewport().setStyleSheet("background: transparent;")
        self.ports_container = QtWidgets.QWidget()
        self.ports_container.setObjectName("runInfoPortsContainer")
        self.ports_container.setStyleSheet(
            "QWidget#runInfoPortsContainer { background: transparent; }"
        )
        self.ports_layout = QtWidgets.QGridLayout(self.ports_container)
        self.ports_layout.setContentsMargins(0, 0, 0, 0)
        self.ports_layout.setSpacing(12)
        self.ports_scroll.setWidget(self.ports_container)

    # ------------------------------------------------------------------
    #  Themed label helpers (mirrors QueryRunInfoWidget's convention)
    # ------------------------------------------------------------------
    def _caption(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text.upper())
        lbl.setStyleSheet(caption_label_qss())
        self._captions.append(lbl)
        return lbl

    def _field_label(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(f"{text}\t=")
        lbl.setStyleSheet(field_label_qss())
        self._field_labels.append(lbl)
        return lbl

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        self._refresh_header_theme()
        for lbl in self._captions:
            lbl.setStyleSheet(caption_label_qss())
        for lbl in self._field_labels:
            lbl.setStyleSheet(field_label_qss())
        tok = ThemeManager.instance().tokens()
        for name, obj in (("runInfoOverlayPath", self.t_runpath), ("runInfoOverlayNotes", self.notes)):
            obj.setStyleSheet(
                f"QPlainTextEdit#{name} {{"
                f"  background: {tok_css(tok['flat_surface'])};"
                f"  border: 1px solid {tok_css(tok['flat_border'])};"
                "  border-radius: 7px;"
                f"  color: {tok_css(tok['flat_text'])};"
                "  padding: 8px 10px;"
                "}"
                f"QPlainTextEdit#{name}:focus {{"
                f"  border: 1px solid {tok_css(tok['flat_accent'])};"
                "}"
            )
        # The outer glass_frame's QSS (background/border/radius) is only
        # ever (re)applied by _apply_panel_appearance, which _refit_to_parent
        # otherwise only calls on resize/move/fullscreen-toggle - without
        # this, a live theme flip leaves the outer panel's background frozen
        # at whatever it was when the overlay was constructed even though
        # every field/card inside it (each repainting itself independently)
        # does pick up the new tokens immediately.
        self._apply_panel_appearance(self._current_margin_frac())

    def _rebuild_fs_icons(self) -> None:
        rebuild_fullscreen_icons(self, self.ICON_EXPAND, self.ICON_COLLAPSE)

    def _apply_panel_appearance(self, frac: float) -> None:
        p = 0.0 if self._default_margin_pct <= 0 else min(1.0, frac / self._default_margin_pct)
        alpha = int(255 + (215 - 255) * p)
        border = 1.5 * p
        radius = 12.0 * p
        self.glass_frame.setStyleSheet(glass_panel_qss("runinfoview", alpha, border, radius))

    # Single-run "wizard" sessions (see QueryRunInfoWidget._enter_wizard_mode)
    # are designed as a narrow, centered, tall-ish card rather than the wide
    # percentage-inset panel every other overlay (and this overlay's own
    # multi-port grid) uses - each step is meant to fit without scrolling,
    # which only reads right at a phone/dialog-like width. Multi-port keeps
    # the inherited percentage-inset behavior untouched.
    _WIZARD_TARGET_WIDTH = 620
    _WIZARD_TARGET_HEIGHT = 640
    _WIZARD_MIN_MARGIN = 24

    def _apply_margin_frac(self, frac: float) -> None:
        if len(self._forms) != 1:
            super()._apply_margin_frac(frac)
            return
        w, h = self.width(), self.height()
        # Blend between the wizard's fixed-size-centered margin (p=1, i.e.
        # not fullscreen) and 0 (p=0, fullscreen) on the same curve
        # _apply_panel_appearance already uses for alpha/border/radius, so
        # the fullscreen-toggle animation interpolates smoothly instead of
        # snapping between two unrelated margin models.
        p = 0.0 if self._default_margin_pct <= 0 else min(1.0, frac / self._default_margin_pct)
        target_mx = max(self._WIZARD_MIN_MARGIN, (w - self._WIZARD_TARGET_WIDTH) // 2)
        target_my = max(self._WIZARD_MIN_MARGIN, (h - self._WIZARD_TARGET_HEIGHT) // 2)
        mx = int(target_mx * p)
        my = int(target_my * p)
        self.base_layout.setContentsMargins(mx, my, mx, my)
        self._apply_panel_appearance(frac)
        self._position_overlay_buttons(mx, my)

    def toggle_fullscreen(self) -> None:
        self._is_fullscreen = not self._is_fullscreen
        self._rebuild_fs_icons()
        target = 0.0 if self._is_fullscreen else self._default_margin_pct
        start = self._default_margin_pct if self._is_fullscreen else 0.0

        def _step(t):
            self._apply_margin_frac(start + (target - start) * t)

        run_variant_animation(
            self, "_fs_anim", duration=240, easing=FULLSCREEN_ANIM_EASING, on_step=_step
        )

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------
    def open_runs(self, forms: list) -> None:
        """Populates the overlay with `forms` (one `QueryRunInfoWidget` per
        captured port) and reveals it.

        If a previous session is still open and unfinished (the user hasn't
        saved/dismissed it yet), the incoming forms are merged into that
        session instead of discarding it - this overlay is a shared
        singleton per app window (unlike the old design, where each capture
        batch got its own independent top-level window), so a second batch
        finishing while the first is still awaiting input must not silently
        drop the user's in-progress entry.
        """
        if self._forms and self.isVisible() and not self._forms_confirmed_done:
            Log.w(
                tag=TAG,
                msg="Run Info overlay already open - merging new run(s) into the current session.",
            )
            self._rebuild(self._forms + list(forms))
        else:
            self._rebuild(list(forms))

    def _rebuild(self, forms: list) -> None:
        """Tears down any existing port content and rebuilds it from
        `forms` - the overlay's chrome (scrim/header/close/fullscreen) is
        reused across calls, only the port content is rebuilt."""
        self._teardown_forms()

        self._forms = list(forms)
        self._forms_confirmed_done = False
        num_runs = len(self._forms)
        if num_runs == 0:
            Log.w(tag=TAG, msg="open_runs() called with no forms - nothing to show.")
            return

        for i, form in enumerate(self._forms):
            form.setRuns(num_runs, i)

        first_name, first_path, _, _, _ = self._forms[0].getRunParams()
        self._post_run = self._forms[0].post_run

        if num_runs == 1:
            self.window_title_label.setText("Run Info")
            self.common_row.setVisible(False)
            self._port_cards = [self._forms[0]]
            self.ports_layout.addWidget(self._forms[0], 0, 0)
            self._forms[0].finished.connect(self._on_single_form_finished)
        else:
            self.window_title_label.setText(f"Run Info ({num_runs} Ports)")
            self.common_row.setVisible(True)
            self._unsaved_changes = self._post_run

            run_name = first_name[: first_name.rindex("_")] if "_" in first_name else first_name
            self.t_runname.setText(run_name)
            self.t_batch.setText("")
            self.notes.setPlainText("")
            self.q_recall.setChecked(True)

            all_run_paths = [
                os.path.join(os.getcwd(), f.getRunParams()[1]) for f in self._forms
            ]
            try:
                common_path = os.path.commonpath(all_run_paths)
            except ValueError:
                common_path = os.path.dirname(first_path)
            self.t_runpath.setPlainText(common_path)

            self._port_cards = []
            self._port_card_layouts = []
            for i, form in enumerate(self._forms):
                card = QATCHPanel()
                vbox = QtWidgets.QVBoxLayout(card)
                vbox.setContentsMargins(12, 10, 12, 10)
                vbox.setSpacing(6)
                vbox.addWidget(self._caption(f"Port {self._portIDfromIndex(i + 1)}"))
                vbox.addWidget(form)
                # Each port card now hosts a full wizard (stepper +
                # composition table, ~620px designed width - see
                # QueryRunInfoWidget._enter_wizard_mode()), which needs more
                # room than the old compact flat card did at 4-wide. 2
                # columns gives a 2x2 grid for the common 4-port case
                # (matches the 1-4 port clamp in main_window.py's multiplex
                # handling) without cramping any one card.
                row, col = divmod(i, 2)
                self.ports_layout.addWidget(card, row, col)
                self._port_cards.append(card)
                self._port_card_layouts.append(vbox)

            self._update_hidden_child_fields()
            QtCore.QTimer.singleShot(500, self.showScanNow)
            QtCore.QTimer.singleShot(1000, self.flashScanNow)
            self.t_batch.setFocus()

        self.setVisible(True)

    def _teardown_forms(self) -> None:
        """Removes any port cards/forms left over from a previous
        `open_runs()` call before rebuilding - the overlay instance is
        reused across invocations, its content is not."""
        for form in self._forms:
            try:
                form.finished.disconnect()
            except TypeError:
                pass  # nothing was connected
        while self.ports_layout.count():
            item = self.ports_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                if w not in self._forms:
                    w.deleteLater()
        for form in self._forms:
            form.setParent(None)
            form.deleteLater()
        self._forms = []
        self._port_cards = []
        self._port_card_layouts = []

    # ------------------------------------------------------------------
    #  Common-row behavior (ported from the old RunInfoWindow)
    # ------------------------------------------------------------------
    def _detect_change(self, *_args) -> None:
        self._unsaved_changes = True

    def _update_hidden_child_fields(self, *_args) -> None:
        run_name = self.t_runname.text()
        batch_num = self.t_batch.text()
        notes_txt = self.notes.toPlainText()
        do_recall = self.q_recall.isChecked()
        for form in self._forms:
            form.setHiddenFields(run_name, batch_num, notes_txt, do_recall)

    def prevent_duplicate_scans(self) -> None:
        current_text = self.t_batch.text()
        min_batch_num_len = 3
        given_batch_num_len = len(current_text)
        split_at_idx = int(given_batch_num_len / 2)
        if split_at_idx > min_batch_num_len and given_batch_num_len % 2 == 0:
            first_half = current_text[:split_at_idx]
            second_half = current_text[split_at_idx:]
            if first_half == second_half:
                Log.w(tag=TAG, msg=f"Duplicate scan ignored: {second_half}")
                self.t_batch.setText(first_half)

    def find_batch_num(self) -> None:
        batch = self.t_batch.text().strip()
        found = False
        if len(batch) == 0:
            self.t_batchAction.setIcon(self.blankIcon)
        elif Constants.get_batch_param(self.t_batch.text()):
            self.t_batchAction.setIcon(self.foundIcon)
            found = True
        else:
            self.t_batchAction.setIcon(self.missingIcon)
        if self.batch_found != found:
            self.batch_found = found
            self._detect_change()

    def showScanNow(self) -> None:
        if len(self._forms) <= 1:
            return
        self.l_scannow.resize(self.t_batch.size())
        self.l_scannow.move(self.t_batch.pos())
        self.l_scannow.setObjectName("scannow")
        self.l_scannow.setStyleSheet(
            "#scannow { background-color: #F5FE49; border: 1px solid #7A7A7A; }"
        )
        h_scannow = QtWidgets.QHBoxLayout()
        h_scannow.setContentsMargins(3, 0, 6, 0)
        self.t_scannow = QtWidgets.QLabel("Scan or enter now!")
        h_scannow.addWidget(self.t_scannow)
        h_scannow.addStretch()
        i_scannow = QtWidgets.QLabel()
        i_scannow.setPixmap(
            QtGui.QPixmap(os.path.join(_ICONS_DIR, "barcode.svg")).scaledToHeight(
                max(1, self.l_scannow.height() - 2)
            )
        )
        h_scannow.addWidget(i_scannow)
        self.l_scannow.setLayout(h_scannow)
        self.l_scannow.setVisible(True)
        self.t_batch.textEdited.connect(self.l_scannow.hide)

    def flashScanNow(self) -> None:
        if self.l_scannow.isVisible():
            if not self.t_batch.hasFocus():
                self.l_scannow.hide()
            elif self.t_scannow.styleSheet() == "":
                self.t_scannow.setStyleSheet("color: #F5FE49;")
                QtCore.QTimer.singleShot(250, self.flashScanNow)
            else:
                self.t_scannow.setStyleSheet("")
                QtCore.QTimer.singleShot(500, self.flashScanNow)

    def copyText(self, _event) -> None:
        try:
            cb = QtWidgets.QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(self.t_runpath.toPlainText(), mode=cb.Clipboard)
        except Exception as e:
            Log.e(tag=TAG, msg=f"Clipboard Error: {e}")
            return
        self.cpy_runpath.show()
        QtCore.QTimer.singleShot(3000, self.cpy_runpath.hide)

    # ------------------------------------------------------------------
    #  Save / close
    # ------------------------------------------------------------------
    def _mark_port_saved(self, i: int) -> None:
        if i >= len(self._port_card_layouts):
            return
        vbox = self._port_card_layouts[i]
        while vbox.count():
            item = vbox.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._forms[i]:
                w.deleteLater()
        saved_lbl = QtWidgets.QLabel("Saved!")
        saved_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        saved_lbl.setStyleSheet(caption_label_qss())
        vbox.addWidget(saved_lbl)

    def _confirm_all(self, force: bool = False) -> bool:
        """Saves every port form in order - ported from the old
        `RunInfoWindow.confirm()`. Aborts (returning False) and rewires
        itself as a retry on the paused form's `finished` signal if a form
        needs more input (e.g. a signature dialog) before it can proceed."""
        if len(self._forms) <= 1:
            # Single-port sessions save via the form's own Save button/
            # Enter shortcut - this only drives the multi-port common row
            # (hidden in that case, but its window-scoped Enter/Return
            # shortcuts can still reach here; no-op rather than double-save).
            return False
        for i, form in enumerate(self._forms):
            if not form.isVisible():
                Log.d(tag=TAG, msg=f"Skipping RUN_IDX = {i} (already saved)")
                self._mark_port_saved(i)
                continue
            if i == 0:
                self._update_hidden_child_fields()
            Log.d(tag=TAG, msg=f"Saving RUN_IDX = {i}")
            if not form.confirm(force):
                try:
                    form.finished.disconnect(self._confirm_all)
                except TypeError:
                    pass
                form.finished.connect(self._confirm_all)
                Log.d(tag=TAG, msg=f"Save paused at RUN_IDX {i}/{len(self._forms)}")
                return False
            self._mark_port_saved(i)
        self._unsaved_changes = False
        self._forms_confirmed_done = True
        self._deferred_close()
        return True

    def _on_single_form_finished(self) -> None:
        self._forms_confirmed_done = True
        self._deferred_close()

    def _deferred_close(self) -> None:
        """Defers to the next event-loop tick before re-calling `self.close()`.

        Qt's `QWidget.close()` has an internal re-entrancy guard: a nested
        `close()` call made synchronously from within this same widget's
        own close chain (e.g. here, where a form finishing triggers this
        from inside the veto delegate's call stack a few frames below the
        original `close()` invocation) is silently swallowed rather than
        re-running `closeEvent` - so `isVisible()` never actually flips to
        `False`. Scheduling it for the next tick lets the outer `close()`
        call return and clear that guard first, so this one actually runs.
        """
        QtCore.QTimer.singleShot(0, self.close)

    def _request_close_multi(self) -> None:
        unsaved = self._unsaved_changes or any(
            f.unsaved_changes for f in self._forms if f.isVisible()
        )
        if unsaved:
            res = PopUp.question(
                self,
                Constants.app_title,
                "You have unsaved changes!\n\nAre you sure you want to close this window?",
                False,
            )
            if not res:
                return  # veto - stay open
            if self._post_run:
                try:
                    self._confirm_all(force=True)
                except Exception as e:
                    Log.e(tag=TAG, msg=str(e))
        self._forms_confirmed_done = True
        self._deferred_close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Intercepts close to run the same unsaved-changes veto the old
        standalone `QueryRunInfoWidget`/`RunInfoWindow` did, delegating the
        actual check to the single form (single-port) or replaying the
        aggregate multi-port check, before handing off to
        `OverlayFadeMixin.closeEvent`'s fade-out."""
        if self._closing or self._forms_confirmed_done:
            super().closeEvent(event)
            return
        event.ignore()
        if not self._forms:
            self._forms_confirmed_done = True
            self._deferred_close()
            return
        if len(self._forms) == 1:
            self._forms[0].close()
        else:
            self._request_close_multi()

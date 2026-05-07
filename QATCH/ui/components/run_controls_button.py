from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class StartStopButton(QToolButton):
    """A custom Start/Stop/Progress button that displays progress and success animations.

    This widget draws a circular progress ring, handles state transitions between
    running, stopped, and completed, and renders specific icons (Play, Stop, Checkmark)
    based on the current state.

    Attributes:
        progress (float): Current progress value ranging from 0.0 to 1.0.
        is_running (bool): Flag indicating if the button is in the 'running' (stop icon) state.
        is_complete (bool): Flag indicating if the task has finished successfully.
        success_angle (float): Current angle for the success animation arc.
        animating_success (bool): Flag indicating if the success animation is currently active.
        color_blue (QColor): Color used for the active progress ring.
        color_green (QColor): Color used for the success state.
        color_track (QColor): Color of the background ring track.
        color_icon (QColor): Default color for the internal icons.
        color_disabled (QColor): Color used when the widget is disabled.
        success_anim (QVariantAnimation): Animation object for the success spin effect.
        progress_anim (QVariantAnimation): Animation object for smooth progress updates.
    """

    def __init__(self, parent=None):
        """Initializes the button with default states, colors, and animations.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setAutoRaise(True)
        self.progress = 0.0
        self.is_running = False
        self.is_complete = False
        self.success_angle = 0
        self.animating_success = False
        self.color_blue = QColor("#00A3DA")
        self.color_green = QColor("#4CAF50")
        self.color_track = QColor("#9E9E9E")
        self.color_icon = QColor("#696969")
        self.color_disabled = QColor("#696969")
        self.color_darkgreen = QColor("#25B101")
        self.color_darkred = QColor("#FA4A3E")

        # Success Animation
        self.success_anim = QVariantAnimation()
        self.success_anim.setStartValue(0)
        self.success_anim.setEndValue(360)
        self.success_anim.setDuration(800)
        self.success_anim.valueChanged.connect(self._update_success_anim)
        self.success_anim.finished.connect(self._finish_success_anim)

        # Progress Animation
        self.progress_anim = QVariantAnimation()
        self.progress_anim.setDuration(500)
        self.progress_anim.setEasingCurve(QEasingCurve.OutQuad)
        self.progress_anim.valueChanged.connect(self._update_progress_anim)

    def animate_progress(self, target_value):
        """Smoothly interpolates the blue ring to a new progress value.

        Args:
            target_value (float): The progress value to animate towards (0.0 to 1.0).
        """
        self.progress_anim.stop()
        self.progress_anim.setStartValue(self.progress)
        self.progress_anim.setEndValue(target_value)
        self.progress_anim.start()

    def _update_progress_anim(self, value):
        """Slot called by the progress animation to update the ring value.

        Args:
            value (float): The current interpolated progress value.
        """
        self.progress = value
        self.update()

    def trigger_success(self):
        """Transitions the button to the success state.

        Stops any active progress animation, sets the state to complete,
        and initiates the success animation.
        """
        # Stop progress animation if we reach a successful fill
        self.progress_anim.stop()

        self.is_running = False
        self.is_complete = True
        self.animating_success = True
        self.success_anim.start()
        self.update()

    def reset(self):
        """Resets the button to its initial idle state.

        Clears progress, stops animations, and resets internal flags.
        """
        self.progress_anim.stop()
        self.is_running = False
        self.is_complete = False
        self.progress = 0.0
        self.animating_success = False
        self.update()

    def _update_success_anim(self, value):
        """Slot called by the success animation to update the angle.

        Args:
            value (int): The current angle of the success arc.
        """
        self.success_angle = value
        self.update()

    def _finish_success_anim(self):
        """Slot called when the success animation finishes to clean up state."""
        self.animating_success = False
        self.update()

    def update(self):
        """Overrides the default update method to trigger a repaint."""
        super().update()

        icon_size = QtCore.QSize(30, 30)
        self.setIcon(self._make_icon(icon_size))
        self.setIconSize(icon_size)

    def _make_icon(self, size: QtCore.QSize) -> QtGui.QIcon:
        """Handles the custom painting of the widget.

        Draws the three main components, the background track ring, the active progress ring/success ring,
        and the central icon.

        Args:
            size (QSize): The size of the QIcon to create.

        Returns:
            QIcon: The custom drawn icon representing the button state and progress.
        """
        pm = QtGui.QPixmap(size)
        pm.fill(Qt.transparent)

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)

        # Custom drawing
        rect = pm.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 - 2

        # Coloring
        if not self.isEnabled():
            track_color = self.color_disabled
            icon_color = self.color_disabled
            ring_color = self.color_disabled
        elif self.is_complete:
            track_color = self.color_track
            icon_color = self.color_green
            ring_color = self.color_green
        elif self.is_running:
            track_color = self.color_track
            icon_color = self.color_icon
            ring_color = self.color_blue
        else:  # enabled, not running or finished
            track_color = self.color_icon
            icon_color = self.color_icon
            ring_color = self.color_disabled

        #  Progress track
        painter.setPen(QPen(track_color, 2.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)

        # Active ring
        if self.isEnabled():
            if self.is_complete:
                # On success, trigger success animation.
                pen = QPen(ring_color, 2.5, Qt.SolidLine, Qt.RoundCap)
                painter.setPen(pen)

                if self.animating_success:
                    start = 90 * 16
                    span = -self.success_angle * 16
                    painter.drawArc(
                        QRectF(
                            center.x() - radius,
                            center.y() - radius,
                            radius * 2,
                            radius * 2,
                        ),
                        int(start),
                        int(span),
                    )
                else:
                    painter.drawEllipse(center, radius, radius)

            elif self.is_running and self.progress > 0:
                # While running, trigger progress animation.
                pen = QPen(ring_color, 2.5, Qt.SolidLine, Qt.RoundCap)
                painter.setPen(pen)
                angle_span = -self.progress * 360 * 16
                painter.drawArc(
                    QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
                    90 * 16,
                    int(angle_span),
                )

        # Render icons
        painter.setPen(Qt.NoPen)
        icon_size = 2 * radius * 0.75

        # NOTE: These have to be drawn dynamically to animate them.  The icons in the icon directory cannot be
        # animated properly so the checkmark is manually drawn, the Stop icon is drawn as a square, and the
        # Start icon is drawn as a triangle.
        if self.is_complete:
            # Checkmark
            painter.setBrush(QBrush(self.color_darkgreen))
            path = QPainterPath()
            path.moveTo(center.x() - icon_size * 0.4, center.y())
            path.lineTo(center.x() - icon_size * 0.1, center.y() + icon_size * 0.3)
            path.lineTo(center.x() + icon_size * 0.4, center.y() - icon_size * 0.4)

            check_pen = QPen(icon_color, 2.5)
            check_pen.setCapStyle(Qt.RoundCap)
            painter.strokePath(path, check_pen)

        elif self.is_running:
            # Stop Square
            painter.setBrush(QBrush(self.color_darkred))
            s = icon_size * 0.5
            painter.drawRect(QRectF(center.x() - s / 2, center.y() - s / 2, s, s))

        else:
            # Start Triangle
            painter.setBrush(QBrush(self.color_darkgreen))
            path = QPainterPath()
            h = icon_size * 0.6
            w = icon_size * 0.5
            x = center.x() - (w / 2) + 1.5
            y = center.y() - (h / 2)
            path.moveTo(x, y)
            path.lineTo(x + w, center.y())
            path.lineTo(x, y + h)
            path.closeSubpath()
            painter.drawPath(path)

        painter.end()

        return QtGui.QIcon(pm)


class RunControls(QWidget):
    """A composite RunControls widget combining a StartStopButton with a sliding status label.

    This control acts as a run controller. When the button is clicked, it emits
    signals to start or stop a process. When running, a status label slides out
    to the right to display textual progress details.

    Attributes:
        startRequested (pyqtSignal): Signal emitted when the user requests to start.
        stopRequested (pyqtSignal): Signal emitted when the user requests to stop.
        layout (QHBoxLayout): Main horizontal layout.
        btn (StartStopButton): The custom circular button instance with text label.
        status_container (QFrame): The collapsible container for the status text.
        status_label (QLabel): The label displaying current step information.
        anim (QPropertyAnimation): Animation for sliding the status container.
    """

    startRequested = pyqtSignal()
    stopRequested = pyqtSignal()

    def __init__(self, parent=None):
        """Initializes the UnifiedProgressControl.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Left container is button and label.
        self.btn = StartStopButton()
        # QSize taken from sizeHint() of tool_Initialize and/or tool_Reset
        self.btn.setFixedSize(QtCore.QSize(60, 56))
        self.btn.setText("Start")
        self.btn.clicked.connect(self.toggle_state)
        self.layout.addWidget(self.btn)

        # Right sliding status container.
        self.status_container = QFrame()
        self.status_container.setFixedWidth(0)
        self.status_container.setStyleSheet("background-color: transparent;")

        self.status_layout = QVBoxLayout(self.status_container)
        self.status_layout.setContentsMargins(5, 5, 5, 5)
        self.status_layout.setAlignment(Qt.AlignBottom)

        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #555; font-size: 12px; font-weight: bold;")
        self.status_label.setWordWrap(True)
        self.status_layout.addWidget(self.status_label)

        self.lbl_status = QLabel("Run Status")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #333; font-size: 11px; margin-top: 2px;")
        self.status_layout.addWidget(self.lbl_status)

        self.layout.addWidget(self.status_container)

        self.anim = QPropertyAnimation(self.status_container, b"minimumWidth")
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.anim.setDuration(1000)

    def setEnabled(self, enabled):
        """Sets the enabled state of the control and its sub-widgets.

        Args:
            enabled (bool): True to enable, False to disable.
        """
        super().setEnabled(enabled)
        self.btn.setEnabled(enabled)
        self.btn.update()

    def toggle_state(self):
        """Toggles the state based on the current button status.

        Emits `stopRequested` if the button is currently running or complete.
        Emits `startRequested` if the button is idle.
        """
        if self.btn.is_complete:
            self.stopRequested.emit()
        elif self.btn.is_running:
            self.stopRequested.emit()
        else:
            self.startRequested.emit()

    def set_running(self, running=True):
        """Updates the UI to reflect whether a process is running.

        If running is True, the status container slides open. If False,
        it slides closed.

        Args:
            running (bool, optional): The target running state. Defaults to True.
        """
        if self.btn.is_running == running and not self.btn.is_complete:
            return

        if running:
            self.btn.is_complete = False
            self.btn.setText("Stop")
            self.anim.setStartValue(0)
            self.anim.setEndValue(160)
            self.anim.start()
        else:
            self.btn.is_complete = False
            self.btn.progress = 0.0
            self.btn.setText("Start")
            self.anim.setStartValue(160)
            self.anim.setEndValue(0)
            self.anim.start()

        self.btn.is_running = running
        self.btn.update()

    def update_progress(self, current_step, max_steps, fill_type_text):
        """Updates the progress button and status label text.

        If `current_step` meets or exceeds `max_steps`, the control triggers
        the success state on the button.

        Args:
            current_step (int): The current step number in the process.
            max_steps (int): The total number of steps.
            fill_type_text (str): Descriptive text to display in the status label.
        """
        if current_step >= max_steps and max_steps > 0:
            if not self.btn.is_complete:
                self.btn.trigger_success()
                self.status_label.setText(f"Done ({fill_type_text})")
                self.btn.setText("Done")
            return

        if max_steps > 0:
            percentage = current_step / max_steps
        else:
            percentage = 0
        self.btn.animate_progress(percentage)
        self.status_label.setText(f"{fill_type_text}")

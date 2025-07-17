from PyQt5 import QtCore, QtGui


class IconUtils:
    """Utility class for handling icon transformations in PyQt5 applications."""

    @staticmethod
    def rotate_and_crop_icon(icon: QtGui.QIcon, angle: float, size: int = 64) -> QtGui.QIcon:
        # Get original pixmap
        original_pixmap = icon.pixmap(size, size)

        # Rotate the pixmap
        transform = QtGui.QTransform()
        transform.rotate(angle)
        rotated_pixmap = original_pixmap.transformed(
            transform, QtCore.Qt.SmoothTransformation)

        # Calculate crop rectangle to center crop back to original size
        rotated_size = rotated_pixmap.size()
        x = (rotated_size.width() - size) // 2
        y = (rotated_size.height() - size) // 2

        # Crop from center
        crop_rect = QtCore.QRect(x, y, size, size)
        cropped_pixmap = rotated_pixmap.copy(crop_rect)

        return QtGui.QIcon(cropped_pixmap)

from PyQt5 import QtCore, QtWidgets


class HorizontalTabBar(QtWidgets.QTabBar):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        font = self.font()
        font.setPointSize(10)  # Set desired font size
        self.setFont(font)

    def tabSizeHint(self, index):
        sz = super().tabSizeHint(index)
        return QtCore.QSize(sz.width() + 20, 90)  # fixed height

    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionTab()
        for idx in range(self.count()):
            self.initStyleOption(opt, idx)
            opt.shape = QtWidgets.QTabBar.RoundedNorth    # draw as if tabs were on top
            # draw the tab “shell”
            painter.drawControl(QtWidgets.QStyle.CE_TabBarTab, opt)
            # draw the label
            painter.drawControl(QtWidgets.QStyle.CE_TabBarTabLabel, opt)

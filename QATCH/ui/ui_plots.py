
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import GraphicsLayoutWidget

class Ui_Plots(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1091, 770)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setStyleSheet("")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.Layout_graphs = QtWidgets.QGridLayout()
        self.Layout_graphs.setObjectName("Layout_graphs")

        self.plt = GraphicsLayoutWidget(self.centralwidget)
        self.pltB = GraphicsLayoutWidget(self.centralwidget)

        self.plt.setAutoFillBackground(False)
        self.plt.setStyleSheet("border: 0px;")
        self.plt.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plt.setFrameShadow(QtWidgets.QFrame.Plain)
        self.plt.setLineWidth(0)
        self.plt.setObjectName("plt")

        self.pltB.setAutoFillBackground(False)
        self.pltB.setStyleSheet("border: 0px;")
        self.pltB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.pltB.setFrameShadow(QtWidgets.QFrame.Plain)
        self.pltB.setLineWidth(0)
        self.pltB.setObjectName("pltB")

        self.label = QtWidgets.QLabel()
        self.Layout_graphs.addWidget(self.label, 0, 0, 1, 1)
        def link1(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))
        self.label.linkActivated.connect(link1)
        self.label.setText('<a href="https://qatchtech.com/"> <font color=#000000 >Open-source Python application for displaying, processing and storing real-time data from QATCH Q-1 Device</font> </a>')

        self.Layout_graphs.addWidget(self.plt, 1, 0, 1, 1)
        self.Layout_graphs.addWidget(self.pltB, 2, 0, 1, 1)

        self.gridLayout.addLayout(self.Layout_graphs, 3, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "icons/qatch-icon.png")
        MainWindow.setWindowIcon(QtGui.QIcon(icon_path)) #.png
        MainWindow.setWindowTitle(_translate("MainWindow", "{} {} - Setup/Control GUI".format(Constants.app_title, Constants.app_version)))

'''
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Plots()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
'''

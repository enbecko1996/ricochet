import sys

import matplotlib
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (QWidget, QApplication, QDesktopWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtGui import QIcon
import ricochet.hyperparameter as hp
from threading import Thread
import ricochet.the_brain as brain


class HyperParameters(QWidget):
    def __init__(self, gui_play):
        super().__init__()
        self.gui_play = gui_play
        self.name_inpt_dict = {}
        self.hp = hp.hyperparams()
        self.initUI()

    def initUI(self):
        self.resize(800, 600)
        self.center()
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('test.png'))

        v_main = QVBoxLayout()
        for attr in dir(self.hp):
            if not callable(getattr(self.hp, attr)) and not attr.startswith("__"):
                hbox = QHBoxLayout()
                lab = QLabel(str(attr), self)
                inp = QLineEdit(str(getattr(self.hp, str(attr))), self)
                self.name_inpt_dict[str(attr)] = inp
                hbox.addWidget(lab)
                hbox.addWidget(inp)
                v_main.addLayout(hbox)

        self.start_btn = QPushButton("start", self)
        self.start_btn.clicked.connect(self.train_click)
        v_main.addWidget(self.start_btn)

        self.setLayout(v_main)
        self.show()

    def plot(self):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(self.epochs, self.steps, '*-')
        ax.set_xlabel('epoch')
        ax.set_ylabel('avg. steps')
        self.canvas.draw()

    def closeEvent(self, event):
        event.accept()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    @pyqtSlot()
    def train_click(self):
        self.gui_play.error.clear()
        if self.gui_play.brain_handler is None:
            self.gui_play.brain_handler = brain.Handler(self.gui_play.name.text(), 0)
            self.gui_play.brain_handler.make_status_gui()
        else:
            self.gui_play.brain_handler.make_status_gui()
        for attr in dir(self.hp):
            if not callable(getattr(self.hp, attr)) and not attr.startswith("__"):
                try:
                    setattr(self.hp, str(attr), int(self.name_inpt_dict[str(attr)]))
                except:
                    pass
                try:
                    setattr(self.hp, str(attr), float(self.name_inpt_dict[str(attr)]))
                except:
                    pass
        self.gui_play.brain_handler.set_hps(self.hp)
        thread = Thread(target=self.gui_play.start_train)
        thread.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = HyperParameters(hp.hyperparams())
    main.show()

    sys.exit(app.exec_())
import sys

import matplotlib
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (QWidget, QApplication, QDesktopWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtGui import QIcon
import numpy as np


class Status(QWidget):
    def __init__(self, brain_handler):
        super().__init__()
        self.brain_handler = brain_handler
        self.epochs = []
        self.game_steps = []
        self.train_steps = []
        self.poly_x = [0, 1]
        self.poly_y = [0, 0]
        self.initUI()

    def initUI(self):
        self.resize(800, 600)
        self.center()
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('test.png'))

        self.name = QLabel(self.brain_handler.name, self)

        self.pause_train = QPushButton('pause', self)
        self.pause_train.clicked.connect(self.pause_click)

        self.stop_train = QPushButton('stop', self)
        self.stop_train.clicked.connect(self.stop_click)

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        v_main = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.setSpacing(10)
        hbox.setAlignment(Qt.AlignLeft)
        hbox.addWidget(self.name)
        hbox.addWidget(self.pause_train)
        hbox.addWidget(self.stop_train)

        v_main.addLayout(hbox)

        v_main.addWidget(self.canvas)
        self.setLayout(v_main)
        self.plot()
        self.show()

    def plot(self):
        self.ax.clear()
        self.ax.plot(self.epochs, self.game_steps, '*-', color='green')
        self.ax.plot(self.epochs, self.train_steps, '*-', color='blue')
        self.ax.plot(self.poly_x, self.poly_y, '*-', color='red')
        self.ax.set_xlabel('epoch')
        self.ax.set_ylabel('avg. steps')
        self.canvas.draw()

    def add_data_point(self, epoch, game_steps, train_steps):
        self.epochs.append(epoch)
        self.game_steps.append(game_steps)
        self.train_steps.append(train_steps)
        m, b = np.polyfit(self.epochs, self.game_steps, 1)
        if len(self.epochs) == 1:
            self.poly_x[0] = np.min(self.epochs)
        self.poly_x[1] = np.max(self.epochs)
        self.poly_y[0] = b
        self.poly_y[1] = m * self.poly_x[1] + b

    def save_plot_to_disk(self, path):
        self.figure.savefig(path + ".png")

    @pyqtSlot()
    def pause_click(self):
        print("pause_click")
        self.brain_handler.pause_training()

    @pyqtSlot()
    def stop_click(self):
        print("stop_clicks")
        self.brain_handler.stop_training()

    def closeEvent(self, event):
        self.pause_click()
        event.accept()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Status(None)
    main.show()

    sys.exit(app.exec_())
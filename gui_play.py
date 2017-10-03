import sys

import numpy as np
from PyQt5.QtCore import QSize, Qt, pyqtSlot
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import (QWidget, QApplication, QDesktopWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QPushButton, QLabel, QGridLayout)
from qtpy.QtGui import QIcon

import game as game
import game_items_drawer as drawer
from game import Environment


class Picker(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.initUI()
        self.opened_at = [0, 0]

    def initUI(self):
        self.btn_size = 50
        self.resize(2 * self.btn_size, np.maximum(game.num_figures,
                                                  game.num_goals) * self.btn_size)
        self.content = np.zeros((game.num_figures, 5), dtype=np.int)
        pass

    def open(self, pos, edit, env):
        self.opened_at = edit
        self.move(pos[0], pos[1])
        longest = 0
        col_goals = 0
        j = 0
        for fig in range(game.num_figures):
            self.content[fig][:] = -1
            if fig not in env.figs_on_board:
                self.content[j][0] = fig
            goals = game.fig_dict[fig][1]
            cur_len = 0
            for i in range(len(goals)):
                self.content[j][i + 1] = -1
                idx = game.get_id_from_name(goals[i], game.goals)
                if idx not in env.goals_on_board:
                    cur_len += 1
                    self.content[j][cur_len] = idx
            if cur_len != 0:
                col_goals += 1
            if cur_len > longest:
                longest = cur_len
            j += 1
        self.resize(self.btn_size + longest * self.btn_size, game.num_figures * self.btn_size)
        self.show()

    def mousePressEvent(self, event):
        mouse = event.localPos()
        mouseX, mouseY = mouse.x(), mouse.y()

        env = self.parentWidget().env
        x, y = int(mouseX / self.btn_size), int(mouseY / self.btn_size)
        if x <= 0 and self.content[y][x] >= 0:
            env.add_single_figure(self.opened_at, self.content[y][x])
        elif self.content[y][x] >= 0:
            env.add_single_goal(self.opened_at, self.content[y][x])
        self.hide()
        self.parentWidget().update()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        env = self.parentWidget().env
        self.drawWidget(qp, env)
        qp.end()

    def drawWidget(self, qp, env):
        size = self.size()
        w = size.width()
        h = size.height()
        qp.setPen(QColor(160, 160, 160))
        qp.setBrush(QColor(255, 255, 255))
        qp.drawRect(0, 0, w - 1, h - 1)
        i = 0
        for fig in range(game.num_figures):
            if fig not in env.figs_on_board:
                col = game.fig_dict[fig][3]
                qp.setPen(QColor(col[0], col[1], col[2]))
                qp.setBrush(QColor(col[0], col[1], col[2]))
                drawer.circle(qp, 0, self.btn_size * i, self.btn_size)
            goals = game.fig_dict[fig][1]
            cur_len = 0
            i += 1
            for j in range(len(goals)):
                idx = game.get_id_from_name(goals[j], game.goals)
                if idx not in env.goals_on_board:
                    cur_len += 1
                    col = game.goal_dict[idx][2]
                    qp.setPen(QColor(col[0] - 100, col[1] - 100, col[2] - 100))
                    qp.setBrush(QColor(col[0], col[1], col[2]))
                    game.goal_dict[idx][3](qp, cur_len * self.btn_size, self.btn_size * (i - 1), self.btn_size)


class Board(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.env = Environment(16)
        self.clear()
        self.initUI()

    def initUI(self):
        self.setMinimumSize(300, 300)
        self.setMaximumSize(self.parentWidget().size().height(), self.parentWidget().size().height())
        self.setSizeIncrement(1, 1)

        self.pick = Picker(self)
        self.pick.hide()

    def clear(self):
        self.env.reset(figure_style='none', board_style=[[2, 0], [3, 1], [6, 0], [0, 1]])
        """self.env.set_quadrant(1, "quadrants/pre_0_1.npy")
        self.env.set_quadrant(2, "quadrants/pre_7_1.npy")
        self.env.set_quadrant(3, "quadrants/pre_3_0.npy")
        self.env.set_quadrant(4, "quadrants/pre_2_1.npy")"""

    def save(self, file_name):
        self.env.save_current_as_quadrant(file_name)

    def sizeHint(self):
        par_size = self.parentWidget().size()
        return QSize(par_size.height() - 20, par_size.height() - 20)

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()

    def mousePressEvent(self, event):
        mouse = event.localPos()
        mouseX, mouseY = mouse.x(), mouse.y()
        size = self.size()
        h = w = size.width()
        grid = self.env.grid_size
        per_box = (w - 1) / grid

        thick = np.maximum(per_box / 12, 2)

        x, y = int(mouseX / per_box), int(mouseY / per_box)
        diffX, diffY = mouseX % per_box, mouseY % per_box
        if diffX < per_box / 2 and diffX < thick:
            self.env.change_wall(([x, y], [x - 1, y]))
        if diffX > per_box / 2 and per_box - diffX < thick:
            self.env.change_wall(([x, y], [x + 1, y]))
        if diffY < per_box / 2 and diffY < thick:
            self.env.change_wall(([x, y], [x, y - 1]))
        if diffY > per_box / 2 and per_box - diffY < thick:
            self.env.change_wall(([x, y], [x, y + 1]))
        self.pick.hide()
        if thick < diffX < per_box - thick and thick < diffY < per_box - thick:
            self.pick.open((mouseX, mouseY), [x, y], self.env)
        self.update()

    def drawWidget(self, qp):
        size = self.size()
        h = w = size.width()

        grid = self.env.grid_size
        per_box = (w - 1) / grid

        qp.setPen(QColor(0, 0, 0))
        for x in range(grid):
            for y in range(grid):
                qp.drawRect(x * per_box, y * per_box, per_box, per_box)
        for x in range(grid):
            for y in range(grid):
                drawer.drawWall(qp, (x, y), self.env.the_state[x][y][:4], per_box)
                drawer.draw_fig(qp, (x, y), game, self.env.the_state[x][y][4:4 + game.num_figures], per_box)
                drawer.draw_goal(qp, (x, y), game, self.env.the_state[x][y][4 + game.num_figures], per_box)

# --------------------------MAIN-----------------------------------------
import the_brain as brain
from pathlib import Path
import tensorflow as tf
import gui_hps as gui_hps


class RicochetGui(QWidget):
    def __init__(self):
        super().__init__()
        self.brain_handler = None
        self.graph = None
        self.initUI()

    def initUI(self):
        self.resize(1400, 1000)
        self.center()
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('test.png'))

        self.board = Board(self)
        self.board.resize(self.board.sizeHint())
        self.board.move(10, 10)

        self.file_name = QLineEdit(self)
        self.file_name.resize(20, 50)

        self.save = QPushButton('save', self)
        self.save.clicked.connect(self.save_click)
        self.save.resize(20, 50)

        self.clear = QPushButton('clear', self)
        self.clear.clicked.connect(self.clear_click)
        self.clear.resize(20, 50)

        self.name_lab = QLabel("name: ", self)
        self.name = QLineEdit(self)

        self.board_lab = QLabel("style: ", self)
        self.board_style = QLineEdit(self)

        self.load_lab = QLabel("load: ", self)
        self.load_name = QLineEdit(self)

        self.newest = QPushButton("newest", self)
        self.newest.clicked.connect(self.load_newest_click)
        self.version_lab = QLabel("version: ", self)
        self.version = QLineEdit(self)
        self.version_push = QPushButton("load", self)
        self.version_push.clicked.connect(self.load_vers_click)

        h_vers = QHBoxLayout()
        h_vers.addWidget(self.version_lab)
        h_vers.addWidget(self.version)
        h_vers.addWidget(self.version_push)

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.load_lab, 1, 0)
        grid.addWidget(self.load_name, 1, 1)

        grid.addWidget(self.newest, 2, 0)
        grid.addLayout(h_vers, 2, 1)

        grid.addWidget(self.name_lab, 3, 0)
        grid.addWidget(self.name, 3, 1)

        grid.addWidget(self.board_lab, 4, 0)
        grid.addWidget(self.board_style, 4, 1)

        self.train = QPushButton('train', self)
        self.train.clicked.connect(self.train_click)

        self.error = QLabel('', self)

        vbox_2 = QVBoxLayout()
        vbox_2.setAlignment(Qt.AlignTop)
        vbox_2.addLayout(grid)
        vbox_2.addWidget(self.train)
        vbox_2.addWidget(self.error)

        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignTop)
        vbox.addWidget(self.file_name)
        vbox.addWidget(self.save)
        vbox.addWidget(self.clear)

        vbox_3 = QVBoxLayout()

        v_main = QVBoxLayout()
        v_main.setAlignment(Qt.AlignTop)
        v_main.addLayout(vbox_2)
        v_main.addStretch(1)
        v_main.addLayout(vbox)
        v_main.addStretch(1)
        v_main.addLayout(vbox_3)

        hbox = QHBoxLayout()
        hbox.setAlignment(Qt.AlignLeft)
        hbox.addWidget(self.board)
        hbox.addLayout(v_main)
        self.setLayout(hbox)

        self.show()

    @pyqtSlot()
    def save_click(self):
        self.error.clear()
        file_name = self.file_name.text()
        self.board.save(file_name)

    @pyqtSlot()
    def clear_click(self):
        self.error.clear()
        self.board.clear()
        self.board.update()

    @pyqtSlot()
    def train_click(self):
        self.hp_tweak = gui_hps.HyperParameters(self)
        self.hp_tweak.show()

    def start_train(self):
        style = self.board_style.text()
        if style == 'same':
            style = [[2, 0], [3, 1], [6, 0], [0, 1]]
        elif style == 'random':
            pass
        else:
            self.error.setText("specified style is invalid")
            return
        if self.graph is not None:
            with self.graph.as_default():
                self.brain_handler.initialize(style)
                self.brain_handler.start_training(style)
        else:
            self.brain_handler.initialize(style)
            self.brain_handler.start_training(style)

    @pyqtSlot()
    def load_newest_click(self):
        self.error.clear()
        name = self.load_name.text()
        folder = Path("models/" + name)
        if folder.is_dir():
            folder = str(folder)
            my_file = Path(folder + "/0/worker.h5")
            my_file_2 = Path(folder + "/0/feeder.h5")
            i = 0
            while my_file.is_file() or my_file_2.is_file():
                i += 1
                print(i)
                my_file = Path(folder + "/" + str(i) + "/worker.h5")
                my_file_2 = Path(folder + "/" + str(i) + "/feeder.h5")
            if i > 0:
                if i > 1:
                    my_file = Path(folder + "/" + str(i) + "/worker.h5")
                    my_file_2 = Path(folder + "/" + str(i) + "/feeder.h5")
                else:
                    my_file = Path(folder + "/0/worker.h5")
                    my_file_2 = Path(folder + "/0/feeder.h5")
                self.name.setText(name)
                self.brain_handler = brain.Handler(name, i, worker=my_file, feeder=my_file_2)
                self.graph = tf.get_default_graph()
                return
        self.error.setText("failed to load: " + name)

    @pyqtSlot()
    def load_vers_click(self):
        self.error.clear()
        name = self.load_name.text()
        folder = Path("models/" + name)
        vers = str(self.version.text())
        if folder.is_dir():
            folder = str(folder)
            my_file = Path(folder + "/" + vers + "/worker.h5")
            my_file_2 = Path(folder + "/" + vers + "/feeder.h5")
            if my_file.is_file() and my_file_2.is_file():
                self.brain_handler = brain.Handler(name, vers, worker=my_file, feeder=my_file_2)
                self.graph = tf.get_default_graph()
                self.name.setText(name)
                return
        self.error.setText("failed to load: " + name + " with vers.: " + vers)

    def closeEvent(self, event):
        event.accpt()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    import os

    print(os.getcwd())
    app = QApplication(sys.argv)
    gui = RicochetGui()
    sys.exit(app.exec_())

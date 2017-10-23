import sys

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (QWidget, QApplication, QDesktopWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QComboBox, QGridLayout)
from qtpy.QtGui import QIcon

import hyperparameter as hp
import inspect


class GUI_HyperParameters(QWidget):
    def __init__(self, gui_play, cur_hps=None):
        super().__init__()
        self.gui_play = gui_play
        self.name_inpt_dict = {}
        self.structured_hyperparams = {}
        self.substructered_items = []
        self.reference_hps = hp.AgentsHyperparameters()
        if cur_hps is None:
            self.hp = hp.AgentsHyperparameters()
        else:
            self.hp = cur_hps
        self.initUI()

    def initUI(self):
        self.resize(800, 600)
        self.center()
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('test.png'))

        self.v_main = QVBoxLayout()
        idx = 0
        for attr in dir(self.hp):
            attr_name = str(attr)
            if not callable(getattr(self.hp, attr)) and not attr.startswith("__"):
                idx += 1
                hbox = QHBoxLayout()
                lab = QLabel(attr_name, self)
                hbox.addWidget(lab)
                the_attrib = getattr(self.hp, attr_name)
                if isinstance(the_attrib, dict):
                    inp = QComboBox()
                    inp.addItems(the_attrib.keys())
                    self.structured_hyperparams[attr_name] = (idx, the_attrib)
                    inp.activated.connect(self.activated_combobox)
                    hbox.addWidget(inp)
                elif isinstance(the_attrib, list):
                    list_key = the_attrib[0]
                    reference = getattr(self.reference_hps, attr_name)
                    if isinstance(reference, dict):
                        if list_key in reference:
                            reference[list_key] = the_attrib[1]
                            inp = QComboBox()
                            inp.addItems(reference.keys())
                            self.structured_hyperparams[attr_name] = (idx, reference)
                            inp.activated.connect(self.activated_combobox)
                            hbox.addWidget(inp)
                        else:
                            inp = None
                    else:
                        inp = None
                else:
                    inp = QLineEdit(str(the_attrib), self)
                    hbox.addWidget(inp)
                if inp is not None:
                    self.name_inpt_dict[attr_name] = inp
                    self.v_main.addLayout(hbox)
        self.add_sub_inputs()

        self.start_btn = QPushButton("start", self)
        self.start_btn.clicked.connect(self.train_click)
        self.v_main.addWidget(self.start_btn)

        self.setLayout(self.v_main)
        self.show()

    def add_sub_inputs(self):
        items_per_row = 3
        for sub_struct in self.substructered_items:
            for i in range(self.v_main.count()):
                layout_item = self.v_main.itemAt(i)
                if isinstance(layout_item, QHBoxLayout) and layout_item.layout() == sub_struct:
                    clear_layout(layout_item)
                    self.v_main.removeItem(layout_item)
        self.substructered_items.clear()
        for struct_hyper_key in self.structured_hyperparams.keys():
            combo_box = self.name_inpt_dict[struct_hyper_key]
            new_items_dict = self.structured_hyperparams[struct_hyper_key][1][combo_box.currentText()]
            new_item_keys = list(new_items_dict.keys())
            for i in range(0, len(new_item_keys), items_per_row):
                h_outer = QHBoxLayout()
                for j in range(items_per_row):
                    if i + j < len(new_item_keys):
                        cur_key = new_item_keys[i + j]
                        hbox = QHBoxLayout()
                        hbox.addWidget(QLabel(cur_key, self))
                        inp = QLineEdit(str(new_items_dict[cur_key]), self)
                        hbox.addWidget(inp)
                        self.name_inpt_dict[cur_key] = inp
                        h_outer.addLayout(hbox)
                self.substructered_items.append(h_outer)
                self.v_main.insertLayout(self.structured_hyperparams[struct_hyper_key][0], h_outer)

    def activated_combobox(self):
        print("connected")
        self.add_sub_inputs()

    def closeEvent(self, event):
        event.accept()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    @pyqtSlot()
    def train_click(self):
        for attr in dir(self.hp):
            if not callable(getattr(self.hp, attr)) and not attr.startswith("__"):
                attr_nam = str(attr)
                if isinstance(self.name_inpt_dict[attr_nam], QComboBox):
                    combo_box = self.name_inpt_dict[attr_nam]
                    selected_item = combo_box.currentText()
                    items_dict = self.structured_hyperparams[attr_nam][1][selected_item]
                    for cur_key in items_dict.keys():
                        value = get_int_or_else_float_from_str(self.name_inpt_dict[cur_key].text())
                        if value is not None:
                            items_dict[cur_key] = value
                    print(items_dict)
                    setattr(self.hp, attr_nam, (selected_item, items_dict))
                    print(getattr(self.hp, attr_nam))
                else:
                    value = get_int_or_else_float_from_str(self.name_inpt_dict[attr_nam].text())
                    if value is not None:
                        setattr(self.hp, attr_nam, value)
        self.gui_play.new_handler_and_start(self.hp)
        self.hide()


def get_int_or_else_float_from_str(the_str):
    try:
        value = int(the_str)
    except:
        try:
            value = float(the_str)
        except:
            value = None
    return value


def clear_layout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
            child.widget().deleteLater()
        elif child.layout() is not None:
            clear_layout(child.layout())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = GUI_HyperParameters(hp.AgentsHyperparameters())
    main.show()

    sys.exit(app.exec_())
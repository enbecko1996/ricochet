from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPolygonF, QColor
import numpy as np
import helper as hlp


def circle(qp, x, y, size, offset=6):
    h_off = offset / 2
    half = size / 2
    qp.drawEllipse(QPointF(x + half, y + half), half - h_off, half - h_off)


def square(qp, x, y, size, offset=6):
    h_off = offset / 2
    qp.drawRect(x + h_off, y + h_off, size - offset, size - offset)


def triangle(qp, x, y, size, offset=6):
    h_off = offset / 2
    tria = QPolygonF([QPointF(x + h_off, y + size - h_off), QPointF(x + size - h_off, y + size - h_off), QPointF(x + size / 2, y + h_off)])
    qp.drawPolygon(tria)


def hexagon(qp, x, y, size, offset=6):
    h_off = offset / 2
    len_side = (size - offset) / 2
    height_step = 0.5 * len_side
    hexa = QPolygonF([QPointF(x + size / 2, y + size - h_off), QPointF(x + size - h_off, y + height_step + len_side + h_off),
                      QPointF(x + size - h_off, y + height_step + h_off), QPointF(x + size / 2, y + h_off),
                      QPointF(x + h_off, y + height_step + h_off), QPointF(x + h_off, y + height_step + len_side + h_off)])
    qp.drawPolygon(hexa)


def drawWall(qp, pos, walls, per_box):
    thick = np.maximum(per_box / 12, 2)
    qp.setPen(QColor(100, 100, 100))
    qp.setBrush(QColor(100, 100, 100))
    if walls[0] == 1:
        qp.drawRect((pos[0] + 1) * per_box - thick, pos[1] * per_box, thick, per_box)
    if walls[1] == 1:
        qp.drawRect(pos[0] * per_box, (pos[1] + 1) * per_box - thick, per_box, thick)
    if walls[2] == 1:
        qp.drawRect(pos[0] * per_box, pos[1] * per_box, thick, per_box)
    if walls[3] == 1:
        qp.drawRect(pos[0] * per_box, pos[1] * per_box, per_box, thick)


def draw_fig(qp, pos, game, one_hot, per_box):
    idx = hlp.one_hot_to_id(one_hot)
    if idx >= 0:
        col = game.fig_dict[idx][3]
        qp.setPen(QColor(col[0], col[1], col[2]))
        qp.setBrush(QColor(col[0], col[1], col[2]))
        circle(qp, pos[0] * per_box, pos[1] * per_box, per_box, offset=20)


def draw_goal(qp, pos, game, idx, per_box):
    if idx >= 1:
        col = game.goal_dict[idx][2]
        qp.setPen(QColor(col[0] - 100, col[1] - 100, col[2] - 100))
        qp.setBrush(QColor(col[0], col[1], col[2]))
        game.goal_dict[idx][3](qp, pos[0] * per_box, pos[1] * per_box, per_box, offset=20)


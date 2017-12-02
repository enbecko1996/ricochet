import random as rand

import numpy as np

import helper as hlp
import hyperparameter as hp
from gui import game_items_drawer as g

same_board_style = [[0, 0], [1, 1], [2, 0], [3, 1]]

grid_size = 12
standard_board_style = 'medium'

figures = ['red', 'green', 'blue', 'yellow', 'grey']
num_figures = len(figures)
num_figures = 4
goals = ['placeholder']
red_goals = ['red_hexagon', 'red_square', 'red_triangle', 'red_circle']
green_goals = ['green_hexagon', 'green_square', 'green_triangle', 'green_circle']
blue_goals = ['blue_hexagon', 'blue_square', 'blue_triangle', 'blue_circle']
yellow_goals = ['yellow_hexagon', 'yellow_square', 'yellow_triangle', 'yellow_circle']
grey_goals = ['all']
goals.extend(red_goals)
goals.extend(green_goals)
goals.extend(blue_goals)
goals.extend(yellow_goals)
goals.extend(grey_goals)
num_goals = len(goals)

red = (255, 60, 60)
green = (60, 255, 60)
blue = (60, 60, 255)
yellow = (255, 255, 60)
grey = (160, 160, 160)

red_g = (255, 140, 140)
green_g = (140, 255, 140)
blue_g = (140, 140, 255)
yellow_g = (255, 255, 140)
grey_g = (0, 0, 0)

goal_dict = {1: ('red_hexagon', 'R*', red_g, g.hexagon), 2: ('red_square', 'R#', red_g, g.square),
             3: ('red_triangle', 'R^', red_g, g.triangle), 4: ('red_circle', 'R°', red_g, g.circle),
             5: ('green_hexagon', 'G*', green_g, g.hexagon), 6: ('green_square', 'G#', green_g, g.square),
             7: ('green_triangle', 'G^', green_g, g.triangle), 8: ('green_circle', 'G°', green_g, g.circle),
             9: ('blue_hexagon', 'B*', blue_g, g.hexagon), 10: ('blue_square', 'B#', blue_g, g.square),
             11: ('blue_triangle', 'B^', blue_g, g.triangle), 12: ('blue_circle', 'B°', blue_g, g.circle),
             13: ('yellow_hexagon', 'Y*', yellow_g, g.hexagon), 14: ('yellow_square', 'Y#', yellow_g, g.square),
             15: ('yellow_triangle', 'Y^', yellow_g, g.triangle), 16: ('yellow_circle', 'Y°', yellow_g, g.circle),
             17: ('all', 'AA', grey_g, g.square)}
dir_dict = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
fig_dict = {0: ('red', red_goals, 'R', red), 1: ('green', green_goals, 'G', green),
            2: ('blue', blue_goals, 'B', blue), 3: ('yellow', yellow_goals, 'Y', yellow),
            4: ('grey', grey_goals, 'g', grey)}

all_quadrants = np.load("quadrants/pre_all.npy")
smallrgb = np.load("small/r-g-b.npy")
mediumrgby = np.load("small/medium.npy")

in_wall_reward = -2.0
step_reward = -1.0
goal_reached_reward = 80.0


class Action:
    def __init__(self, fig, direc):
        self.figure = fig
        self.direction = direc


actions = [Action(f, d) for f in range(num_figures) for d in range(4)]
action_size = len(actions)


class Environment:
    def __init__(self, g_size=grid_size, board_style=standard_board_style):
        self.visible_state = np.zeros((g_size, g_size, 4 + num_figures + 1), dtype=np.int)
        self.game_state = np.zeros((g_size, g_size, 4 + num_figures + num_figures), dtype=np.int)
        self.grid_size = g_size
        self.num_figures = num_figures
        self.num_goals = num_goals
        self.observation_space = self.visible_state.shape
        self.flattened_input_size = self.grid_size ** 2 * (4 + num_figures + num_figures)
        self.action_size = action_size
        self.dims = 4 + num_figures + num_figures
        self.cur_goal_name = None
        self.cur_goal = None
        self.cur_goal_pos = None
        self.cur_goal_color = None
        self.figs_on_board = []
        self.goals_on_board = []
        self.apply_board_style(board_style)

    def reset(self, flattened=True, figure_style='test_case', board_style=standard_board_style, goal_style='random'):
        # self.visible_state = np.zeros((self.grid_size, self.grid_size, 4 + num_figures + 1), dtype=np.int)
        # self.game_state = np.zeros((self.grid_size, self.grid_size, 4 + num_figures + num_figures), dtype=np.int)
        # self.figs_on_board.clear()
        # self.goals_on_board.clear()
        # self.apply_board_style(board_style)
        self.set_figures(figure_style)
        if isinstance(goal_style, str):
            if goal_style == 'random':
                if len(self.goals_on_board) > 0:
                    self.set_current_goal(goals[self.goals_on_board[rand.randrange(0, len(self.goals_on_board))]])
            elif goal_style == 'test_case':
                self.set_current_goal('blue_triangle')
        else:
            if len(self.goals_on_board) > 0:
                self.set_current_goal('green_square')
        # self.render()

        if flattened:
            return self.get_flattened_reduced_state()
        else:
            return np.array(self.game_state)

    def set_state_from_other_env(self, other_env):
        self.visible_state = np.array(other_env.visible_state)
        self.game_state = np.array(other_env.game_state)
        self.set_current_goal(other_env.cur_goal)

    def apply_board_style(self, board_style):
        if isinstance(board_style, str):
            if board_style == 'random':
                self.set_random_board()
            if board_style == 'small':
                self.set_full(smallrgb)
            elif board_style == 'border':
                self.add_surrounding()
            elif board_style == 'medium':
                self.set_full(mediumrgby)
        elif board_style is not None:
            for i in range(4):
                self.set_quadrant(i + 1, all_quadrants[board_style[i][0]][board_style[i][1]])
        self.cleanup()

    def set_random_board(self):
        taken = list(range(len(all_quadrants)))
        for i in range(4):
            idx = rand.randrange(0, len(taken))
            next_quad = taken[idx]
            del taken[idx]
            style = rand.randrange(0, 2) if next_quad < 4 else 1
            self.set_quadrant(i + 1, all_quadrants[next_quad][style])

    def save_current_game(self, filename):
        np.save(filename, self.visible_state)

    def save_current_as_quadrant(self, filename):
        lis = [self.visible_state, hlp.get_rotated_quadrant(2, self.visible_state, self.grid_size / 2),
               hlp.get_rotated_quadrant(3, self.visible_state, self.grid_size / 2),
               hlp.get_rotated_quadrant(4, self.visible_state, self.grid_size / 2)]
        np.save(filename, np.array(lis))

    def load_game(self, filename):
        self.visible_state = hlp.load_state(filename)

    def cleanup(self):
        half_g = int(self.grid_size / 2)
        for quad in range(1, 5):
            if quad == 1:
                for y in range(0, half_g):
                    if self.visible_state[half_g][y][2] == 1:
                        self.visible_state[half_g - 1][y][0] = 1
                        self.game_state[half_g - 1][y][0] = 1
                for x in range(half_g, self.grid_size):
                    if self.visible_state[x][half_g - 1][1] == 1:
                        self.visible_state[x][half_g][3] = 1
                        self.game_state[x][half_g][3] = 1
            elif quad == 2:
                for y in range(half_g, self.grid_size):
                    if self.visible_state[half_g][y][2] == 1:
                        self.visible_state[half_g - 1][y][0] = 1
                        self.game_state[half_g - 1][y][0] = 1
                for x in range(half_g, self.grid_size):
                    if self.visible_state[x][half_g][3] == 1:
                        self.visible_state[x][half_g - 1][1] = 1
                        self.game_state[x][half_g - 1][1] = 1
            elif quad == 3:
                for y in range(half_g, self.grid_size):
                    if self.visible_state[half_g - 1][y][0] == 1:
                        self.visible_state[half_g][y][2] = 1
                        self.game_state[half_g][y][2] = 1
                for x in range(0, half_g):
                    if self.visible_state[x][half_g][3] == 1:
                        self.visible_state[x][half_g - 1][1] = 1
                        self.game_state[x][half_g - 1][1] = 1
            elif quad == 4:
                for y in range(0, half_g):
                    if self.visible_state[half_g - 1][y][0] == 1:
                        self.visible_state[half_g][y][2] = 1
                        self.game_state[half_g][y][2] = 1
                for x in range(0, half_g):
                    if self.visible_state[x][half_g - 1][1] == 1:
                        self.visible_state[x][half_g][3] = 1
                        self.game_state[x][half_g][3] = 1

    def set_full(self, full):
        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                self.visible_state[x][y] = full[x][y]
                self.game_state[x][y][:4 + num_figures] = full[x][y][:4 + num_figures]
                if self.visible_state[x][y][4 + num_figures] > 0:
                    self.goals_on_board.append(self.visible_state[x][y][4 + num_figures])

    def set_quadrant(self, quad, quad_state):
        if 1 <= quad <= 4:
            if isinstance(quad_state, str):
                quad_state = hlp.load_state(quad_state)
            half_g = int(self.grid_size / 2)
            if quad == 1:
                for x in range(half_g, self.grid_size):
                    for y in range(0, half_g):
                        self.visible_state[x][y] = quad_state[0][x - half_g][y]
                        self.game_state[x][y][:4 + num_figures] = quad_state[0][x - half_g][y][:4 + num_figures]
                        if self.visible_state[x][y][4 + num_figures] > 0:
                            self.goals_on_board.append(self.visible_state[x][y][4 + num_figures])
            elif quad == 2:
                for x in range(half_g, self.grid_size):
                    for y in range(half_g, self.grid_size):
                        self.visible_state[x][y] = quad_state[1][x - half_g][y - half_g]
                        self.game_state[x][y][:4 + num_figures] = quad_state[1][x - half_g][y - half_g][
                                                                  :4 + num_figures]
                        if self.visible_state[x][y][4 + num_figures] > 0:
                            self.goals_on_board.append(self.visible_state[x][y][4 + num_figures])
            elif quad == 3:
                for x in range(0, half_g):
                    for y in range(half_g, self.grid_size):
                        self.visible_state[x][y] = quad_state[2][x - half_g][y - half_g]
                        self.game_state[x][y][:4 + num_figures] = quad_state[2][x - half_g][y - half_g][
                                                                  :4 + num_figures]
                        if self.visible_state[x][y][4 + num_figures] > 0:
                            self.goals_on_board.append(self.visible_state[x][y][4 + num_figures])
            elif quad == 4:
                for x in range(0, half_g):
                    for y in range(0, half_g):
                        self.visible_state[x][y] = quad_state[3][x - half_g][y]
                        self.game_state[x][y][:4 + num_figures] = quad_state[3][x - half_g][y][:4 + num_figures]
                        if self.visible_state[x][y][4 + num_figures] > 0:
                            self.goals_on_board.append(self.visible_state[x][y][4 + num_figures])

    def set_figures(self, style):
        if style == 'as_is':
            if len(self.figs_on_board) > 0:
                if rand.random() <= 0.0625:
                    self.set_figures('random')
            else:
                self.set_figures('random')
        elif style == 'none':
            pass
        elif style == 'test_case':
            self.remove_all_figures()
            self.add_single_figure([7, 0], 'red')
            self.add_single_figure([2, 3], 'green')
            self.add_single_figure([5, 5], 'blue')
        elif style == 'random':
            self.remove_all_figures()
            poss = []
            for i in range(num_figures):
                x, y = rand.randrange(0, self.grid_size), rand.randrange(0, self.grid_size)
                if (x, y) not in poss:
                    poss.append((x, y))
                    self.add_single_figure([x, y], i)

    def remove_all_figures(self, state=None):
        if state is None:
            state = self.game_state
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state[x][y][4:4 + self.num_figures] = 0
        self.figs_on_board.clear()

    def get_current_goal_pos(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                gol = self.game_state[x][y][4 + num_figures: 4 + 2 * num_figures]
                gol = hlp.one_hot_to_id(gol)
                if gol > -1:
                    return x, y

    def set_current_goal(self, gol):
        if self.cur_goal_pos is not None:
            self.game_state[self.cur_goal_pos[0]][self.cur_goal_pos[1]][4 + num_figures:4 + 2 * num_figures] = 0
        if isinstance(gol, str):
            self.cur_goal_name = gol
            self.cur_goal = goals.index(self.cur_goal_name)
        elif isinstance(gol, int):
            self.cur_goal_name = goal_dict[gol][0]
            self.cur_goal = gol
        self.cur_goal_pos = self.get_pos_on_board(self.cur_goal_name, state=self.visible_state)
        self.cur_goal_color = get_goal_color(self.cur_goal_name)
        if self.cur_goal_name != 'all':
            self.game_state[self.cur_goal_pos[0]][self.cur_goal_pos[1]][4 + num_figures:4 + 2 * num_figures] = \
                hlp.as_one_hot(self.cur_goal_color, num_figures)
        else:
            self.game_state[self.cur_goal_pos[0]][self.cur_goal_pos[1]][4 + num_figures:4 + 2 * num_figures] = 1

    def get_flattened_reduced_state(self, state=None):
        if state is None:
            state = self.game_state
        return np.array(np.reshape(state, (self.flattened_input_size,)))

    def get_reduced_state(self, state=None):
        if state is None:
            state = self.game_state
        return np.array(state)

    def clear_pos(self, x, y):
        fig = hlp.one_hot_to_id(self.game_state[x][y][4:4 + num_figures])
        gol = self.visible_state[x][y][4 + num_figures]
        if fig in self.figs_on_board:
            self.figs_on_board.remove(fig)
        if gol in self.goals_on_board:
            self.goals_on_board.remove(gol)
        self.visible_state[x][y][4:] = 0
        self.game_state[x][y][4:] = 0

    def get_goal_at(self, x, y):
        idx = self.visible_state[x][y][4 + num_figures]
        if idx > 0:
            return goal_dict[idx][0]
        return None

    def get_pos_on_board(self, test, state=None):
        if test is not None:
            if isinstance(test, str):
                if test in figures:
                    if state is None:
                        state = self.game_state
                    test = hlp.as_one_hot(figures.index(test), num_figures)
                    for x in range(self.grid_size):
                        for y in range(self.grid_size):
                            if np.array_equal(state[x][y][4:4 + num_figures], test):
                                return x, y
                if test in goals:
                    if state is None:
                        state = self.visible_state
                    test = goals.index(test)
                    for x in range(self.grid_size):
                        for y in range(self.grid_size):
                            if state[x][y][4 + num_figures] == test:
                                return x, y

    def get_valid_actions(self, state=None, flattened=False):
        if flattened and state is not None:
            state = state.reshape(self.grid_size, self.grid_size, 4 + num_figures + num_figures)
        if state is None:
            state = self.game_state
        valid = []
        for act in actions:
            fig_pos = self.get_pos_on_board(fig_dict[act.figure][0], state=state)
            if fig_pos is not None:
                if self.is_valid_action(state, fig_pos, act.direction):
                    valid.append(actions.index(act))
        return valid

    def is_valid_action(self, state, pos, direc):
        new_x, new_y = self.iterate_step(state, pos[0], pos[1], direc, steps=1)
        if new_x != pos[0] or new_y != pos[1]:
            return True
        return False

    def step(self, a_int, flattened=True, state=None, hyperparams=None):
        a = actions[a_int]
        apply = self.apply_action_and_get_reward(a.figure, a.direction, state=state, hyperparams=hyperparams)
        if flattened:
            return self.get_flattened_reduced_state(state=state), apply[0], apply[1], None
        else:
            return self.get_reduced_state(state=state), apply[0], apply[1], None

    # fig as id and not one-hot, direc as well
    def apply_action_and_get_reward(self, fig, direc, state=None, hyperparams=None):
        in_wall = in_wall_reward
        goal = goal_reached_reward
        step = step_reward
        if hyperparams is not None:
            if hasattr(hyperparams, "REWARD_IN_WALL"):
                in_wall = hyperparams.REWARD_IN_WALL
                goal = hyperparams.REWARD_GOAL_REACHED
                step = hyperparams.REWARD_STEP
        if state is None:
            state = self.game_state
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if state[x][y][4 + fig] == 1:
                    # print(fig_dict[fig][0], dir_dict[direc])
                    new_x, new_y = self.iterate_step(state, x, y, direc)
                    if new_x == x and new_y == y:
                        return in_wall, False
                    else:
                        # self.the_state[x][y][4 + fig] = 0
                        # self.the_state[new_x][new_y][4 + fig] = 1
                        state[x][y][4 + fig] = 0
                        state[new_x][new_y][4 + fig] = 1
                        if np.array_equal(state[new_x][new_y][4 + self.num_figures:4 + 2 * self.num_figures],
                                          state[new_x][new_y][4:4 + self.num_figures]):
                            return goal, True
                        else:
                            return step, False
        return in_wall_reward, True

    def iterate_step(self, state, x, y, direc, steps=None):
        if steps is not None:
            steps -= 1
        if state[x][y][direc] == 0:
            if direc == 0:
                if self.valid_x_y(x + 1, y) and max(state[x + 1][y][4:4 + self.num_figures]) == 0:
                    if steps is None or steps >= 0:
                        return self.iterate_step(state, x + 1, y, direc, steps=steps)
            if direc == 1:
                if self.valid_x_y(x, y + 1) and max(state[x][y + 1][4:4 + self.num_figures]) == 0:
                    if steps is None or steps >= 0:
                        return self.iterate_step(state, x, y + 1, direc, steps=steps)
            if direc == 2:
                if self.valid_x_y(x - 1, y) and max(state[x - 1][y][4:4 + self.num_figures]) == 0:
                    if steps is None or steps >= 0:
                        return self.iterate_step(state, x - 1, y, direc, steps=steps)
            if direc == 3:
                if self.valid_x_y(x, y - 1) and max(state[x][y - 1][4:4 + self.num_figures]) == 0:
                    if steps is None or steps >= 0:
                        return self.iterate_step(state, x, y - 1, direc, steps=steps)
        return x, y

    def add_single_figure(self, pos, fig):
        if isinstance(fig, str):
            fig = get_id_from_name(fig, figures)
        if fig is not None and fig not in self.figs_on_board and fig < self.num_figures:
            if -1 < pos[0] < self.grid_size and -1 < pos[1] < self.grid_size:
                # self.visible_state[pos[0]][pos[1]][4:4 + self.num_figures] = hlp.as_one_hot(fig, self.num_figures)
                self.game_state[pos[0]][pos[1]][4:4 + self.num_figures] = hlp.as_one_hot(fig, self.num_figures)
                self.figs_on_board.append(fig)

    def add_figures(self, pos_figures_tuples):
        for pos_fig in pos_figures_tuples:
            self.add_single_figure(pos_fig[0], pos_fig[1])

    def add_single_goal(self, pos, gol):
        if isinstance(gol, str):
            gol = get_id_from_name(gol, goals)
        if gol is not None and gol not in self.goals_on_board and gol < self.num_goals:
            if self.valid_x_y(pos[0], pos[1]):
                self.visible_state[pos[0]][pos[1]][4 + self.num_figures] = gol
                self.goals_on_board.append(gol)

    def add_goals(self, pos_goal_tuples):
        for pos_goal in pos_goal_tuples:
            self.add_single_goal(pos_goal[0], pos_goal[1])

    def add_surrounding(self):
        for x in range(self.grid_size):
            self.add_single_wall(([x, 0], [x, -1]))
            self.add_single_wall(([x, self.grid_size - 1], [x, self.grid_size]))
        for y in range(self.grid_size):
            self.add_single_wall(([0, y], [-1, y]))
            self.add_single_wall(([self.grid_size - 1, y], [self.grid_size, y]))

    def add_quadrant_1(self):
        for x in range(self.grid_size):
            self.add_single_wall(([x, 0], [x, -1]))
        for y in range(self.grid_size):
            self.add_single_wall(([self.grid_size - 1, y], [self.grid_size, y]))
        self.add_single_wall(([0, self.grid_size - 2], [0, self.grid_size - 1]))
        self.add_single_wall(([0, self.grid_size - 1], [1, self.grid_size - 1]))

    def add_single_wall(self, between):
        self.change_wall(between, 1)

    def remove_single_wall(self, between):
        self.change_wall(between, 0)

    def change_wall(self, between, change=None):
        # between is a list of coordinate np.arrays
        p1 = between[0]
        p2 = between[1]
        d = np.array(p2) - np.array(p1)
        if (d[0] != 0 and d[1] != 0) or (abs(d[0]) > 1 or abs(d[1]) > 1):
            return
        if self.valid_x_y(p1[0], p1[1]):
            p = get_wall_one_hot_pos(d)
            if change is None:
                change = abs(self.visible_state[p1[0]][p1[1]][p] - 1)
            self.visible_state[p1[0]][p1[1]][p] = change
            self.game_state[p1[0]][p1[1]][p] = change
        if self.valid_x_y(p2[0], p2[1]):
            p = get_wall_one_hot_pos(d, invert=True)
            if change is None:
                change = abs(self.visible_state[p2[0]][p2[1]][p] - 1)
            self.visible_state[p2[0]][p2[1]][p] = change
            self.game_state[p2[0]][p2[1]][p] = change

    def add_walls(self, between):
        # between is a list of coordinate np.arrays
        for new in between:
            self.add_single_wall(new)

    def render(self, state=None, flattened=False, reduced=False):
        if state is None:
            state = self.visible_state
        else:
            if flattened:
                state = np.reshape(state, (self.grid_size, self.grid_size, 4 + num_figures + num_figures))
        board = np.chararray((self.grid_size, self.grid_size), itemsize=7, unicode=True)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                board[x][y] = self.get_symbolic_for_pos(state[x][y], reduced=reduced)
        print(np.transpose(board))

    def get_symbolic_for_pos(self, pos, reduced=False):
        out = list('~~~~~~~')
        if pos[0] == 1:
            out[6] = '|'
        if pos[1] == 1:
            out[5] = '_'
        if pos[2] == 1:
            out[0] = '|'
        if pos[3] == 1:
            out[1] = '^'
        figure = hlp.one_hot_to_id(pos[4:4 + num_figures])
        if not reduced:
            goal = pos[4 + num_figures]
            if goal > 0:
                out[3:4] = goal_dict[goal][1]
        else:
            goal = pos[4 + num_figures:4 + 2 * num_figures]
            goal = hlp.one_hot_to_id(goal)
            if goal > -1:
                out[3] = fig_dict[goal][2]
                out[4] = '*'
        if figure != -1:
            out[2] = fig_dict[figure][2]
        return ''.join(out)

    def valid_x_y(self, x, y):
        return -1 < x < self.grid_size and -1 < y < self.grid_size


def get_inverse_action(act):
    fig = int(act / 4)
    dir = act % 4
    out = dir + 2
    if out >= 4:
        return fig * 4 + out - 4
    return fig * 4 + out


def print_action(act):
    if act is not None:
        act = actions[act]
        return "{} {}".format(fig_dict[act.figure][0], dir_dict[act.direction])
    else:
        return str(None)


def get_goal_color(gol):
    if isinstance(gol, str):
        for col in range(len(fig_dict)):
            if gol in fig_dict[col][1]:
                return col


def get_id_from_name(name, search_list):
    for idx in range(len(search_list)):
        if search_list[idx] == name:
            return idx


def get_wall_one_hot_pos(d, invert=False):
    if d[0] > 0 and not invert or d[0] < 0 and invert:
        return 0
    if d[1] > 0 and not invert or d[1] < 0 and invert:
        return 1
    if d[0] < 0 and not invert or d[0] > 0 and invert:
        return 2
    if d[1] < 0 and not invert or d[1] > 0 and invert:
        return 3


def get_wall_one_hot(d, invert=False):
    out = np.zeros(4, dtype=np.int)
    if d[0] > 0 and not invert or d[0] < 0 and invert:
        out[0] = 1
    if d[1] > 0 and not invert or d[1] < 0 and invert:
        out[1] = 1
    if d[0] < 0 and not invert or d[0] > 0 and invert:
        out[2] = 1
    if d[1] < 0 and not invert or d[1] > 0 and invert:
        out[3] = 1
    return out

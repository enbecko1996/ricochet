import numpy as np
import ricochet.hyperparameter as hp

grid_size = 4
figures = ['red', 'green', 'grey']
num_figures = len(figures)
goals = ['placeholder']
red_goals = ['red_star', 'red_line']
green_goals = ['green_star', 'green_line']
goals.extend(red_goals)
goals.extend(green_goals)
num_goals = len(goals)

goal_dict = {1: ('red_star', 'R*'), 2: ('red_line', 'R-'), 3: ('green_star', 'G*'), 4: ('green_line', 'G-')}
dir_dict = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
fig_dict = {0: ('red', red_goals, 'R'), 1: ('green', green_goals, 'G'), 2: ('grey', [], 'g')}


class action():
    def __init__(self, fig, direc):
        self.figure = fig
        self.direction = direc


actions = [action(f, d) for f in range(num_figures) for d in range(4)]


class environment():
    def __init__(self, g_size):
        self.the_state = np.zeros((g_size, g_size, 4 + num_figures + 1), dtype=np.int)
        self.reduced_state = np.zeros((g_size, g_size, 4 + num_figures + num_figures), dtype=np.int)
        self.grid_size = g_size
        self.num_figures = num_figures
        self.num_goals = num_goals
        self.observation_space = self.the_state.shape
        self.flattened_input_size = self.grid_size**2 * (4 + num_figures + num_figures)
        self.action_size = len(actions)
        self.cur_goal_name = None
        self.cur_goal = None
        self.cur_goal_pos = None
        self.cur_goal_color = None
        self.figs_on_board = []
        pass

    def reset(self, flattened=True):
        self.the_state = np.zeros((self.grid_size, self.grid_size, 4 + num_figures + 1), dtype=np.int)
        self.reduced_state = np.zeros((self.grid_size, self.grid_size, 4 + num_figures + num_figures), dtype=np.int)
        self.add_surrounding()
        self.figs_on_board.clear()
        self.add_single_figure([3, 3], 'red')
        self.add_single_figure([3, 0], 'grey')
        self.add_single_figure([2, 3], 'green')
        self.add_single_goal([0, 0], 'red_star')
        self.add_single_goal([3, 0], 'red_line')
        self.set_current_goal('red_line')
        if flattened:
            return self.get_flattened_reduced_state()
        else:
            return self.reduced_state

    def set_current_goal(self, gol):
        if isinstance(gol, str):
            self.cur_goal_name = gol
            self.cur_goal = goals.index(self.cur_goal_name)
            self.cur_goal_pos = self.get_pos_on_board(self.cur_goal_name)
            self.cur_goal_color = self.get_goal_color(self.cur_goal_name)
            self.reduced_state[self.cur_goal_pos[0]][self.cur_goal_pos[1]][4 + num_figures:4 + 2 * num_figures] = \
                as_one_hot(self.cur_goal_color, num_figures)
            # print(self.the_state, self.reduced_state)

    def get_flattened_reduced_state(self):
        return self.reduced_state.reshape(1, self.flattened_input_size)

    def get_pos_on_board(self, test):
        if test is not None:
            if isinstance(test, str):
                if test in figures:
                    test = as_one_hot(figures.index(test), num_figures)
                    for x in range(self.grid_size):
                        for y in range(self.grid_size):
                            if self.the_state[x][y][4:4 + num_figures] == test:
                                return x, y
                if test in goals:
                    test = goals.index(test)
                    for x in range(self.grid_size):
                        for y in range(self.grid_size):
                            if self.the_state[x][y][4 + num_figures] == test:
                                return x, y

    def get_goal_color(self, gol):
        if isinstance(gol, str):
            for col in range(len(fig_dict)):
                if gol in fig_dict[col][1]:
                    return col

    def step(self, a_int, flattened=True):
        a = actions[a_int]
        apply = self.apply_action_and_get_reward(a.figure, a.direction)
        if flattened:
            return self.get_flattened_reduced_state(), apply[0], apply[1], None
        else:
            return self.reduced_state, apply[0], apply[1], None

    # fig as id and not one-hot, direc as well
    def apply_action_and_get_reward(self, fig, direc):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.the_state[x][y][4 + fig] == 1:
                    # print(fig_dict[fig][0], dir_dict[direc])
                    new_x, new_y = self.iterate_step(x, y, direc)
                    if new_x == x and new_y == y:
                        return hp.in_wall_reward, False
                    else:
                        self.the_state[x][y][4 + fig] = 0
                        self.the_state[new_x][new_y][4 + fig] = 1
                        self.reduced_state[x][y][4 + fig] = 0
                        self.reduced_state[new_x][new_y][4 + fig] = 1
                        if self.cur_goal is not None \
                                and self.the_state[new_x][new_y][4 + self.num_figures] == self.cur_goal \
                                and self.cur_goal_name in fig_dict[fig][1]:
                            return hp.goal_reached_reward, True
                        else:
                            return hp.step_reward, False

    def iterate_step(self, x, y, direc):
        if self.the_state[x][y][direc] == 0:
            if direc == 0:
                if self.valid_x_y(x + 1, y) and max(self.the_state[x + 1][y][4:4 + self.num_figures]) == 0:
                    return self.iterate_step(x + 1, y, direc)
            if direc == 1:
                if self.valid_x_y(x, y + 1) and max(self.the_state[x][y + 1][4:4 + self.num_figures]) == 0:
                    return self.iterate_step(x, y + 1, direc)
            if direc == 2:
                if self.valid_x_y(x - 1, y) and max(self.the_state[x - 1][y][4:4 + self.num_figures]) == 0:
                    return self.iterate_step(x - 1, y, direc)
            if direc == 3:
                if self.valid_x_y(x, y - 1) and max(self.the_state[x][y - 1][4:4 + self.num_figures]) == 0:
                    return self.iterate_step(x, y - 1, direc)
        return x, y

    def add_single_figure(self, pos, fig):
        if isinstance(fig, str):
            fig = get_id_from_name(fig, figures)
        if fig is not None and fig not in self.figs_on_board and fig < self.num_figures:
            if -1 < pos[0] < self.grid_size and -1 < pos[1] < self.grid_size:
                self.the_state[pos[0]][pos[1]][4:4 + self.num_figures] = as_one_hot(fig, self.num_figures)
                self.reduced_state[pos[0]][pos[1]][4:4 + self.num_figures] = as_one_hot(fig, self.num_figures)
                self.figs_on_board.append(fig)

    def add_figures(self, pos_figures_tuples):
        for pos_fig in pos_figures_tuples:
            self.add_single_figure(pos_fig[0], pos_fig[1])

    def add_single_goal(self, pos, gol):
        if isinstance(gol, str):
            gol = get_id_from_name(gol, goals)
        if gol is not None and gol < self.num_goals:
            if self.valid_x_y(pos[0], pos[1]):
                self.the_state[pos[0]][pos[1]][4 + self.num_figures] = gol

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

    def add_single_wall(self, between):
        # between is a list of coordinate np.arrays
        p1 = between[0]
        p2 = between[1]
        d = np.array(p2) - np.array(p1)
        if (d[0] != 0 and d[1] != 0) or (abs(d[0]) > 1 or abs(d[1]) > 1):
            return
        if self.valid_x_y(p1[0], p1[1]):
            self.the_state[p1[0]][p1[1]][:4] += get_wall_one_hot(d)
            self.reduced_state[p1[0]][p1[1]][:4] += get_wall_one_hot(d)
        if self.valid_x_y(p2[0], p2[1]):
            self.the_state[p2[0]][p2[1]][:4] += get_wall_one_hot(d, invert=True)
            self.reduced_state[p2[0]][p2[1]][:4] += get_wall_one_hot(d, invert=True)

    def add_walls(self, between):
        # between is a list of coordinate np.arrays
        for new in between:
            self.add_single_wall(new)

    def render(self):
        board = np.chararray((self.grid_size, self.grid_size), itemsize=7, unicode=True)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                board[x][y] = self.get_symbolic_for_pos(self.the_state[x][y])
        print(np.transpose(board))

    def get_symbolic_for_pos(self, pos):
        out = list('~~~~~~~')
        if pos[0] == 1:
            out[6] = '|'
        if pos[1] == 1:
            out[5] = '_'
        if pos[2] == 1:
            out[0] = '|'
        if pos[3] == 1:
            out[1] = '^'
        figure = one_hot_to_id(pos[4:4 + num_figures])
        goal = pos[4 + num_figures]
        if figure != -1:
            out[2] = fig_dict[figure][2]
        if goal > 0:
            out[3:4] = goal_dict[goal][1]
        return ''.join(out)

    def valid_x_y(self, x, y):
        return -1 < x < self.grid_size and -1 < y < self.grid_size


def get_id_from_name(name, search_list):
    for idx in range(len(search_list)):
        if search_list[idx] == name:
            return idx


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


def as_one_hot(make_one_hot, num):
    out = np.zeros(num)
    out[make_one_hot] = 1
    return out


def one_hot_to_id(one_hot):
    for idx in range(len(one_hot)):
        if one_hot[idx] == 1:
            return idx
    return -1

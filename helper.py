import numpy as np


def as_one_hot(make_one_hot, num):
    out = np.zeros(num, dtype=np.int)
    out[make_one_hot] = 1
    return out


def one_hot_to_id(one_hot):
    for idx in range(len(one_hot)):
        if one_hot[idx] == 1:
            return idx
    return -1


def get_rotated_quadrant(quad, state, size):
    if 1 <= quad <= 4:
        if isinstance(state, str):
            state = load_state(state)
        out = np.zeros_like(state[0])
        half_g = int(size)
        if quad == 1:
            out = np.array(state[0])
        elif quad == 2:
            for x in range(0, half_g):
                for y in range(0, half_g):
                    out[x][y] = state[0][y][half_g - x - 1]
                    new_walls = np.zeros(4)
                    for i in range(4):
                        if out[x][y][i] == 1:
                            if i + 1 < 4:
                                new_walls[i + 1] = 1
                            else:
                                new_walls[i - 3] = 1
                    out[x][y][:4] = new_walls
        elif quad == 3:
            for x in range(0, half_g):
                for y in range(0, half_g):
                    out[x][y] = state[0][half_g - x - 1][half_g - y - 1]
                    new_walls = np.zeros(4)
                    for i in range(4):
                        if out[x][y][i] == 1:
                            if i + 2 < 4:
                                new_walls[i + 2] = 1
                            else:
                                new_walls[i - 2] = 1
                    out[x][y][:4] = new_walls
        elif quad == 4:
            for x in range(0, half_g):
                for y in range(0, half_g):
                    out[x][y] = state[0][half_g - y - 1][x]
                    new_walls = np.zeros(4)
                    for i in range(4):
                        if out[x][y][i] == 1:
                            if i + 3 < 4:
                                new_walls[i + 3] = 1
                            else:
                                new_walls[i - 1] = 1
                    out[x][y][:4] = new_walls
        return out


def combine_quadrants(file_name):
    one = load_state(file_name + "_0.npy")
    two = load_state(file_name + "_1.npy")
    lis = [one, two]
    np.save(file_name, np.array(lis))


def load_state(filename):
    return np.load(filename)


def to_wrkdir():
    import os
    try:
        print(os.getcwd())
        os.chdir("/home/nic/Dokumente/ricochet")
        print(os.getcwd())
    except FileNotFoundError:
        print("error")
        pass

"""for i in range(8):
    for j in range(0):
        states = load_state("quadrants/pre_"+str(i)+"_"+str(j)+".npy")
        lis = [states[0], get_rotated_quadrant(0, states, 8), get_rotated_quadrant(3, states,  8), get_rotated_quadrant(4, states, 8)]
        np.save("quadrants/pre_"+str(i)+"_"+str(j)+"", np.array(lis))"""

"""for i in range(8):
    combine_quadrants("quadrants/pre_"+str(i))"""


"""lis = []
for i in range(8):
    lis.append(load_state("quadrants/pre_" + str(i) + ".npy"))
np.save("quadrants/pre_all", np.array(lis))"""


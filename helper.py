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

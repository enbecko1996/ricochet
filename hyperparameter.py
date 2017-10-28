import math


e_decays = {'do_lin_decay': {'MAX_EPSILON': 1, 'MIN_EPSILON': 0.001, 'decay_per_epoch': -1, 'decay_per_step': - 1,
                             'decay_full_in_epochs': -1, 'decay_full_in_steps': -1, 'last_epoch': -1},
            'do_exp_decay': {'MAX_EPSILON': 1, 'MIN_EPSILON': 0.001, 'LAMBDA': 0.0000016}}


def do_exp_decay(inp_dict, e, cur_epoch=0, cur_steps=0):
    return inp_dict['MIN_EPSILON'] + (inp_dict['MAX_EPSILON'] - inp_dict['MIN_EPSILON']) * \
                                     math.exp(-inp_dict['LAMBDA'] * cur_steps)


def do_lin_decay(inp_dict, e, cur_epoch=0, cur_steps=0):
    if cur_epoch > inp_dict['last_epoch'] and inp_dict['decay_per_epoch'] > -1:
        inp_dict['last_epoch'] = cur_epoch
        return e - inp_dict['decay_per_epoch']
    elif inp_dict['decay_per_step'] > -1:
        return e - inp_dict['decay_per_step']
    elif inp_dict['decay_full_in_epochs'] > -1:
        return inp_dict['MAX_EPSILON'] - (inp_dict['MAX_EPSILON'] - inp_dict['MIN_EPSILON']) * \
               cur_epoch / inp_dict['decay_full_in_epochs'] if e > inp_dict['MIN_EPSILON'] else inp_dict['MIN_EPSILON']
    elif inp_dict['decay_full_in_steps'] > -1:
        return inp_dict['MAX_EPSILON'] - (inp_dict['MAX_EPSILON'] - inp_dict['MIN_EPSILON']) * \
               cur_steps / inp_dict['decay_full_in_steps'] if e > inp_dict['MIN_EPSILON'] else inp_dict['MIN_EPSILON']
    return e


class AgentsHyperparameters:
    def __init__(self):
        # neural_net
        self.MAX_STEPS = 300
        self.EPOCHS = 75000
        self.LEARNING_RATE_EARLY_STOP = 50
        self.LEARNING_RATE_MIN = 0.01
        self.LEARNING_RATE_MAX = 0.01

        self.MEMORY_CAPACITY = 100000
        self.REPLAY = 2
        self.BATCH_SIZE = 64

        self.GAMMA = 0.99

        self.E_DECAY = e_decays

        # self.MAX_EPSILON = 1
        # self.MIN_EPSILON = 0.01
        # self.LAMBDA = 0.0000016  # speed of decay

        self.UPDATE_TARGET_FREQUENCY = 1000

        self.MINIMUM_CAPTURE_THRESH = 300

        self.REWARD_IN_WALL = -2.0
        self.REWARD_GOAL_REACHED = 80.0
        self.REWARD_STEP = -1.0

        self.DEBUG_LOG_EPOCHS = 300
        self.DEBUG_SNAPSHOT = 1200

    def get_attr(self, name):
        if name == 'LEARNING_RATE':
            if hasattr(self, 'LEARNING_RATE'):
                return self.LEARNING_RATE
            elif hasattr(self, 'LEARNING_RATE_MAX'):
                return self.LEARNING_RATE_MAX

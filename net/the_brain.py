# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use a full DQN implementation
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at:
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
#
# author: Jaromir Janisch, 2016

import math
import random
from collections import deque

# import gym
import matplotlib.pyplot as plt
import numpy


# ----------------DEBUG-----------------


class stats_collector():
    collect_step_counts = True
    collect_big_step_counts = True
    collect_n_last_games = 0
    collect_rewards = True
    steps = []
    big_steps = []
    games = deque(maxlen=collect_n_last_games)
    reward = []
    diction = {'steps': (collect_step_counts, steps), 'games': (collect_n_last_games > 0, games),
               'rewards': (collect_rewards, reward), 'big_steps': (collect_big_step_counts, big_steps)}

    def reset(self):
        self.steps.clear()
        self.reward.clear()

    def collect(self, name, val):
        if self.diction[name][0]:
            self.diction[name][1].append(val)

    def plt(self, name):
        if self.diction[name][0]:
            plt.plot(self.diction[name][1])
            plt.show()
#

class debugger:
    render = False
    log_epoch = 30
    snapshot = 300

    def reset(self):
        pass


stats = stats_collector()
debug = debugger()


# ---------------huber-------------------------------------
HUBER_LOSS_DELTA = 1.0


def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)


# -------------------- BRAIN ---------------------------
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import *


class Brain:
    def __init__(self, handler, conv, stateCnt=0, actionCnt=0, grid_size=0, dims=0, worker=None, feeder=None):
        self.handler = handler
        self.conv = conv
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.grid_size = grid_size
        self.dims = dims

        if worker is None:
            self.model = self._createModel()
        else:
            self.model = load_model(worker, custom_objects={'huber_loss': huber_loss})
        if feeder is None:
            self.model_ = self._createModel()
        else:
            self.model_ = load_model(feeder, custom_objects={'huber_loss': huber_loss})

    def _createModel(self):
        model = Sequential()

        if self.conv:
            model.add(Conv2D(30, kernel_size=4, padding='same', input_shape=(self.grid_size, self.grid_size, self.dims)))
            model.add(MaxPooling2D())
            model.add(Flatten())
            model.add(Dense(units=80, activation='relu'))
            model.add(Dense(units=self.actionCnt, activation='linear'))
        else:
            model.add(Dense(units=int(self.stateCnt * (2./3) + self.actionCnt), activation='relu', input_dim=self.stateCnt))
            # model.add(Dense(units=200, activation='relu'))
            model.add(Dense(units=self.actionCnt, activation='linear'))

        opt = RMSprop(lr=self.handler.hp.LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=self.handler.hp.BATCH_SIZE, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(np.expand_dims(s, 0), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ )
    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = []

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity


# -------------------- AGENT ---------------------------
class Agent:
    def __init__(self, handler, conv, state_count=0, action_count=0, grid_size=0, dims=0, worker=None, feeder=None):
        self.handler = handler
        self.steps = 0
        self.epsilon = self.handler.hp.MAX_EPSILON
        self.conv = conv
        self.stateCnt = state_count
        self.actionCnt = action_count
        print("hey45")

        self.brain = Brain(self.handler, conv, state_count, action_count, grid_size, dims, worker, feeder)
        print("hey6")
        self.memory = Memory(handler.hp.MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randrange(0, self.actionCnt)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        if self.steps % self.handler.hp.UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = self.handler.hp.MIN_EPSILON + (self.handler.hp.MAX_EPSILON - self.handler.hp.MIN_EPSILON) \
                                                    * math.exp(-self.handler.hp.LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(self.handler.hp.BATCH_SIZE)
        batchLen = len(batch)

        if not self.conv:
            no_state = numpy.zeros(self.stateCnt)
        else:
            no_state = numpy.zeros_like(batch[0][0])

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        if not self.conv:
            x = numpy.zeros((batchLen, self.stateCnt))
        else:
            base_shape = batch[0][0].shape
            x = numpy.zeros((batchLen, base_shape[0], base_shape[1], base_shape[2]))
        y = numpy.zeros((batchLen, self.actionCnt))

        # print("\n\n___________________________\nNEW replay\n__________________________")
        for i in range(batchLen):
            o = batch[i]
            s = o[0]
            # the_environment.my_env.render(state=s, flattened=True, reduced=True)
            a = o[1]
            r = o[2]
            s_ = o[3]
            t = list(p[i])
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.handler.hp.GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = list(t)

            """print("\n\n___________________________\nNEW STATE/ACTION\n__________________________")
            the_environment.my_env.render(state=s, flattened=True, reduced=True)
            print("taken action = {}, reward = {}, next state = ".format(a, r))
            print("current = {}, aim = {}".format(p[i], y[i]))
            the_environment.my_env.render(state=s_, flattened=True, reduced=True)"""

        self.brain.train(x, y)


class RandomAgent:
    def __init__(self, handler, action_count):
        self.handler = handler
        self.memory = Memory(handler.hp.MEMORY_CAPACITY)
        self.actionCnt = action_count

    def act(self, s):
        return random.randrange(0, self.actionCnt)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass


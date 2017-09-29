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

import random
import numpy
import math
import gym
from collections import deque
import matplotlib.pyplot as plt
import ricochet.hyperparameter as hyperparameter
import PyQt5 as qt

import sys
from keras import backend as K

import tensorflow as tf


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


class debugger():
    render = False
    log_epoch = 30

    def reset(self):
        pass


stats = stats_collector()
debug = debugger()


# ---------------huber-------------------------------------

def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < hp.HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = hp.HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * hp.HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)


# -------------------- BRAIN ---------------------------
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import *


class Brain:
    def __init__(self, stateCnt, actionCnt, worker=None, feeder=None):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        if worker is None:
            self.model = self._createModel()
        else:
            self.model = load_model(worker)
        if feeder is None:
            self.model_ = self._createModel()
        else:
            self.model_ = load_model(feeder)

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=200, activation='relu', input_dim=self.stateCnt))
        model.add(Dense(units=200, activation='relu'))
        model.add(Dense(units=self.actionCnt, activation='linear'))

        opt = RMSprop(lr=hp.LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=hp.BATCH_SIZE, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

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
    steps = 0
    epsilon = hp.MAX_EPSILON

    def __init__(self, state_count, action_count, worker=None, feeder=None):
        print("hey9")
        self.stateCnt = state_count
        self.actionCnt = action_count

        self.brain = Brain(state_count, action_count, worker, feeder)
        print("hey91")
        self.memory = Memory(hp.MEMORY_CAPACITY)
        print("hey6")

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randrange(0, self.actionCnt)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        if self.steps % hp.UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # debug the Q function in poin S
        """if self.steps % 100 == 0:
            S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
            pred = agent.brain.predictOne(S)
            print(pred[0])
            sys.stdout.flush()"""

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = hp.MIN_EPSILON + (hp.MAX_EPSILON - hp.MIN_EPSILON) * math.exp(-hp.LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(hp.BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
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
                t[a] = r + hp.GAMMA * numpy.amax(p_[i])

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


# -------------------- ENVIRONMENT ---------------------
import ricochet.game as the_game


class Environment:
    def __init__(self, problem=None):
        self.samples = []
        if isinstance(problem, the_game.Environment):
            self.my_env = problem
        else:
            self.problem = problem
            self.my_env = gym.make(problem)

    def run(self, current_agent, board_style):
        s = self.my_env.reset(figure_style='random', board_style=board_style)
        total_reward = 0

        steps = 0
        while steps < 300:
            # self.env.render()

            steps += 1
            a = current_agent.act(s)

            s_, r, done, info = self.my_env.step(a)

            if done:  # terminal state
                s_ = None

            current_agent.observe((s, a, r, s_))
            current_agent.replay()

            s = s_
            total_reward += r

            if done:
                break
        return steps, total_reward

    def play_game(self, current_agent, board_style):
        s = self.my_env.reset(figure_style='random', board_style=board_style)
        total_reward = 0
        steps = 0
        while steps < 500:
            steps += 1
            self.my_env.render()
            prediction = current_agent.brain.predictOne(s)
            print(prediction)
            if random.random() < 0.05:
                print("random")
                a = random.randint(0, current_agent.actionCnt - 1)
            else:
                a = numpy.argmax(prediction)
            print("max = {}, {}, taken = {}, {}".format(numpy.argmax(prediction),
                                                        the_game.print_action(numpy.argmax(prediction)), a,
                                                        the_game.print_action(a)))
            s_, r, done, info = self.my_env.step(a)
            if done:  # terminal state
                s_ = None
            s = s_
            total_reward += r
            if done:
                break
        return steps, total_reward


# -------------------- MAIN ----------------------------
import ricochet.helper as hlp
import datetime
from pathlib import Path


brd_stl = [[2, 0], [3, 1], [6, 0], [0, 1]]


class Handler:
    def __init__(self, name, worker=None, feeder=None, hyperparams=None):
        hlp.to_wrkdir()
        if hyperparams is not None:
            self.hp = hyperparams
        self.the_environment = Environment(the_game.Environment(16, self.hp))
        self.training = False
        self.finished = False
        self.save = True
        self.name = name
        self.stateCnt = self.the_environment.my_env.flattened_input_size
        self.actionCnt = self.the_environment.my_env.action_size

        self.agent = Agent(self.stateCnt, self.actionCnt, worker, feeder)
        self.randomAgent = RandomAgent(self.actionCnt)

    def set_hps(self, hyperparams):
        self.hp = hyperparams

    def initialize(self, board_style):
        print("performing Random Agent")
        while not self.randomAgent.memory.isFull():
            self.the_environment.run(self.randomAgent, board_style)

        self.agent.memory.samples = self.randomAgent.memory.samples
        self.randomAgent = None

    def start_training(self, board_style):
        self.training = True
        self.finished = False
        self.save = True
        epoch = 0
        while epoch < hp.EPOCHS and self.training:
            epoch += 1
            steps, reward = self.the_environment.run(self.agent, board_style)
            stats.collect('steps', steps)
            stats.collect('rewards', reward)
            if epoch % debug.log_epoch == 0:
                print("\n\n__________________\nNEW GAME\n___________________")
                game_steps, game_rewards = self.the_environment.play_game(self.agent, board_style)
                print("game_steps = {}, game_reward = {}".format(game_steps, game_rewards))
                print("avg. reward = {0}".format(np.mean(stats.reward[-debug.log_epoch:])))
                print("episode: {}/{}, steps: {}".format(epoch, hp.EPOCHS, steps),
                      "avg. steps last finishes {} = {}".format(debug.log_epoch,
                                                                np.mean(stats.steps[-debug.log_epoch:])),
                      "epsilon = {}".format(self.agent.epsilon))
                stats.collect('big_steps', np.mean(stats.steps[-debug.log_epoch:]))
                stats.steps.clear()
        if self.save:
            my_file = Path("models/" + self.name + "_worker.h5")
            my_file_2 = Path("models/" + self.name + "_feeder.h5")
            i = 0
            while my_file.is_file() or my_file_2.is_file():
                i += 1
                print(i)
                my_file = Path("models/" + self.name + "_worker_" + str(i) + ".h5")
                my_file_2 = Path("models/" + self.name + "_feeder_" + str(i) + ".h5")
            self.agent.brain.model.save(my_file)
            self.agent.brain.model_.save(my_file_2)
            np.save("models/" + self.name + "_stats", np.array(stats.big_steps))
        self.finished = True
        print("finished")

    def stop_training(self):
        self.save = False
        self.training = False

    def pause_training(self):
        self.save = True
        self.training = False


"""PROBLEM = 'CartPole-v0'
the_environment = Environment(the_game.Environment(16))

stateCnt = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

stateCnt = the_environment.my_env.flattened_input_size
actionCnt = the_environment.my_env.action_size

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

try:
    print("performing Random Agent")
    while not randomAgent.memory.isFull():
        the_environment.run(randomAgent)

    print("hello")
    agent.memory.samples = randomAgent.memory.samples
    randomAgent = None

    print("performing Agent")
    epoch = 0
    while epoch < hp.EPOCHS:
        epoch += 1
        steps, reward = the_environment.run(agent)
        stats.collect('steps', steps)
        stats.collect('rewards', reward)
        if epoch % debug.log_epoch == 0:
            print("\n\n__________________\nNEW GAME\n___________________")
            game_steps, game_rewards = the_environment.play_game(agent)
            print("game_steps = {}, game_reward = {}".format(game_steps, game_rewards))
            print("avg. reward = {0}".format(np.mean(stats.reward[-debug.log_epoch:])))
            print("episode: {}/{}, steps: {}".format(epoch, hp.EPOCHS, steps),
                  "avg. steps last finishes {} = {}".format(debug.log_epoch,
                                                            np.mean(stats.steps[-debug.log_epoch:])),
                  "epsilon = {}".format(agent.epsilon))
finally:
    print("saving")
    agent.brain.model.save("cartpole-dqn.h5")"""

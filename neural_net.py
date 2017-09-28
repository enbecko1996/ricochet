from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, RMSprop
import numpy as np
import random as rand
import ricochet.game as envi
import ricochet.hyperparameter as hp
import matplotlib.pyplot as plt
import atexit
from collections import deque
import time


class stats_collector():
    collect_step_counts = True
    collect_n_last_games = 0
    collect_rewards = True
    steps = []
    games = deque(maxlen=collect_n_last_games)
    reward = []
    diction = {'steps': (collect_step_counts, steps), 'games': (collect_n_last_games > 0, games),
               'rewards': (collect_rewards, reward)}

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
    log_epoch = 400

    def reset(self):
        pass


stats = stats_collector()
debug = debugger()


class Q_network():
    def __init__(self, envir, model=None):
        self.memory = []
        self.env = envir
        self.model = model
        if self.model is None:
            self.make_model()
        self.e = hp.e_start

    def make_model(self):
        self.model = Sequential()
        self.model.add(Dense(300, activation='relu', input_shape=(self.env.flattened_input_size,)))
        # model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        self.model.add(Dense(150, activation='relu'))
        # model.add(Dropout(0.2))

        self.model.add(Dense(self.env.action_size, activation='linear'))
        print('action_size =', self.env.action_size)

        rms = RMSprop(lr=hp.lr)
        self.model.compile(loss='mse', optimizer=rms)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            del self.memory[0]

    def act(self, state, e_greedy, only_valid=False):
        if only_valid:
            cur_actions = self.env.get_valid_actions()
            cur_size = len(cur_actions)
            if np.random.rand() <= e_greedy:
                return cur_actions[rand.randrange(cur_size)]
            act_values = np.take(self.model.predict(state)[0], cur_actions)
        else:
            if np.random.rand() <= e_greedy:
                return rand.randrange(self.env.action_size)
            act_values = self.model.predict(state)[0]
        return np.argmax(act_values)

    def replay(self, batch_size, qmax_func, last=False):
        if len(self.memory) > batch_size:
            if not last:
                mini_batch = rand.sample(self.memory, batch_size)
            else:
                mini_batch = self.memory[- batch_size + 1:]
        else:
            mini_batch = self.memory
        x_train = []
        y_train = []
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                # print(stats, action, reward, next_state, done)
                Q_max = qmax_func(self.model.predict(next_state), self.env, next_state)
                target = reward + hp.gamma * Q_max
            y = self.model.predict(state)[0]
            y[action] = target
            x_train.append(state.reshape(self.env.flattened_input_size,))
            y_train.append(y.reshape(self.env.action_size,))
        self.model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, epochs=1, verbose=0)
        if self.e > hp.e_min:
            self.e -= hp.e_decay

    def train(self, method):
        method(self)

    def online_learning(self):
        # Iterate the game
        cum_r = 0.
        for epoch in range(hp.epochs):
            s = self.env.reset(flattened=True)
            for steps in range(500):
                a = self.act(s, self.e)
                first_Q = self.model.predict(s, batch_size=1)
                s1, r, d, _ = self.env.step(a, flattened=True)
                cum_r += r
                next_Q = self.model.predict(s1, batch_size=1)
                max_Q = np.max(next_Q)
                y = np.zeros((1, self.env.action_size))
                y[:] = first_Q[:]
                if r != hp.goal_reached_reward:
                    update = r + hp.gamma * max_Q
                else:
                    update = r
                y[0][a] = update
                self.model.fit(s, y, batch_size=1, nb_epoch=1, verbose=0)
                s = s1
                if d:
                    stats.collect('steps', steps)
                    if epoch > 0 and epoch % debug.log_epoch == 0:
                        self.model.save('my_model.h5')
                        print("avg. reward = {0}".format(cum_r / debug.log_epoch))
                        print("episode: {}/{}, steps: {}".format(epoch, hp.epochs, steps),
                              "avg. steps last finishes {} = {}".format(debug.log_epoch,
                                                                        np.mean(stats.steps[-debug.log_epoch:])),
                              "epsilon = {}".format(self.e))
                        cum_r = 0.
                    break
            if self.e > hp.e_min:
                self.e -= hp.e_decay
        stats.plt('steps')
        self.model.save('my_model.h5')

    def experience_replay(self):
        # Iterate the game
        cum_r = 0.
        for epoch in range(hp.epochs):
            s = self.env.reset(flattened=True)
            for steps in range(500):
                a = self.act(s, self.e)
                s1, r, d, _ = self.env.step(a, flattened=True)
                cum_r += r
                self.remember(s, a, r, s1, d)
                s = s1
                if d:
                    stats.collect('steps', steps)
                    if epoch > 0 and epoch % debug.log_epoch == 0:
                        self.model.save('my_model_lr='+str(hp.lr)+'_'+str(time.time())+'.h5')
                        print("avg. reward = {0}".format(cum_r / debug.log_epoch))
                        print("episode: {}/{}, steps: {}".format(epoch, hp.epochs, steps),
                              "avg. steps last finishes {} = {}".format(debug.log_epoch,
                                                                        np.mean(stats.steps[-debug.log_epoch:])),
                              "epsilon = {}".format(self.e))
                        cum_r = 0.
                    break
            self.replay(32, qmax_func=all_qmax)
            if self.e > hp.e_min:
                self.e -= hp.e_decay
        stats.plt('steps')
        self.model.save('my_model.h5')


def valid_qmax(prediction, env, state):
    return np.max(np.take(prediction, env.get_valid_actions(state, flattened=True)))


def all_qmax(prediction, env=None, state=None):
    return np.max(prediction)


@atexit.register
def termination():
    if stats.collect_step_counts:
        plt.plot(stats.steps)
        plt.show()


def test_agent(agent):
    s = agent.env.reset()
    for steps in range(500):
        agent.env.render()
        q_cur = agent.model.predict(s)[0]
        print(q_cur)
        a = agent.act(s, 0.05)
        s1, r, d, _ = agent.env.step(a, flattened=True)
        s = s1
        if d:
            print("agent had a score of {} in the test".format(steps))
            break


if __name__ == "__main__":
    # initialize gym environment and the agent
    environment = envi.Environment(4)
    q_agent = Q_network(environment)
    q_agent.train(Q_network.experience_replay)
    test_agent(q_agent)

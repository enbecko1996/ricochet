from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, RMSprop
import numpy as np
import random as rand
import ricochet.environment as envi
import ricochet.hyperparameter as hp
import matplotlib.pyplot as plt
import atexit
from collections import deque


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
    log_epoch = 100

    def reset(self):
        pass


stats = stats_collector()
debug = debugger()


class Q_network():
    def __init__(self, envir):
        self.memory = deque(maxlen=2000)
        self.env = envir
        self.model = None
        self.make_model()
        self.e = hp.e_start

    def make_model(self):
        self.model = Sequential()
        self.model.add(Dense(300, activation='relu', kernel_initializer='lecun_uniform', input_shape=(self.env.flattened_input_size,)))
        # model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        # self.model.add(Dense(150, activation='relu'))
        # model.add(Dropout(0.2))

        self.model.add(Dense(self.env.action_size, activation='linear'))
        print('action_size =', self.env.action_size)

        rms = RMSprop()
        self.model.compile(loss='mse', optimizer=rms)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.e:
            return rand.randrange(self.env.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) > batch_size:
            mini_batch = rand.sample(self.memory, batch_size)
        else:
            mini_batch = self.memory
        x_train = []
        y_train = []
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                # print(stats, action, reward, next_state, done)
                target = reward + hp.gamma * \
                                  np.max(self.model.predict(next_state)[0])
            y = self.model.predict(state)[0]
            y[action] = target
            x_train.append(state.reshape(self.env.flattened_input_size,))
            y_train.append(y.reshape(self.env.action_size,))
        self.model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, epochs=1, verbose=0)
        if self.e > hp.e_min:
            self.e -= hp.e_decay


@atexit.register
def termination():
    if stats.collect_step_counts:
        plt.plot(stats.steps)
        plt.show()


def train_agent(agent):
    # Iterate the game
    cum_r = 0.
    for epoch in range(hp.epochs):
        s = agent.env.reset(flattened=True)
        for steps in range(500):
            a = agent.act(s)
            s1, r, d, _ = agent.env.step(a, flattened=True)
            cum_r += r
            agent.remember(s, a, r, s1, d)
            s = s1
            if d:
                stats.collect('steps', steps)
                if epoch > 0 and epoch % debug.log_epoch == 0:
                    print("avg. reward = {0}".format(cum_r / debug.log_epoch))
                    print("episode: {}/{}, steps: {}".format(epoch, hp.epochs, steps),
                          "avg. steps last finishes {} = {}".format(debug.log_epoch, np.mean(stats.steps[-debug.log_epoch:])),
                          "epsilon = {}".format(agent.e))
                    cum_r = 0.
                break
        agent.replay(64)
    stats.plt('steps')


def test_agent(agent):
    s = agent.env.reset()
    for steps in range(500):
        a = agent.act(s)
        s1, r, d, _ = agent.env.step(a, flattened=True)
        s = s1
        if d:
            print("agent had a score of {} in the test".format(steps))
            agent.env.render()
            break


if __name__ == "__main__":
    # initialize gym environment and the agent
    environment = envi.environment(4)
    q_agent = Q_network(environment)
    train_agent(q_agent)

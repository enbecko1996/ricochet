from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import numpy as np
import random as rand
import ricochet.environment as envi
import ricochet.hyperparameter as hp
import matplotlib.pyplot as plt

env = envi.environment(4)


class stats_collector():
    collect_steps = True
    collect_rewards = True
    steps = []
    reward = []
    
    def reset(self):
        self.steps.clear()
        self.reward.clear()


class debugger():
    render = False
    log_epoch = 200

    def reset(self):
        pass


stats = stats_collector()
debug = debugger()


class Q_network():
    def __init__(self):
        self.model = Sequential()
        self.model = Sequential()
        self.model.add(Dense(164, activation='relu', input_shape=(env.flattened_input_size,)))
        # model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        self.model.add(Dense(150, activation='relu'))
        # model.add(Dropout(0.2))

        self.model.add(Dense(env.action_size, activation='softmax'))

        rms = RMSprop()
        self.model.compile(loss='mse', optimizer=rms)

    e = hp.e_start    
    def train(self):
        stats.reset()
        for epoch in range(hp.epochs):
            steps = 0
            s = env.reset(flattened=True)
            d = False
            if epoch % debug.log_epoch == 0:
                print("game = {}".format(epoch))
            while not d:
                steps += 1
                if debug.render:
                    env.render()
                first_Q = self.model.predict(s, batch_size=1)
                a = np.argmax(first_Q)
                if rand.uniform(0, 1) < self.e:
                    a = rand.randint(0, env.action_size - 1)
                s1, r, d, _ = env.step(a, flattened=True)
                next_Q = self.model.predict(s1.reshape(1, env.flattened_input_size), batch_size=1)
                max_Q = np.max(next_Q)
                y = np.zeros((1, env.action_size))
                y[:] = first_Q[:]
                if r != hp.goal_reached_reward:
                    update = r + hp.gamma * max_Q
                else:
                    update = r
                y[0][a] = update
                self.model.fit(s, y, batch_size=1, nb_epoch=1, verbose=0)
                s = s1
                if d:
                    if stats.collect_steps:
                        stats.steps.append(steps)
                    if debug.render:
                        env.render()
            if self.e > 0.05:
                self.e -= (1 / hp.epochs)
        plt.plot(stats.steps)
        plt.show()


q = Q_network()
q.train()

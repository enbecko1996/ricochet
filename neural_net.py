from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import numpy as np
import random as rand
import ricochet.environment as envi

env = envi.environment(4)


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
        print(self.model.predict(env.the_state.reshape(1, env.flattened_input_size), batch_size=1))

    epochs = 1000
    gamma = 0.9
    e = 1
    def train(self):
        for epoch in range(self.epochs):
            s = env.reset()
            d = False
            print("game = {}".format(epoch))
            while not d:
                env.render()
                first_Q = self.model.predict(s.reshape(1, env.flattened_input_size), batch_size=1)
                a = np.argmax(first_Q)
                if rand.uniform(0, 1) < self.e:
                    a = rand.randint(0, env.action_size - 1)
                print(first_Q, a)
                s1, r, d, _ = env.step(a)
                next_Q = self.model.predict(s1.reshape(1, env.flattened_input_size), batch_size=1)
                max_Q = np.max(next_Q)
                y = np.zeros((1, env.action_size))
                y[:] = first_Q[:]
                if r != 10:
                    update = r + self.gamma * max_Q
                else:
                    update = r
                y[0][a] = update
                self.model.fit(s.reshape(1, env.flattened_input_size), y, batch_size=1, nb_epoch=1, verbose=0)
                s = s1
            if self.e > 0.1:
                self.e -= (1 / self.epochs)


q = Q_network()
q.train()

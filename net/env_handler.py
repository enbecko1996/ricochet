import random

import gym
import numpy

import game as the_game
import numpy as np
from queue import Queue
import helper as hlp


class Environment:
    def __init__(self, conv, problem=None):
        self.samples = []
        self.conv = conv
        if isinstance(problem, the_game.Environment):
            self.my_env = problem
        else:
            self.problem = problem
            self.my_env = gym.make(problem)

    def get_action_cnt(self):
        return self.my_env.action_size

    def get_state_cnt(self):
        return self.my_env.flattened_input_size

    def get_grid_size(self):
        return self.my_env.grid_size

    def get_dims(self):
        return self.my_env.dims

    def run(self, current_agent, board_style, epoch=0):
        s = self.my_env.reset(flattened=not self.conv, figure_style='random', board_style=board_style)
        total_reward = 0

        steps = 0
        while steps < current_agent.handler.hp.MAX_STEPS:
            # self.my_env.render()

            steps += 1
            a = current_agent.act(s, steps)

            s_, r, done, info = self.my_env.step(a, flattened=not self.conv, hyperparams=current_agent.handler.hp)

            if done:  # terminal state
                s_ = None
            current_agent.observe((s, a, r, s_), epoch)
            if steps % current_agent.handler.hp.REPLAY == 0:
                current_agent.replay()

            s = s_
            total_reward += r

            if done:
                break
        if steps < current_agent.handler.hp.REPLAY:
            current_agent.replay()
        return steps, total_reward

    def play_game(self, current_agent, board_style='none', env_on=None, return_actions=False, ran=0):
        if env_on is not None:
            self.my_env.set_state_from_other_env(env_on)
            s = self.my_env.get_flattened_reduced_state() if not self.conv else self.my_env.get_reduced_state()
        else:
            s = self.my_env.reset(flattened=not self.conv, figure_style='random', board_style=board_style)
        total_reward = 0
        steps = 0
        actions = []
        # self.my_env.render(state=self.my_env.get_flattened_reduced_state(), flattened=True, reduced=True)
        while steps < current_agent.handler.hp.MAX_STEPS:
            steps += 1
            prediction = current_agent.brain.predictOne(s)
            if ran > 0:
                rand = random.random()
                rand = 1 - rand**steps
                for i in range(8):
                    i = 7 - i
                    if rand < ran**i:
                        for j in range(i):
                            prediction[numpy.argmax(prediction)] = -99999
                        a = numpy.argmax(prediction)
                        break
            else:
                a = numpy.argmax(prediction)

            if return_actions:
                actions.append(a)
            s_, r, done, info = self.my_env.step(a, flattened=not self.conv)
            if done:  # terminal state
                s_ = None
            s = s_
            total_reward += r
            if done:
                break
        if return_actions:
            return steps, total_reward, actions
        else:
            return steps, total_reward

    def brute_force(self, board_style='none', env_on=None, return_actions=False, ran=0):
        s = self.my_env.reset(flattened=False, figure_style='random', board_style=board_style)
        k = 4
        brute_result = self.iterate_all_valid_on(s, 0, k, Queue())
        while brute_result is None:
            k += 1
            brute_result = self.iterate_all_valid_on(s, 0, k, Queue())
        print(brute_result)

    def iterate_all_valid_on(self, state, k, k_max, the_queue):
        save_state = np.array(state)
        acts = self.my_env.get_valid_actions(state=state, flattened=False)
        if k == k_max:
            for a in acts:
                state = np.array(save_state)
                s_, r, done, _ = self.my_env.step(a, state=state, flattened=False)
                if done:
                    the_queue.put(a)
                    return the_queue
            return None
        else:
            for a in acts:
                if k <= k_max - 4:
                    for indent in range(k):
                        print('  ', end='')
                    print(f'{k + 1}: {acts.index(a)} / {len(acts)}')
                state = np.array(save_state)
                s_, r, done, _ = self.my_env.step(a, state=state, flattened=False)
                if done:
                    the_queue.put(a)
                    return the_queue
                else:
                    the_queue.put(a)
                    next_queue = self.iterate_all_valid_on(s_, k + 1, k_max, the_queue)
                    if next_queue is not None:
                        return next_queue
                    else:
                        the_queue.get_nowait()
            return None


def do_stuff():
    the_environment = Environment(False, the_game.Environment(8))
    print(the_environment.brute_force(board_style='small'))
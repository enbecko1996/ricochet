import random

import gym
import numpy

import game as the_game


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

    def run(self, current_agent, board_style):
        s = self.my_env.reset(flattened=not self.conv, figure_style='random', board_style=board_style)
        total_reward = 0

        steps = 0
        while steps < current_agent.handler.hp.MAX_STEPS:
            # self.env.render()

            steps += 1
            a = current_agent.act(s)

            s_, r, done, info = self.my_env.step(a, flattened=not self.conv)

            if done:  # terminal state
                s_ = None
            current_agent.observe((s, a, r, s_))
            if steps % current_agent.handler.hp.REPLAY == 0:
                current_agent.replay()

            s = s_
            total_reward += r

            if done:
                break
        if steps < current_agent.handler.hp.REPLAY:
            current_agent.replay()
        return steps, total_reward

    def play_game(self, current_agent, board_style='none', env_on=None, return_actions=False):
        if env_on is not None:
            self.my_env.set_state(env_on)
            s = self.my_env.get_flattened_reduced_state() if not self.conv else self.my_env.get_reduced_state()
        else:
            s = self.my_env.reset(flattened=not self.conv, figure_style='random', board_style=board_style)
        total_reward = 0
        steps = 0
        actions = []
        while steps < current_agent.handler.hp.MAX_STEPS:
            steps += 1
            # self.my_env.render()
            prediction = current_agent.brain.predictOne(s)
            rand = random.random()
            if rand < 0.05**2:
                prediction[numpy.argmax(prediction)] = -99999
                prediction[numpy.argmax(prediction)] = -99999
                a = numpy.argmax(prediction)
            elif rand < 0.05:
                prediction[numpy.argmax(prediction)] = -99999
                a = numpy.argmax(prediction)
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



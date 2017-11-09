import game as emulator
import math
import numpy as np


the_emulator = emulator.Environment()


class Node:
    def __init__(self, parent, action, value=1, node_reward=0, is_end_node=False):
        self.parent = parent
        self.value = value
        self.action = action
        self.node_reward = node_reward
        self.is_end_node = is_end_node
        self.play_count = 1
        self.child_nodes = []

    def get_child_nodes(self):
        return self.child_nodes

    def add_child_node(self, action, value, node_reward, is_end_node):
        new_node = Node(self, action, value=value, node_reward=node_reward, is_end_node=is_end_node)
        self.child_nodes.append(new_node)
        return new_node

    def get_best_child_according_to_ucb(self, mcts_obj):
        max_val = 0
        if len(self.child_nodes) > 0:
            max_node = self.child_nodes[0]
            for child in self.child_nodes:
                if child.is_leaf_node:
                    return child
                my_val = child.value + math.sqrt((2 * math.log1p(mcts_obj.n)) / child.play_count)
                if my_val > max_val:
                    max_val = my_val
                    max_node = child
            return max_node

    def has_childs(self):
        return len(self.child_nodes) > 0

    def has_all_childs(self):
        return len(self.child_nodes) >= the_emulator.action_size

    def get_parent(self):
        return self.parent

    def get_child_count(self):
        return len(self.child_nodes)

    def has_parent(self):
        return self.parent is not None


class MCTS:

    def __init__(self):
        self.tree = Node(None, None)
        self.state = None
        self.agent = None
        self.n = 0
        self.cur_node = self.tree

    def do_mcts(self, first_state, agent):
        self.state = first_state
        self.agent = agent
        for i in range(200):
            leaf_node, leaf_state, prev_reward = self.select()
            new_leaf_node = self.expand(leaf_node, leaf_state, prev_reward)
            self.backpropagate(new_leaf_node)

    def simulate(self, state):
        total_reward = 0
        steps = 0
        s = state
        while steps < self.agent.handler.hp.MAX_STEPS:
            steps += 1
            prediction = self.agent.brain.predictOne(state)
            a = np.argmax(prediction)
            s_, r, done, info = the_emulator.step(a, flattened=False, state=s)
            if done:
                s_ = None
            s = s_
            total_reward += r
            if done:
                break
        self.n += 1
        return total_reward

    def backpropagate(self, leaf_node):
        cur_node = leaf_node
        while cur_node.has_parent():
            cur_node = cur_node.parent
            max_val = 0
            for child in cur_node.get_child_nodes():
                if child.value > max_val:
                    max_val = child.value
            cur_node.value = max_val
            cur_node.play_count += 1

    def expand(self, node, state, prev_reward):
        prediction = self.agent.brain.predictOne(state)
        child_c = node.get_child_count()
        for i in range(child_c):
            del prediction[np.argmax(prediction)]
        a = np.argmax(prediction)
        s_, r, d, _ = the_emulator.step(a, flattened=False, state=state)
        rew = self.simulate(s_)
        return node.add_child_node(action=a, value=rew + prev_reward, node_reward=r, is_end_node=d)

    def select(self):
        prev_reward = 0
        cur_node = self.tree
        s = self.state
        while cur_node.has_all_childs():
            cur_node = cur_node.get_best_child_according_to_ucb(self)
            s_, r, d, _ = the_emulator.step(cur_node.action, flattened=False, state=s)
            s = s_
            if cur_node.is_leaf:
                self.n += 1
                break
            prev_reward += cur_node.node_reward
        return cur_node, s, prev_reward

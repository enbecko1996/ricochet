import game as emulator
import math
import numpy as np


the_emulator = emulator.Environment()


class Node:
    def __init__(self, parent, layer, action, value=1, node_reward=0, is_end_node=False):
        self.parent = parent
        self.value = value
        self.action = action
        self.node_reward = node_reward
        self.is_end_node = is_end_node
        self.play_count = 1
        self.child_nodes = []
        self.layer = layer

    def get_child_nodes(self):
        return self.child_nodes

    def add_child_node(self, action, value, node_reward, is_end_node):
        new_node = Node(self, self.layer + 1, action, value=value, node_reward=node_reward, is_end_node=is_end_node)
        self.child_nodes.append(new_node)
        return new_node

    def get_best_child_according_to_ucb(self, mcts_obj):
        if len(self.child_nodes) > 0:
            max_node = self.child_nodes[0]
            max_val = max_node.value
            for child in self.child_nodes:
                if child.is_end_node:
                    return child
                my_val = child.value + math.sqrt((2 * math.log(mcts_obj.n)) / child.play_count)
                # print(mcts_obj.n, my_val, child)
                if my_val > max_val:
                    max_val = my_val
                    max_node = child
            return max_node

    def get_most_played_child(self):
        if len(self.child_nodes) > 0:
            most_node = self.child_nodes[0]
            most_plays = most_node.play_count
            for child in self.child_nodes:
                if child.is_end_node:
                    return child
                my_play_count = child.play_count
                if my_play_count > most_plays:
                    most_plays = my_play_count
                    most_node = child
            return most_node

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

    def __str__(self):
        return f"layer: {str(self.layer)}, action: {emulator.print_action(self.action)}, value: {str(self.value)}, " \
               f"plays: {str(self.play_count)}"


class MCTS:

    def __init__(self):
        self.tree = Node(None, 1, None)
        self.state = None
        self.agent = None
        self.n = 0
        self.cur_node = self.tree

    def do_mcts(self, first_state, agent):
        self.state = first_state
        self.agent = agent
        for i in range(20000):
            if (i + 1) % 20 == 0:
                print((i * 100) / 20000, "% mcts done")
            leaf_node, leaf_state, prev_reward = self.select()
            if leaf_node.is_end_node:
                new_leaf_node = leaf_node
            else:
                new_leaf_node = self.expand(leaf_node, leaf_state, prev_reward)
            self.backpropagate(new_leaf_node)
        cur_node = self.tree
        actions = []
        total_rew = 0
        while cur_node.has_childs():
            cur_node = cur_node.get_most_played_child()
            print(cur_node.action, cur_node.value, cur_node.node_reward)
            actions.append(cur_node.action)
            total_rew += cur_node.node_reward
            if cur_node.is_end_node:
                return total_rew, actions
        return total_rew, actions

    def simulate(self, state):
        total_reward = 0
        steps = 0
        s = state
        # print("new sim")
        while steps < self.agent.handler.hp.MAX_STEPS:
            steps += 1
            prediction = self.agent.brain.predictOne(the_emulator.get_flattened_reduced_state(s))
            a = np.argmax(prediction)
            # print(emulator.print_action(a))
            s_, r, done, info = the_emulator.step(a, flattened=False, state=s, hyperparams=self.agent.handler.hp)
            s = s_
            total_reward += r
            if done:
                break
        self.n += 1
        return total_reward

    def backpropagate(self, leaf_node):
        cur_node = leaf_node
        while cur_node.has_parent():
            cur_node.play_count += 1
            cur_node = cur_node.parent
            childs = cur_node.get_child_nodes()
            cumulative = 0
            for child in childs:
                cumulative += child.value
            cur_node.value = cumulative / cur_node.get_child_count()

    def expand(self, node, state, prev_reward):
        prediction = self.agent.brain.predictOne(the_emulator.get_flattened_reduced_state(state))
        child_c = node.get_child_count()
        for i in range(child_c):
            prediction[np.argmax(prediction)] = -999999
        a = np.argmax(prediction)
        s_, r, d, _ = the_emulator.step(a, flattened=False, state=state, hyperparams=self.agent.handler.hp)
        if not d:
            rew = self.simulate(s_)
        else:
            rew = r
        return node.add_child_node(action=a, value=rew + prev_reward, node_reward=r, is_end_node=d)

    def select(self):
        prev_reward = 0
        cur_node = self.tree
        s = np.array(self.state)
        while cur_node.has_all_childs():
            cur_node = cur_node.get_best_child_according_to_ucb(self)
            prev_reward += cur_node.node_reward
            s_, r, d, _ = the_emulator.step(cur_node.action, flattened=False, state=s, hyperparams=self.agent.handler.hp)
            s = s_
            if cur_node.is_end_node:
                self.n += 1
                break
        return cur_node, s, prev_reward

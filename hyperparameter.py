# environment
goal_reached_reward = 10.
step_reward = -1.
in_wall_reward = -10.

# neural_net
epochs = 4000
lr = 0.03
gamma = 0.9
e_start = 1
e_min = 0.05
e_decay = 1./(epochs - epochs/6.)

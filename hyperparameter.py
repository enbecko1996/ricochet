# environment
goal_reached_reward = 10.
step_reward = -1.
in_wall_reward = -1.

# neural_net
EPOCHS = 20000
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025

MEMORY_CAPACITY = 100000
BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.00002  # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

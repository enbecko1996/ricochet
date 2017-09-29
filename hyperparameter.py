class hyperparams:
    def __init__(self):
        # environment
        self.goal_reached_reward = 150.
        self.step_reward = -1.
        self.in_wall_reward = -1.

        # neural_net
        self.EPOCHS = 75000
        self.HUBER_LOSS_DELTA = 1.0
        self.LEARNING_RATE = 0.00025

        self.MEMORY_CAPACITY = 100000
        self.BATCH_SIZE = 32

        self.GAMMA = 0.99

        self.MAX_EPSILON = 1
        self.MIN_EPSILON = 0.01
        self.LAMBDA = 0.000002  # speed of decay

        self.UPDATE_TARGET_FREQUENCY = 1000
        pass

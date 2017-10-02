class hyperparams:
    def __init__(self):
        # environment
        self.goal_reached_reward = 16.
        self.step_reward = -1.
        self.in_wall_reward = -5.

        # neural_net
        self.EPOCHS = 75000
        self.LEARNING_RATE = 0.01

        self.MEMORY_CAPACITY = 100000
        self.REPLAY = 2
        self.BATCH_SIZE = 64

        self.GAMMA = 0.99

        self.MAX_EPSILON = 1
        self.MIN_EPSILON = 0.01
        self.LAMBDA = 0.0000016  # speed of decay

        self.UPDATE_TARGET_FREQUENCY = 1000

        self.MINIMUM_CAPTURE_THRESH = 300
        pass

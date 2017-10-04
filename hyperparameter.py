class hyperparams:
    def __init__(self):
        # neural_net
        self.MAX_STEPS = 300
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

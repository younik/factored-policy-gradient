class RandomPolicy:

    def __init__(self, action_space) -> None:
        self.action_space = action_space
    

    def __call__(self, _):
        return self.action_space.sample()
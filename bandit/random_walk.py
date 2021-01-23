from bandit import Normal


class RandomWalk(Normal):
    """
    A random walk bandit.

    This class features k arms with rewards from the arms drawn from normal
    distributions. The means, when initialized, are drawn from a uniform range
    of [-1, 1). However, after each call to select, the means for every arm
    is changed. Each arm's mean is adjusted by a randomly selected value drawn
    from a normal distribution with mean 0 and standard deviation 0.01. These
    values are drawn independently for each arm.
    """

    def __init__(self, k: int) -> None:
        super().__init__(k)

    def select(self, index):
        rewards = super().select(index)
        # Now modify the means.
        return rewards

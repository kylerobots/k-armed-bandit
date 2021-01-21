from bandit.base import Bandit
import numpy


class Static(Bandit):
    """
    This class implements a bandit with a constant reward value each time
    an arm is chosen.

    The reward is not drawn from a distribution, nor does it change over time.
    The user can specify the reward values at instantiation if they want.
    """

    def __init__(self, k, rewards=None):
        """
        Instantiate the class.

        If reward_values is provided, it will be used for the reward values.
        Otherwise, a random value on the interval [0, 1) will be chosen for
        each arm.
        @param k The number of arms to create.
        @param reward_values If provided, the fixed reward for each arm. Must
        be an iterable with length k.
        """
        super().__init__(k)
        if rewards is None:
            self._rewards = numpy.random.uniform(low=0, high=1, size=self.k)
        else:
            try:
                assert(len(rewards) == self.k)
                self._rewards = numpy.fromiter(rewards, dtype=numpy.float)
            except:
                raise ValueError(
                    'rewards must be an iterable of length k with numeric values.')

    @property
    def rewards(self):
        return self._rewards

    def select(self, index):
        """
        Get a reward from the chosen arm.
        @param index The arm to pick. Must be an integer between 0 and k-1 (inclusive)
        @return The reward for that arm.
        @raise ValueError Raised if the provided index is not a valid arm.
        """
        if index < 0 or index >= self.k:
            raise ValueError(
                'Selected arm must be on the range [0, {0:d})'.format(self.k))
        return self.rewards[index]

    def trueValues(self):
        """
        Provide a list of the rewards for each arm.
        @return A numpy array where each index corresponds to the reward value
        for the associated arm at that index.
        """
        return self.rewards

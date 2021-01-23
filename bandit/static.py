from bandit import BaseBandit
import numpy


class Static(BaseBandit):
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
        @param k An int greater than or equal to one representing the number
        of arms this bandit has.
        @param reward_values If provided, the fixed reward for each arm. It can
        be a list, array, numpy array, or any sort of iterable object, but must
        have a length equal to k. It can also be None to let the bandit pick
        random rewards from the interval [0, 1).
        """
        super().__init__(k)
        if rewards is None:
            self._rewards = numpy.random.uniform(low=0, high=1, size=self.k)
        else:
            if len(rewards) != self.k:
                raise ValueError(
                    'rewards_value must have a length of {0}, not {1}'.format(
                        self.k, len(rewards)))
            self._rewards = numpy.fromiter(rewards, dtype=numpy.float)

    @property
    def rewards(self):
        return self._rewards

    def select(self, index):
        """
        Get a reward from the chosen arm.
        @param index The arm to pick. It can be any input that allows for
        indexing of a numpy array, including single integers or a set of
        integers.
        @return The reward for that arm. The type will be either a single
        float if a single arm was chosen or an array of floats representing
        the rewards for each arm identified by index. If None is passed in
        to index, this will return None.
        """
        # Numpy arrays allow use of None, which serves as newaxis. This
        # behavior should be guarded against since this method should not
        # be manipulating the rewards array (or its copies), only providing
        # values from it. If a None occurs, just pass it back along.
        if index is None:
            return None
        else:
            return self.rewards[index]

    def trueValues(self):
        """
        Provide a numpy array of the rewards for each arm.
        @return A numpy array where each index corresponds to the reward value
        for the associated arm at that index.
        """
        return self.rewards

from bandit import BaseBandit
import numpy


class Normal(BaseBandit):
    """
    This bandit draws a reward from a set normal distribution each time
    an arm is chosen. Each arm has its own distribution that is fixed upon
    construction. Each distribution has a standard deviation of 1 and a mean
    randomly drawn from the uniform range [-1, 1).
    """

    def __init__(self, k: int) -> None:
        """
        Construct the class.

        This includes defining the normal distribution parameters for each
        arm. There is a different distribution for each arm. The means are
        sampled from the uniform range [-1, 1). The standard deviations are
        1.0.
        @param k The number of arms this bandit should have. This must be an
        int greater than 0.
        """
        super().__init__(k)
        # The standard deviations are fixed.
        self._std = numpy.ones(shape=(k,), dtype=numpy.float)
        # The means are drawn from a uniform range.
        self._mean = numpy.random.uniform(low=-1.0, high=1.0, size=(k,))

    def select(self, index):
        """
        Select one or several arms to obtain a reward from.

        @param index Any numpy valid indexing method to select which arms
        a reward should be drawn from. None can also be passed, but will only
        return a reward of None.
        @return The rewards. The size of this will depend on the type of index.
        If a single integer is passed in, a single float will be returned.
        Otherwise, a numpy array will be returned. If None is passed in, this
        will also be None.
        """
        if index is None:
            return None
        means = self._mean[index]
        stds = self._std[index]
        return numpy.random.normal(loc=means, scale=stds)

    def trueValues(self):
        """
        Return the distribution parameters for the arms.

        @return A tuple containing the parameters for each arm's distribution.
        The first element of the tuple will be a numpy array holding the means
        for each arm. The second element will also be a numpy array with the
        standard deviations.
        """
        return (self._mean, self._std)

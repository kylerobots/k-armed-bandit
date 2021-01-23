from bandit import RandomWalk
import numpy
import unittest


class TestRandomWalkBandit(unittest.TestCase):
    """
    Tests the implementation of a bandit with random walk on the distributions.

    As this class inherits from Normal, the only difference is testing that the
    distribution means change after each call to select.
    """

    def test_mean_change(self):
        """
        Test that means change over time.

        The class should randomly walk the mean of the distribution of each arm
        after a call to select. So call it a number of times and watch for some
        sort of change. Because it is random, the exact change can't be known,
        so just ensure that the numbers do in fact change.
        """
        K = 100
        bandit = RandomWalk(K)
        (previous_mean, _) = bandit.trueValues()
        values_have_changed = False
        # Call select a number of times. As long as it changes at least once,
        # then the method is working.
        for i in range(100):
            _ = bandit.select(10)
            (mean, _) = bandit.trueValues()
            values_have_changed |= not numpy.array_equal(previous_mean, mean)
            previous_mean = mean
        self.assertTrue(values_have_changed)

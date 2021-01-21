from numpy.lib.function_base import kaiser
from bandit.base import Bandit
import unittest
from bandit.static import Static
import numpy


class TestStaticBandit(unittest.TestCase):
    """
    Test case to test that the static bandit works correctly.
    """

    def test_instantiateK(self):
        """
        Test that the Static bandit can handle various values of k
        correctly.
        """
        # These values should all work
        for k in (1, 2, 100):
            bandit = Static(k, None)
            self.assertIsNotNone(bandit)
        # These should all fail
        for k in (0, -1, 0.5, '1', 'the', None):
            # with self.subTest(k=k):
            with self.assertRaises(ValueError):
                Static(k, None)

    def test_instantiateRewards(self):
        """
        Test that the class can handle reward values.

        What values are acceptable somewhat depend on k, so there are several
        tests to check.
        """
        # Test that it requires an iterable of the right length.
        k = 3
        # These are all valid ways of defining the values
        for values in ((1, 2, 3), [1, 2, 3], numpy.array([1, 2, 3]), None):
            bandit = Static(k, values)
            self.assertIsNotNone(bandit)
        # These are all invalid ways
        for values in ((1, 2), ('the', 'it', 4), 4, '1 2 3'):
            with self.assertRaises(ValueError):
                Static(k, values)
        # Test that if none is provided, the resulting assigned values are in
        # the right range. Since this is nondeterministic, check this a few
        # times.
        for i in range(50):
            bandit = Static(k, None)
            values = bandit.trueValues()
            for value in values:
                self.assertGreaterEqual(value, 0)
                self.assertLess(value, 1)

    def test_correctRewards(self):
        """
        Test that the right reward is returned when each arm is selected.
        """
        bandit = Static(3, None)
        rewards = bandit.trueValues()
        # Each reward should match
        for i in range(bandit.k):
            reward = bandit.select(i)
            self.assertEqual(reward, rewards[i])
        # Additionally, incorrect inputs should be rejected.
        for i in (-1, 0, 4, 0.5, '1', None (1, 2, 3)):
            with self.assertRaises(ValueError):
                reward = bandit.select(i)
        pass


if __name__ == '__main__':
    unittest.main()

import numpy
from agent import Greedy
import unittest


class TestGreedy(unittest.TestCase):
    """
    Test the Greedy agent solver.
    """

    def setUp(self) -> None:
        """
        Create an object to help with testing.
        """
        self.agent = Greedy(k=4, start_value=0.0)

    def test_always_exploit(self):
        """
        Test that the algorithm always chooses exploit.

        Since this is greedy, once a value is the best, it should always pick that option.
        """
        # Manually manipulate the table to be sure one value is better than the others.
        ACTUAL_BEST = 2
        self.agent._table[ACTUAL_BEST] = 100.0
        # Test a number of times to be sure
        for _ in range(100):
            expected_best = self.agent.act()
            self.assertEqual(ACTUAL_BEST, expected_best)

    def test_update(self):
        """
        Test that the update works correctly.

        The agent uses a weighted average, so use known values to ensure correct calculation.
        """
        # Working out the formula by hand produces the following values.
        rewards = numpy.array(range(15, 26))
        expected_results = numpy.array(
            [15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20])
        for i in range(expected_results.size):
            # Apply the reward first, then check that the table updated correctly.
            self.agent.update(action=0, reward=rewards[i])
            self.assertEqual(self.agent.table[0], expected_results[i])

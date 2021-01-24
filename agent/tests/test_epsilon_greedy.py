from agent import EpsilonGreedy
import numpy
import unittest


class TestEpsilonGreedy(unittest.TestCase):
    """
    Test case to verify behavior of the epsilon greedy agent implementation.
    """

    def setUp(self) -> None:
        """
        Create an agent to use for tests.
        """
        self.agent = EpsilonGreedy(k=10, epsilon=0.5, start_value=0.0)

    def test_action_selection(self):
        """
        Test appropriate actions are selected.

        As the agent can select via explore or exploit, no specific action is assumed. So this just tests that all
        selected actions are within the correct range of [0, k).
        """
        # Repeat multiple times.
        for _ in range(100):
            action = self.agent.act()
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.agent.table.size)

    def test_epsilon_bounds(self):
        """
        Test that epsilon only accepts probability values.
        """
        # These values are all valid probabilities.
        for e in (0.0, 0.25, 0.12345, 1.0, 1, 0):
            agent = EpsilonGreedy(10, e, 0.0)
            self.assertEqual(e, agent.epsilon)
        # These are not allowed.
        for e in (-0.01, 100, 1.1, '0.5', (0.0, 0.5, 0.25)):
            with self.assertRaises(Exception, 'Agent did not reject invalid epsilon inputs.'):
                agent = EpsilonGreedy(3, e, 0.0)  # type: ignore

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

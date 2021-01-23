from agent import BaseAgent
import unittest


class FakeAgent(BaseAgent):
    """
    A fake child class to allow testing of BaseAgent.

    BaseAgent is an abstract class, so can't be instantiated directly. This FakeChild class implements the bare minimum
    to allow testing of the elements of the base class that can be tested.
    """

    def __init__(self, k: int, start_value: float = 0.0) -> None:
        super().__init__(k, start_value=start_value)

    def act(self) -> int:
        return 0

    def update(self, action: int, reward: float) -> None:
        pass


class TestBaseAgent(unittest.TestCase):
    """
    Test the BaseAgent class.

    This tests the creating of the Q-table, as well as the exploration/exploitation helper methods.
    """

    def test_q_table_creation(self):
        """
        Verify that the table initializes with the right values.
        """
        # Positive integers are valid and should produce a Q-table of the same size.
        for k in (1, 10, 100):
            agent = FakeAgent(k=k, start_value=0.0)
            self.assertEqual(agent.table.size, k,
                             msg='BaseAgent did not make a correct table size.')
        # Anything else should fail.
        for k in (0, -1, 0.5, '1', 'the', None):
            with self.assertRaises(Exception, msg='Static bandit did not reject invalid k input.'):
                FakeAgent(k)  # type:ignore

    def test_exploitation(self):
        """
        Test that the class picks the best action based on the table.

        This tests that the class will reliable return the index associated with the best action. It should also
        arbitrarily break ties in the case of multiple entries with an equivalent score.
        """
        agent = FakeAgent(k=4, start_value=0.0)
        # Set one clearly better than the others to make sure it is returned.
        ACTUAL_BEST = 2
        BEST_REWARD = 100.0
        agent._table[ACTUAL_BEST] = BEST_REWARD
        expected_best = agent.exploit()
        self.assertEqual(ACTUAL_BEST, expected_best,
                         'Exploitation picked an incorrect index.')
        # Set another equal to force a tie.
        agent._table[ACTUAL_BEST + 1] = BEST_REWARD
        # Sample several times to make sure it never picks anything else.
        for _ in range(100):
            expected_best = agent.exploit()
            result = (expected_best == ACTUAL_BEST) or (
                expected_best == ACTUAL_BEST + 1)
            self.assertTrue(
                result, msg='Exploitation picked an incorrect index when breaking a tie.')

    def test_exploration(self):
        """
        Test that the class picks a random valid action from the table.
        """
        K = 4
        agent = FakeAgent(k=K, start_value=0.0)
        # Sample several times and make sure the result is always a valid index
        possible_actions = list(range(K))
        for _ in range(100):
            action = agent.explore()
            self.assertTrue(action in possible_actions,
                            msg='Exploration produced an invalid index.')

from bandit import Normal
import unittest


class TestNormalBandit(unittest.TestCase):
    """
    Tests the methods for a normal distribution bandit.
    """

    def test_distribution_generation(self):
        """
        Ensure that arm distributions have means on the range [-1, 1) and
        standard deviations of 1.
        """
        # As these are randomly selected, do this several times to
        # somewhat increase confidence.
        for i in range(100):
            K = 100
            bandit = Normal(k=K)
            (mean, std) = bandit.trueValues()
            # All means should be between [-1, 1), so use the all() to alert if
            # there are any that aren't.
            self.assertTrue((mean >= -1.0).all(),
                            msg='Selected means have a value below -1.0.')
            self.assertTrue((mean < 1.0).all(),
                            msg='Selected means have a value above 1.0.')
            # Standard deviations are fixed at 1.0.
            self.assertTrue((std == 1.0).all(),
                            msg='Standard deviations are not all 1.0.')

    def test_reward_selection(self):
        """
        Test that rewards of the correct shape are produced.

        Since any numeric value is technically possible, only check that the
        right number of rewards are returned.
        """
        K = 10
        bandit = Normal(K)
        # Integers should return single values, as long as they are within
        # range
        for arm in (-3, 0, 2, 9):
            reward = bandit.select(arm)
            self.assertTrue(isinstance(reward, float))
        # Ranges should return the same number of elements.
        reward = bandit.select(range(K))
        self.assertEqual(len(reward), K)
        # None should return None.
        self.assertIsNone(bandit.select(None))
        # Other values should produce an error.
        for i in (0.5, K, '1', 'the'):
            with self.subTest(i=i):
                with self.assertRaises(Exception, msg='Incorrect indices not rejected.'):
                    reward = bandit.select(i)

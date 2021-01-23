from bandit import Static
import numpy
import unittest


class TestStaticBandit(unittest.TestCase):
    """
    Test case to test that the static bandit works correctly.
    """

    def test_instantiate_rewards(self):
        """
        Test that the class can handle reward values. Acceptable values should
        be some sort of iterable with a number of elements equal to k. Each
        element should be numeric. Alternatively, None can be used to have the
        class randomly select values.
        """
        k = 4
        # Iterables can include lists, arrays, or numpy arrays, for starters.
        for values in ((1, 2, 3, 4), [1, 2, 3, 4], numpy.array([1, 2, 3, 4])):
            with self.subTest(values=values):
                bandit = Static(k, values)
                #  Make sure that the stored rewards are what was provided.
                rewards = bandit.trueValues()
                for i, value in enumerate(values):
                    self.assertEqual(
                        value, rewards[i], 'Stored reward does not match provided.')
                self.assertIsNotNone(bandit)
        # The user can pass in None to randomly select rewards between [0, 1).
        # Since this is technically random, try several times.
        for i in range(50):
            bandit = Static(k, None)
            values = bandit.trueValues()
            for value in values:
                self.assertGreaterEqual(value, 0)
                self.assertLess(value, 1)
        # An iterable must have the same length as k, otherwise it should fail.
        with self.assertRaises(ValueError, msg='Static bandit did not reject incorrect length rewards.'):
            Static(k=3, rewards=(1, 2))
        # Other, non-numeric iterables and data types should produce some form
        # of error. This includes multi-dimensional numpy arrays.
        for values in ((1, 2), ('the', 'it', 4, 5), 4,
                       '1 2 3 4', numpy.array([[1, 2], [3, 4]])):
            with self.assertRaises(Exception, msg='Static bandit did not reject non-numeric rewards'):
                Static(k, values)

    def test_correct_rewards(self):
        """
        Test that the right reward is returned when each arm is selected. This
        should allow any indexing inputs that you could use for numpy arrays
        and reject everything else. The correct inputs should return the
        appropriate rewards that match the reward values set by the class.
        """
        k = 10
        true_rewards = numpy.random.uniform(low=0, high=1, size=10)
        bandit = Static(k, true_rewards)
        # This method can accept a single integer.
        for arm in (-3, 0, 2, 9):
            expected_reward = bandit.select(arm)
            self.assertEqual(
                expected_reward,
                true_rewards[arm],
                'Static bandit did not provide a correct reward.')
        # It can also accept an iterable type.
        arms = range(k)
        expected_rewards = bandit.select(arms)
        self.assertTrue(numpy.array_equal(
            true_rewards[arms], expected_rewards))
        # Additionally, incorrect inputs should be rejected, including indices
        # out of range.
        for i in (0.5, 10, '1', 'the'):
            rewards = bandit.trueValues()
            with self.subTest(i=i):
                with self.assertRaises(Exception, msg='Incorrect indices not rejected.'):
                    reward = bandit.select(i)
        # None is a special case and will return None
        self.assertIsNone(bandit.select(None))


if __name__ == '__main__':
    unittest.main()

from bandit import BaseBandit
import unittest


class FakeBandit(BaseBandit):
    """
    This is a fake child class.

    Since Bandit is an abstract class, it can't be instantiated directly. So
    it's methods can't be called, and therefore tested. So create a fake class
    that defines the required methods in the simplist terms and use that
    instead.
    """

    def __init__(self, k):
        super().__init__(k)

    def select(self, index):
        pass

    def trueValues(self):
        pass


class TestBaseBandit(unittest.TestCase):
    """
    Tests the base class that all bandits rely on. Since this is an abstract
    class, it uses a simple inheriting class to allow testing of the
    defined functionality.
    """

    def test_instantiate_k(self):
        """
        Test that the bandit properly sets the number of arms. Any positive
        integer is permitted and everything else is rejected. This mimics real
        life where you can't have a non-natural number of arms. Zero is
        explicitly excluded, since it makes the object trivial.
        """
        # These values should all work
        for k in (1, 2, 100):
            bandit = FakeBandit(k)
            self.assertEqual(
                bandit.k, k, msg='Static bandit did not create the correct number of arms.')
        # These should all fail
        for k in (0, -1, 0.5, '1', 'the', None):
            with self.assertRaises(ValueError, msg='Static bandit did not reject invalid k input.'):
                FakeBandit(k)

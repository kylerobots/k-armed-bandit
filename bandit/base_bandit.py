import abc
""" @package docstring
This package provides several k-Armed Bandit implementations with different
features. It also includes a base class to allow creation of new
implementations.
"""


class BaseBandit(abc.ABC):
    """
    A base class for the various bandit implementations.

    This class defines several abstract methods and properties that must be
    implemented by any k-armed bandit implementation. This ensures consistent
    APIs across all of them.
    """

    def __init__(self, k: int) -> None:
        """
        Initialize the object with a set number of arms.

        @param k The number of arms this bandit should have. This must be an
        integer greater than zero.
        @raise ValueError if k is not an integer greater than zero.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError('k must be an integer greater than 0.')
        self._k = k

    @property
    def k(self) -> int:
        """
        Return the number of arms this bandit has.
        @return An int greater than or equal to one.
        """
        return self._k

    @abc.abstractmethod
    def select(self, index):
        """
        The method to select one of the arms of the bandit.

        When implemented, this method should return the reward obtained when
        selecting the given arm index.
        @param index Some sort of index representation to select which arms to
        get rewards from. Typically, this will be a single integer or some
        sort of list, array, etc. of integers.
        """
        raise NotImplementedError("Subclass does not implement select method.")

    @abc.abstractmethod
    def trueValues(self):
        """
        Return the true reward values of the bandit.

        When implemented, this should provide the user with the complete truth
        of the bandit's state at the moment called. It is up to the
        implementation what exact information this is.
        """
        raise NotImplementedError(
            'Subclass does not implement trueValues method.')

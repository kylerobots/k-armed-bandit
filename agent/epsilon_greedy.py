import numpy
from agent import BaseAgent


class EpsilonGreedy(BaseAgent):
    """
    A greedy agent that occasionally explores.

    This agent will primarily exploit when deciding its actions. However, it will occasionally choose to explore at a
    rate of epsilon, which is provided at initialization. This gives it a chance to see if other actions are better
    options.
    """

    def __init__(self, k: int, epsilon: float, start_value: float = 0.0) -> None:
        """
        Construct the agent.

        @param k The number of actions to consider. This must be an int greater than zero.
        @param epsilon The rate at which actions should randomly explore. As this is a probability, it should be between
        0 and 1.
        @param start_value The initial value to use in the table. All actions start with the same value.
        @raise ValueError if epsilon is not a valid probability (between 0 and 1).
        """
        super().__init__(k, start_value=start_value)
        self.epsilon = epsilon
        # Track how many selections have been made to use in the update formula.
        self._n = 0
        # Per Numpy documentation, this is the preferred way to sample from random distributions.
        self._rng = numpy.random.default_rng()

    def act(self) -> int:
        """
        Determine which action to take.

        This will explore randomly over the actions at a rate of epsilon and inversely will exploit based on table
        values at a rate of (1.0 - epsilon).
        @return The index of the selected action to take. Gauranteed to be an int on the range [0, k).
        """
        # Decide if the agent should explore or exploit using epsilon
        samples = self._rng.binomial(n=1, p=self.epsilon, size=1)
        should_explore = (samples[0] == 1)
        if should_explore:
            action = self.explore()
        else:
            action = self.exploit()
        return action

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError(
                'Epsilon must be a valid probability, so between 0 and 1 (inclusive)!')
        self._epsilon = value

    def update(self, action: int, reward: float) -> None:
        """
        Update the Q-table based on the last action.

        This will use an incremental formulation of the mean of all rewards obtained so far as the values of the table.
        @param action An index representing which action on the table was selected. It must be between [0, k).
        @param reward The reward obtained from this action.
        """
        self._n += 1
        self.table[action] += (reward - self.table[action]) / self._n

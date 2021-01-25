import abc
import numpy


class BaseAgent(abc.ABC):
    """
    A base class used to create a variety of bandit solving agents.

    This class provides a table that can be used to store reward estimates. It also defines the interface that any
    agent must define when implemented. This ensures consistent API across each agent type.
    """

    def __init__(self, k: int, start_value: float = 0.0) -> None:
        """
        Construct the agent.

        @param k The number of possible actions the agent can pick from at any given time. Must be an int greater than
        zero.
        @param start_value An initial value to use for each possible action. This assumes that each action is equally
        likely at start, so all values in the Q-table are set to this value.
        @exception ValueError if k is not an integer greater than 0.
        """
        super().__init__()
        # Create a Q-table with size k.
        if k <= 0:
            raise ValueError('k must be an integer greater than zero.')
        self._table = start_value * numpy.ones(shape=(k,), dtype=numpy.float)

    @abc.abstractmethod
    def act(self) -> int:
        """
        Use a specific algorithm to determine which action to take.

        This method should define how exactly the agent selects an action. It is free to use @ref explore and @ref
        exploit as needed.
        @return An int representing which arm action to take. This int should be between [0, k).
        """

    def exploit(self) -> int:
        """
        Select the best action.

        This will use the Q-table to select the action with the highest likelihood. Ties are broken arbitrarily.
        @return An int representing which arm action to take. This int will be between [0, k).
        """
        # Argmax will return all the highest arguments as an array. If there is only a single highest, argmax will
        # just return a single value. Otherwise, use choice to pick amongst all the indices.
        possible_actions = numpy.argmax(a=self.table)
        if possible_actions.size == 1:
            selected_action = possible_actions
        else:
            # Default of choice is to pick a single value
            selected_action = numpy.random.choice(a=possible_actions)
        return selected_action

    def explore(self) -> int:
        """
        Explore a new action.

        This will select a random action to take from the Q-table, to explore the decision space more.
        @return An int representing which arm action to take. This int will be between [0, k).
        """
        # Just pick a random action. When provided a single value, choice chooses from the range [0, k), so use the
        # size of the table as the input to choice.
        return numpy.random.choice(a=self.table.size, size=1)

    @property
    def table(self) -> numpy.ndarray:
        """
        Return the Q-Table.
        @return a Numpy array of k elements. the i-th element holds the estimated value for the i-th action/arm.
        """
        return self._table

    @abc.abstractmethod
    def update(self, action: int, reward: float) -> None:
        """
        Update the Q-Table.

        This takes the result of the previous action and the resulting reward and should update the Q-Table. How it
        updates will depend on the specific implementation.
        @param action An int representing which arm action was taken. This should be between [0, k].
        @param reward A float representing the resulting reward obtained from the selected action.
        """

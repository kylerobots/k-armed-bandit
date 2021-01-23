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
        """
        super().__init__()

    def explore(self) -> int:
        """
        Explore a new action.

        This will select a random action to take from the Q-table, to explore the decision space more.
        @return An int representing which arm action to take. This int will be between [0, k).
        """

    def exploit(self) -> int:
        """
        Select the best action.

        This will use the Q-table to select the action with the highest likelihood. Ties are broken arbitrarily.
        @return An int representing which arm action to take. This int will be between [0, k).
        """

    @abc.abstractmethod
    def act(self) -> int:
        """
        Use a specific algorithm to determine which action to take.

        This method should define how exactly the agent selects an action. It is free to use @ref explore and @ref
        exploit as needed.
        @return An int representing which arm action to take. This int should be between [0, k).
        """

    @abc.abstractmethod
    def update(self, action: int, reward: float) -> None:
        """
        Update the Q-Table.

        This takes the result of the previous action and the resulting reward and should update the Q-Table. How it
        updates will depend on the specific implementation.
        @param action An int representing which arm action was taken. This should be between [0, k].
        @param reward A float representing the resulting reward obtained from the selected action.
        """

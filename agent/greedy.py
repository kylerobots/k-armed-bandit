from agent import BaseAgent


class Greedy(BaseAgent):
    """
    An agent that always exploits, never explores.

    It will always pick the action with the highest value from the Q-table. While these values will be updated, it
    never explores, so will likely quickly converge on a single action.
    """

    def __init__(self, k: int, start_value: float = 0.0) -> None:
        """
        Construct the agent.

        @param k The number of arms to select from. Should be an int greater than zero.
        @param start_value The starting reward to use for each arm. All arms assume the same value at the start.
        """
        super().__init__(k, start_value=start_value)

    def act(self) -> int:
        """
        Select an action to take from the available ones.
        """
        pass

    def update(self, action: int, reward: float) -> None:
        """
        Update the table values based on the last action.

        This uses an iterative version of a running average to update table values.
        @param action The index corresponding to the action that was taken.
        @param reward The resulting reward that was earned.
        """
        pass

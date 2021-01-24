import agent
import bandit
import numpy
import matplotlib.pyplot
"""
Compete various agents against each other and display the results.

This script analyzes the performance of different agents. In general, it will simulate them on a given bandit M times,
then repeat this action on N different bandits. The rewards obtained are tracked over the whole simulation. Afterwards,
some statistics are calculated and plotted for consideration.

The main statistic under consideration is the total reward earned by each agent. A better agent should have better
performance in the long run. This is tracked at each time step and plotted to show how each agent performs over time.
"""
# Set the simulation parameters.
# How many arms each bandit has
K = 10
# How many bandits to test on.
N = 2000
# How many times to select an arm on the bandit.
M = 1000

# Create the bandit and agents. Use several different epsilon values.
test_bandits = []
for i in range(N):
    test_bandit = bandit.Normal(k=K)
    test_bandits.append(test_bandit)
test_agents = [
    agent.Greedy(k=K),
    agent.EpsilonGreedy(k=K, epsilon=0.01),
    agent.EpsilonGreedy(k=K, epsilon=0.1)
]

rewards = numpy.zeros(shape=(M,))
test_bandit = bandit.Normal(k=K)
test_agent = agent.EpsilonGreedy(k=K, epsilon=0.01)
cumulative_mean_reward = 0.0
for m in range(M):
    action = test_agent.act()
    reward = test_bandit.select(index=action)
    test_agent.update(action=action, reward=reward)  # type: ignore
    # Use a cumulative mean formula
    cumulative_mean_reward += (reward - cumulative_mean_reward) / (m + 1)
    rewards[m] = cumulative_mean_reward

matplotlib.pyplot.plot(rewards)
matplotlib.pyplot.show()

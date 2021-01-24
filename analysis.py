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
bandits = []
for i in range(N):
    single_bandit = bandit.Normal(k=K)
    bandits.append(single_bandit)
agents = [
    agent.Greedy(k=K),
    agent.EpsilonGreedy(k=K, epsilon=0.01),
    agent.EpsilonGreedy(k=K, epsilon=0.1),
]
agent_names = [
    '0.0',
    '0.01',
    '0.1',
]

rewards = numpy.zeros(shape=(len(agents), N, M), dtype=numpy.float)
for i, test_agent in enumerate(agents):
    # Iterate through each sample trial
    for n in range(N):
        # Reset the Q-table. Typically, this is a private property and shouldn't be modified this way, but a reset
        # feature is not available.
        test_agent._table = numpy.zeros_like(a=test_agent.table)
        cumulative_mean_reward = 0.0
        # Select actions the appropriate number of times
        for m in range(M):
            action = test_agent.act()
            reward = bandits[n].select(index=action)
            test_agent.update(action=action, reward=reward)
            cumulative_mean_reward += (reward -
                                       cumulative_mean_reward) / (m + 1)
            rewards[i, n, m] = cumulative_mean_reward

# Once all trials are complete, average across the N bandits to get the average performance for each agent at each
# iteration
mean_rewards = numpy.mean(a=rewards, axis=1)
for i, agent_name in enumerate(agents):
    matplotlib.pyplot.plot(mean_rewards[i])
matplotlib.pyplot.legend(agent_names)
matplotlib.pyplot.show()

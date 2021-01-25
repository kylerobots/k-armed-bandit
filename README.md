# k-Armed-Bandit

See full documentation at https://kylerobots.github.io/k-armed-bandit/

Implementation of the k-Armed Bandit reinforcement learning problem and associated solvers. This is inspired by Chapter
2 of [**Reinforcement Learning: An Introduction**](http://incompleteideas.net/book/the-book.html) by Sutton and Barto.
That chapter offers several variants of the k-armed bandit problem and some different approaches for agents to solve for
it. This code attempts to be a test bed of sorts. It provides many of those bandits as well as a few agents to attempt
to solve. Both bandits and agents inherit from their respective base classes to better promote modularity.

## How to Use ##
If you are just using the existing agents and bandits, see the file @ref analysis.py for an example implementation.
After constructing a bandit and agent, the decision and learning steps are typically as follows.
```python
action = agent.act()
reward = bandit.select(action)
agent.update(action, reward)
```
Not surprisingly, the first step has the agent pick an action via whatever criteria it uses. Then, this action is
provided to the bandit to determine the resulting reward. Lastly, the action and its associated reward are fed to the
agent so it can update its tables. How it updates will depend on the agent. This can proceed as many times as
necessary.

## Adding New Entities ##
Adding a new bandit or agent is straightforward. Both have base classes implemented with abstract methods. When creating
a new class, inherit this base class and implement the methods. This ensures compatibility with the usage instructions
above. Any new classes should also be added to the \_\_init\_\_.py file to include with the module.

### Bandit ###
The base class is called BaseBandit.  There are two abstract methods:

```python
def select(self, index: int) -> float:
```
This is the method used to select one of the arms. It should return a reward using whatever algorithm is used to
decide how the reward is determined. For example, it could return a fixed value or select a random number. This method
can also do other things, such as modify the reward.

```python
def trueValues(self) -> Any:
```
This method should return any information needed to describe the accurate state of the bandit at the moment this method
is called. This may be fixed values, distribution parameters, or whatever else describes its state. This is basically
a complex getter.

### Agent ###
Likewise, the base class for agents is called BaseAgent. There are also two abstract methods to implement:

```python
def act(self) -> int:
```
This method is used to determine which action the agent should take. It should then return the corresponding index
of that action on the bandit. The base class offers two methods, called *explore* and *exploit* that this
method is free to use. Additionally, the base class maintains a Q-table that can be referenced.

```python
def update(self, action: int, reward: float) -> None:
```
This method should perform updates to the Q-table based on the received action and reward. How this updates depends on
the algorithm. This should modify the table property provided by the base class.
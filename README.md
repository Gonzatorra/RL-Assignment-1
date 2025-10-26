# RL-Assignment-1

## Introduction
This assignment focuses on developing an intelligent agent using Reinforcement Learning (RL) so it can take autonomous actions to reach a goal. In this case, the agent is a dinosaur that must reach its cave while moving through the environment. In each grid, there are five types of cells: start point, cave, palm trees, meteorites, and normal cells. Obviously, when reaching a meteorite, the agent dies and the episode finishes. This is key for the agent to learn which actions are best.

Moreover, different grids with varying difficulty levels are used to test the agent’s performance. This is very helpful to analyze how each algorithm works and to demonstrate that, taking into account the environment, the “best” algorithm can change. 

Taking into account all these aspects, the goal of this work is to show how an agent can learn to make the best decisions by interacting with the environment and considering rewards and obstacles.



## Methodology
As it is a tabular task, three main algorithms have been used to solve the problem:
- Q-learning
- Montecarlo
- SARSA
Comparing these algorithms allows us to determine which performs best depending on the grid and the selected hyperparameters.

As mentioned before, different grids (see Annex) were designed to analyze the agent’s behavior. Key elements in the grid include:
- Start point: (0,0)
- Goal/Cave: (0,8)
- Palm trees: Dark green cells
- Meteorites: Blue cells

For the initial experiments, the rewards were set as follows:
- Goal/Cave: +100
- Palm trees: +20
- Meteorites: -100
- Normal cells/Taking steps: -1

The hyperparameters used for all algorithms were:
- Gamma : 0.95
- Epsilon : 1.0
- Epsilon decay: 0.995
- Epsilon min: 0.05
- Max steps per episode: 100

Also, a seed was selected to ensure reproducibility of the results, and each algorithm was trained for 1000 episodes.




## Algorithms
We decided to compare the performance of the algorithms (Q-Learning, Montecarlo, and SARSA) on the different grids. The goal is to determine how each algorithm handles the environment and to see whether the best-performing algorithm changes depending on the obstacles present in the grid.

All these algorithms use an epsilon-greedy policy to select the next action. That’s why the agent chooses a random action with probability epsilon  (exploration) or the action with the highest Q-value with probability 1- (exploitation). Also, the algorithms are created to allow the decay of the epsilon so the agents explore most at the beginning of the training (this will be explained later).

### Q-Learning
At each step, the Q-value of the current state-action is updated using the reward and the maximum Q-value of the next state. This allows the agent to learn the optimal policy without taking into account the actions it actually takes.

### Montecarlo (Every-Visit)
Updates the Q-values by averaging the returns of all visits to each state-action in a complete episode. This approach considers the total accumulated rewards after visiting a state-action pair and is useful for learning from entire episodes.


### SARSA
Unlike Q-Learning, SARSA updates the Q-value using the true next action taken by the agent, meaning the learning is influenced by the current policy being followed.


For all algorithms, the Q-table is initialized to zeros, and the agent interacts with the environment for a fixed number of episodes. At each episode, the agent accumulates rewards, which are also recorded with the number of steps taken.

## Libraries
alembic==1.17.0
cloudpickle==3.1.1
colorlog==6.10.1
contourpy==1.3.3
cycler==0.12.1
Farama-Notifications==0.0.4
fonttools==4.60.1
greenlet==3.2.4
gymnasium==1.2.1
kiwisolver==1.4.9
Mako==1.3.10
MarkupSafe==3.0.3
matplotlib==3.10.7
numpy==2.3.4
optuna==4.5.0
packaging==25.0
pandas==2.3.3
pillow==12.0.0
pygame==2.6.1
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
seaborn==0.13.2
six==1.17.0
SQLAlchemy==2.0.44
tqdm==4.67.1
typing_extensions==4.15.0
tzdata==2025.2

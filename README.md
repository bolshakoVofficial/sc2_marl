# Master's degree project


Using [SMAC](https://github.com/oxwhirl/smac) to access the environment of SC2. Please check their [paper](https://arxiv.org/abs/1902.04043) and [blogpost](http://whirl.cs.ox.ac.uk/blog/smac/).


Content:
* *bin* - default SMAC folder
* *env* - info about maps and environment
* *examples*  - contains all algorithms


Algorithms:
- Q-table based
  - *1m_vs_1zg* - marine vs zergling
  - *2m_vs_2zg_JAL* - 2 vs 2 Joint Action Learning
  - *2m_vs_2zg_JAL_fictious_play* - 2 vs 2 Joint Action Learning with Fictitious Play
  - *2m_vs_2zg_NashQ* - [Hu and Wellman NashQ](http://www.jmlr.org/papers/volume4/hu03a/hu03a.pdf) implementation for StarCraft II
  - *2m_vs_2zg_common_q_table* - agents share common Q-table
  - *2m_vs_2zg_independent* - each agent has own Q-table
  - **
  
- *Deep Learning (tf/keras)*
  - *2m_vs_2zg_DQN* - classic DQN implementation with CNN
  - *2m_vs_2zg_DQN_deltas* - FNN using "numerical" features from env
  - *2m_vs_2zg_NashDQN.py* - each agent handle own FNN + searching for Nash equilibrium
  - *2m_vs_2zg_NashDQN_noNash.py* - own FNN, not searching for Nash equilibrium


## Results after training
![Results](/graphs.png)

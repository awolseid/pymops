# MARLSE4MOPS: Multi-Agent Reinforcement Learning Simulation Environment for Multi-Objective Power Scheduling

### Introduction
Within the framework of MARL, the generating units are represented as multi-RL agents, each with different unit-specific characteristics and multiple conflicting objectives. The framework MARL manifests the form of state $\cal S$, action  $\cal A$, transition (probability) function  $\cal P$ and reward  $\cal R$ for a sequence of discrete timesteps $t$. 
#### The MARL Framework
The scheduling horizon is an hourly divided day, each hour of a day is considered a timestep $t,\forall t$.  Hence, one cycle of determination of commitments and load dispatches for a day represents a complete episode.
- **State Space** $(\cal s_t\in S)$: The state $\cal s_t$ at timestep $t$ consists of the current timestep$t$, a vector of minimum capacities $p_t^{min}$, a vector of maximum capacities $p_t^{max}$, a vector of current (online/offline) duration $tt_t$ ; and the demand $d_t$.
- **Action Space** $(\cal a_t\in A)$: Each of the $n$ agents selects either of the two possible actions (switch-to/stay ON or switch-to/stay OFF), that is, $\cal a_t\in \{0,1\}^n$ at timestep $t$ (or in state $\cal s_t$).
- **Transition Function** $\cal P(s_t' |s_t,a_t)$: Once the agents take actions $\cal a_t\in A$ in $\cal s_t\in S$, there is a transition (or probability) function $\cal P(s_t' |s_t,a_t)$ leading to the next state $\cal s_t'$.
- **Reward function** $(\cal R)$: Based on the state $\cal s_t\in S$, the agents get a reward based on a predefined reinforcement function  $\cal R(s_t,a_t,s_t')$.

The MOPS dynamics can be formally formulated as a 4-tuple $\cal (S,A,P,R)$ Markov Decision Process (MDP).
#####



 

$\frac{2}{3}$

Multi-Agent Reinforcement Learning Simulation Environment for Power Scheduling

This environment is used to simulate mono- to tri-objective power scheduling problems dynamics in the form of Markov Decision Processes (MDPs).
To tackle the dimensionality challenges associated with MOPS, 
- Illegal actions by agents are to be automatically corrected.
- Shortages or excess capacities are adjusted.
The MDPS can then be used to train a custom deep learning model.
The practical viability of the environment is evaluated on different test systems featuring mono- to tri-objective problems.

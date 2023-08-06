# MARLSE4MOPS: Multi-Agent Reinforcement Learning Simulation Environment for Multi-Objective Power Scheduling

### Introduction
Within the framework of MARL, the generating units are represented as multi-RL agents, each with different unit-specific characteristics and multiple conflicting objectives. The framework MARL manifests the form of state $\cal S$, action  $\cal A$, transition (probability) function  $\cal P$ and reward  $\cal R$ for a sequence of discrete timesteps $t$. 
#### The MARL Framework Components
The scheduling horizon is an hourly divided day, each hour of a day is considered a timestep $t,\forall t$.  Hence, one cycle of determination of commitments and load dispatches for a day represents a complete episode. 
- bf{State Space} $(\cal S)$: The state $\cal s_t$ at timestep $t$ is defined as $\cal s_t=(t,p_t^{min},p_t^{max},t_t,d_t)$ where $t$ is the current timestep, $p_t^{min}$ is a vector of minimum capacities, $p_t^{max}$ is a vector of maximum capacities, $t_t$ is a vector of current (online/offline) duration; and $d_t$ is the demand.
- \textbf{Action Space} ($\cal A$): Each of the $n$ agents have two possible actions (switch-to/stay ON or switch-to/stay OFF), that is, $\cal a_{it}\in {0,1}$ at timestep $t$ (or in state $\cal s_t$). This implies, there are a total of $2^n$ commitments in the action space $\cal A$.
- \textbf{Transition Function} {$\cal P(s_t^' |s_t,a_t)$}: Once the agents take actions $\cal a_t\in A$ in the current state $\cal s_t\in S$, there is a transition (or probability) function $\cal P(s_t^' |s_t,a_t)$ leading to the next state $\cal s_t^'$. The transition function must satisfy all the constraints from Eqs. (9)-(13) at each timestep $t,\forall t \in T$. If any of the constraints are violated, it would not be legitimate to advance to the succeeding state $\cal s_t^'=s_(t+1)$. The legality and viability of each agent action and the joint actions must be verified before moving on to the next state $s_(t+1)$. In this study, instead, an auto-corrective simulation environment is built to make context-based adjustments and corrections on any illegal agents’ decisions or infeasible combined actions that are against the state. For instance, the small capacity of the agents will be adjusted by switching each of the unconstrained OFF agents back to ON in an increasing order of the priority list described in Step 4.2 of Section 3.2. Similarly, the same applies to adjusting for excess capacity.
- \textbf{Reward function} ($\cal R}): Based on the state $\cal s_t\in S$, the agents get a reward based on a predefined reinforcement function  $\cal R(s_t,a_t,s_t^')$. The reward in this study is defined as the negative of the normalized operation (production, startup, and shutdown) values of each objective function. 
	Hence, the MOPS dynamics can be formally formulated as a 4-tuple $\cal (S,A,P,R)$ Markov Decision Process (MDP) since the framework of MARL satisfies the properties of MDP (Sutton & Barto, 2018). Therefore, the dynamics of the MOPS can be simulated in the form of MDPs in the custom agent-based simulation environment, whose main components are described below.
#####



 

$\frac{2}{3}$

Multi-Agent Reinforcement Learning Simulation Environment for Power Scheduling

This environment is used to simulate mono- to tri-objective power scheduling problems dynamics in the form of Markov Decision Processes (MDPs).
To tackle the dimensionality challenges associated with MOPS, 
- Illegal actions by agents are to be automatically corrected.
- Shortages or excess capacities are adjusted.
The MDPS can then be used to train a custom deep learning model.
The practical viability of the environment is evaluated on different test systems featuring mono- to tri-objective problems.

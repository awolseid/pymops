# MARLSE4MOPS: Multi-Agent Reinforcement Learning Simulation Environment for Multi-Objective Power Scheduling

## Project Description
### Objective 
Cost function: $$\cal C_{it}=z_{it}f^c(p_{it})+z_{it}(1-z_{i,t-1})s_{it}^{ON,c}+(1-z_{it})z_{i,t-1}s_{it}^{OFF,c}$$
where $f^c(p_{it})=\alpha_i^cp_{it}^2+\beta^cp_{it}+\delta+|\rho^csin[\varphi^c_i(p_{it}^{min}+p_{it})]|$

Emission function: $$\cal E_{it}=z_{it}f^e(p_{it})+z_{it}(1-z_{i,t-1})s_{it}^{ON,e}+(1-z_{it})z_{i,t-1}s_{it}^{OFF,e}$$
where $f^e(p_{it})=\alpha_i^ep_{it}^2+\beta^ep_{it}+\delta+\rho^eexp(\varphi^e_ip_{it})$

The MOPS problem:
$$\cal \Phi(C,E)=\sum\limits_{t=1}^{24}\sum\limits_{i=1}^n[\omega_0C_{it}+\sum\limits_{h=1}^m\omega_h\eta_{ih}E_{it}^{(h)}]$$

subject to:

| Constraints | Specification | 
| --------------- | --------------- | 
| Min and max power capacity limits:    | $z_{it}p_{i*}^{min}\le p_{it}\le z_{it}p_{i*}^{max}$ | 
| Max ramp down and up rates:    | $z_{it}p_{i,t-1}-z_{it}p_{it}\le p_{i*}^{down}, z_{it}p_{it}-z_{i,t-1}p_{it}\le p_{i*}^{up}$    | 
| Min operating (online/offline) durations:     | $\script{t}_{it}^{ON}\ge $    | 
| Max ramp down and up rates:    | Row 2, Col 2    | 
| Power supply and demand balance:    | Row 1, Col 2    | 
| Min available reserve capacity:   | Row 2, Col 2    | 

Within the framework of MARL, the generating units are represented as multi-RL agents, each with different unit-specific characteristics and multiple conflicting objectives. The framework MARL manifests the form of state $\cal S$, action  $\cal A$, transition (probability) function  $\cal P$ and reward  $\cal R$ for a sequence of discrete timesteps $t$. 
### The MARL Framework
The scheduling horizon is an hourly divided day, each hour of a day is considered a timestep $t,\forall t$.  Hence, one cycle of determination of commitments and load dispatches for a day represents a complete episode.
- **State Space** $\cal s_t\in S$: The state $\cal s_t$ at timestep $t$ consists of the current timestep$t$, a vector of minimum capacities $p_t^{min}$, a vector of maximum capacities $p_t^{max}$, a vector of current (online/offline) duration $tt_t$ ; and the demand $d_t$.
- **Action Space** $\cal a_t\in A$: The action $\cal a_t$ consists of the decision of either of the two possible actions (switch-to/stay ON (1) or switch-to/stay OFF (0)) of each of the $n$ agents at each timestep $t$ (or state $\cal s_t$).
- **Transition Function** $\cal P(s_t' |s_t,a_t)$: Once the agents take actions $\cal a_t\in A$ in $\cal s_t\in S$, the transition (or probability) function $\cal P(s_t' |s_t,a_t)$ determines whether to advance to the next succeeding state $\cal s_t'=s_{t+1}$ or re-initialize the next state $\cal s_t'=s_{0}$.
  - The illegal decisions of the agents are corrected by the environment.
  - The environment adjusts for both excess and shortages of power supplies.
- **Reward function** $\cal R$: The agents get a common reward based on a predefined reinforcement function $\cal R(s_t,a_t,s_t')$ based on the action $\cal a_t\in A$ in state $\cal s_t\in S$.

The MOPS dynamics can be formally formulated as a 4-tuple $\cal (S,A,P,R)$ Markov Decision Process (MDP), which can then be used to train a custom deep learning model.The practical viability of the environment is evaluated on different test systems featuring mono- to tri-objective problems.

## Installation

The simulation environment can be installed by running:

    ```
    git clone https://github.com/??.git
    cd ?
    pip install .
    ```

## Usage

Below, we will try an action on the 5 generator system. An action is a commitment decision for the following time period, defined by a binary numpy array: 1 indicates that we want to turn (or leave) the generator on, 0 indicates turn or leave it off. 

### Import package

```python 
from ?.environment import make_env
import numpy as np
```

### Create simulation environment
```
env = make_env()
```
### Reset environment
```
obs_init = env.reset()
```
### Define decisions of agents
```
action_vec = np.array([1,1,0,0,0])
```

### Execute action in the environment
```
observation, reward, done = env.step(action)
```

```
print("Dispatch: {}".format(env.disp))
print("Finished? {}".format(done))
print("Reward: {:.2f}".format(reward))
```

## Documentation
A detailed description of how to use it can be found at [jupyter notebook](notebooks/tutorial.ipynb).

### Contact Information
Any questions, issues, suggestions, or collaboration opportunities can be reached at: es.awol@gmail.com. 

### Acknowledgment


## Citation

Users of the repository should cite the following paper: 
    
    Ebrie, A.S.; Paik, C.; Chung, Y.; Kim, Y.J. (2023)..energies, 16(?).



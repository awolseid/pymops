# `pymops`: A multi-agent reinforcement learning simulation environment for multi-objective optimization in power scheduling

## About

Power scheduling is an NP-hard optimization problem with high dimensionality, combinatorial nature, non-convex, non-smooth, and discontinuous properties together with multi-period multiple constraints.

- Two sequential tasks:
  - Unit Commitment
  - Load Dispatch:
    - Economic Load Dispatch
    - Environmental Load Dispatch

Multi-Objective Power Scheduling aims to determine an optimal load dispatch schedule for simultaneously minimizing different conflicting objectives, particularly economic costs and environmental emissions.

`pymops` is an open-source Python package developed for solving mono- to tri-objective optimization in power scheduling problems. The package is built on a novel multi-agent reinforcement learning (MARL)-based adaptive simulation environment, where the power-generating units are represented as multiple RL agents. The agents are heterogeneous, each with multiple conflicting objectives. The MOPS dynamics are simulated from the environment using Markov Decision Processes (MDPs), which are then used to train a Multi-Agent Deep RL (MADRL) model. 



### Multi-Objective Function

The multi-objective function is formulated by combining the different conflicting objectives via a hybrid approach that uses both weighting hyperparameters and unit-specific cost-to-emission conversion factors:

$$\cal \Phi(C,E)=\sum\limits_{t=1}^{24}\sum\limits_{i=1}^n[\omega_0C_{ti}+\sum\limits_{h=1}^m\omega_h\eta_{ih}E_{ti}^{(h)}]$$

where $\eta_i$ denotes cost-to-emission conversion parameter which is defined as $$\displaystyle \eta_i = exp[\frac{\cal \nabla C^{on}(p_i)/\nabla E^{on}(p_i)}{{max[\cal \nabla C^{on}(p_i)/\nabla E^{on}(p_i);\forall i]-min[\cal \nabla C^{on}(p_i)/\nabla E^{on}(p_i);\forall i]}}];\forall i$$
and $\omega_h, h=0,1,...,m$ represents the weight hyperparameter associated with objective $m$.

**Economic Cost Functions**:

$$\cal C_{ti}=z_{ti}C^{on}(p_{ti})+z_{ti}(1-z_{t-1,i})C_{ti}^{su}+(1-z_{ti})z_{t-1,i}C_{ti}^{sd};\forall i,t$$ 
where 
$$\cal C^c(p_{ti})=a_i^cp_{ti}^2+b^cp_{ti}+c^c+|d^c sin[e^c_i(p_{ti}^{min}+p_{ti})]|;\forall i,t$$

**Environmental Emission Functions**: 
$$\cal E_{ti}=z_{ti}E^{on}(p_{ti})+z_{ti}(1-z_{t-1,i})E_{ti}^{su}+(1-z_{ti})z_{t-1,i}E_{ti}^{sd};\forall i,t$$
 where 
 $$\cal E^e(p_{ti})=a_i^ep_{ti}^2+b^ep_{ti}+c^e+d^eexp(e^e_ip_{ti});\forall i,t$$



| Constraints                              | Specification                            |
| ---------------------------------------- | ---------------------------------------- |
| Minimum and maximum power capacities:    | $\cal z_{ti}p_{i}^{min}\le p_{ti}\le z_{ti}p_{i}^{max}$ |
| Maximum ramp-down and ramp-up rates:     | $\cal z_{ti}p_{t-1,i}-z_{ti}p_{ti}\le p_{i}^{down}$ and $z_{ti}p_{ti}-z_{t-1,i}p_{ti}\le p_{i}^{up}$ |
| Mininmum operating (online/offline) durations: | $\cal tt_{ti}^{ON}\ge tt_{i}^{OFF}$ and $tt_{ti}^{OFF}\ge tt_{i}^{OFF}$ |
| Power supply and demand balance:         | $\cal \sum\limits_{i=1}^nz_{ti}p_{ti}=d_t$ |
| Minimum available reserve:               | $\cal \sum\limits_{i=1}^nz_{ti}p_{ti}^{max}\ge (1+ r) d_t$ |




### The MARL Framework
The framework MARL manifests the form of state $\cal S$, action  $\cal A$, transition (probability) function  $\cal P$ and reward  $\cal R$. 

- **Planning Horizon**: The scheduling horizon is an hourly divided day.
  - **Timestep/Period**: Each hour of a day is considered a timestep.  

  - **Episode**: One cycle of determination of unit commitments and load dispatches for a day.

    ​

- **Simulation Enviroment**: Custom MARL simulation environment, structurally similar to OpenAI Gym.
  - Mono-objective to tri-objective scheduling problem (cost, CO2 and SO2).

  - Ramp rate constraints and valve point effects are taken into account.

    ​

- **Agents**: The generating units are represented as multiple agents.
  - The agents are heterogenous (different generating-unit-specific characteristics).
  - Each agent has multiple conflicting objectives. 
  - The agents are cooperative type of RL agents: 
    - Agents collaborate the satisfy the demand at each period/timestep.

    - Agents also strive to minimize the multi-objective function in the entire planning horizon.

      ​

- **State Space**:  Consists of timestep, minimum and maximum capacities,  operating (online/offline) durations, demand to be satisfied.

  ​

- **Action Space**: The commitment statuses (ON/OFF) of all agents.

  ​

- **Transition Function**: The probability of making transition from current state to the next state (no specific formula).
  - The decisions of agents violating any constraint is automatically corrected by the environment.

  - The environment makes also adjustments for both excess and shortages of power supplies.

    ​

- **Reward function**: Agents get a common reward which is the inverse of the average of the normalized value of all objectives.



The MOPS dynamics can be simulated as a 4-tuple $\cal (S,A,P,R)$ MDP:
- The MDPs are input for the deep RL model.

- The deep RL model predicts decision (action) of agents.

- The predicted agents' action is input for the transition function in the environment

  ​



## Installation

The simulation environment can be installed using `pip` :

        ```
        pip install pymops
        ```

Or it can be cloned from GitHub repo and installed.

        ```
        git clone https://github.com/awolseid/pymops.git
        cd pymops
        pip install .
        ​```

### Import package

        ```python 
        import pymops
        from pymops.environ import SimEnv
        ```

### Create simulation environment

        ```
        env = SimEnv(
                    supply_df = default_supply_df, # Units' profile dataframe
                    demand_df = default_demand_df, # Demands profile
                    SR = 0.0, # proportion of spinning reserve => [0, 1]
                    RR = "Yes", # Ramp rate => "yes" or (default "no" (=None)) 
                    VPE = None, # Valve point effects => "yes" or (default "no" (=None))
                    n_objs = None, # Objectives => "tri" for 3 or (default "bi" (=None) for bi-objective)
                    w = None, # Weight => [0, 1] for bi-objective, a list [0.2,0.3,0.5] for tri-objective
                    duplicates = None # Num of duplicates: duplicate units and adjust demands proportionally
                )
        ```

#### Reset environment

        ```
        initial_flat_state, initial_dict_state = env.reset()
        ```

#### Get current state

        ```
        flat_state, dict_state = env.get_current_state()
        ```

#### Execute decision (action) of agents

        ```
        action_vec = np.array([1,1,0,1,0,0,0,0,0,0])
        flat_next_state, reward, done, next_state_dict, dispatch_info = env.step(action_vec)
        ```

## Develop and training (own customized) model

### Import packages

        ```python 
        from pymops.define_dqn import DQNet
        from pymops.madqn import DQNAgents
        from pymops.replaymemory import ReplayMemory
        from pymops.schedules import get_schedules
        ```

### Define model

          ```
          model_0 = DQNet(env, 64)
          print(model_0)
          ```

### Create instance

          ```
          RL_agents = DQNAgents(
                                  environ = env, 
                                  model = model_0, 
                                  epsilon_max = 1.0,
                                  epsilon_min = 0.1,
                                  epsilon_decay = 0.99,
                                  lr = 0.001
                                  )
          ```

### Replay memory

          ```
          memory = ReplayMemory(environ = env, buffer_size = 64)
          ```

### Train model

          ```
          training_results_df = RL_agents.train(memory = memory, batch_size = 64, num_episodes = 500)
          ```

### Get schedule solutions

          ```
          cost, emis, CO2, SO2, schedules_df = get_schedules(environ = env, trained_agents = RL_agents)
          schedules_df
          ```

### Contact Information
Any questions, issues, suggestions, or collaboration opportunities can be reached at: awolseid@pukyong.ac.kr ; youngk@pknu.ac.kr. 


### Citation

Users should cite the following resources. 

- Code Ocean Reproducible Capsule: https://codeocean.com/capsule/0242917/tree:

  - **Ebrie, A.S.**;, **Kim, Y.J.** (2023). pymops: *A multi-agent reinforcement learning simulation environment for multi-objective optimization in power scheduling* [Software Code]. https://doi.org/10.24433/CO.9235622.v1 
- **[Article](https://www.mdpi.com/1996-1073/16/16/5920) produced from the very first version of the package:
  - **Ebrie, A.S.**; **Paik, C.**; **Chung, Y.**; **Kim, Y.J.** (2023). *Environment-Friendly Power Scheduling Based on Deep Contextual Reinforcement Learning*. *Energies*, 16, 5920. https://doi.org/10.3390/en16165920.   
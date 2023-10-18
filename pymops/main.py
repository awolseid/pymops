import pandas as pd

from pymops.environ import SimEnv
from pymops.define_dqn import DQNet
from pymops.madqn import DQNAgents
from pymops.replaymemory import ReplayMemory
from pymops.schedules import get_schedules


def main():
    sup_url = "https://github.com/awolseid/pymops/raw/main/pymops/data/supply_profiles.csv"
    dem_url = "https://github.com/awolseid/pymops/raw/main/pymops/data/demand_profiles.csv"

    SR          =  0.10
    HN          = 64
    EPS_MAX     =  1
    EPS_MIN     =  0
    LR          =  0.01
    MEMORY_SIZE = 64
    BATCH       = 64


    settings_dict = {
        "N_OBJS":    ["bi", "bi", "bi", "bi", "tri", "tri"],
        "RR":        ["no", "yes", "no", "yes", "no", "yes"],
        "VPE":       ["no", "no", "yes", "yes", "no", "no"],
        "W":         [0.36, 0.38, 0.92, 0.69, [0.61, 0.06, 0.33], [0.41, 0.28, 0.31]],
        "EPS_DECAY": [0.999, 0.999, 0.999, 0.999, 0.9991, 0.9991],
        "EPISODES":  [8000, 8000, 8000, 8000, 10000, 10000]
    }
    
    for i in range(len(settings_dict)):
        N_OBJS    = settings_dict['N_OBJS'][i]
        RR        = settings_dict['RR'][i]
        VPE       = settings_dict['VPE'][i]
        W         = settings_dict['W'][i]
        EPS_DECAY = settings_dict['EPS_DECAY'][i]
        EPISODES  = settings_dict['EPISODES'][i]

        env = SimEnv(supply_df  = pd.read_csv(sup_url), 
                     demand_df  = pd.read_csv(dem_url), 
                     n_objs     = N_OBJS, 
                     SR         = SR, 
                     RR         = RR, 
                     VPE        = VPE,
                     w          = W
                    )

        RL_agents = DQNAgents(environ       = env, 
                              model         = DQNet(env, HN),
                              epsilon_max   = EPS_MAX,
                              epsilon_min   = EPS_MIN,
                              epsilon_decay = EPS_DECAY,
                              lr            = LR
                             )
        
        results_df = RL_agents.train(memory       = ReplayMemory(environ = env, buffer_size = MEMORY_SIZE), 
                                     batch_size   = BATCH, 
                                     num_episodes = EPISODES
                                    )
        result = N_OBJS.upper()+"-OBJECTIVE_"+"RampRate"+str(RR.upper())+"_ValvePointEffect"+str(VPE.upper())
        results_df.to_excel(result+".xlsx", index = False) 
        print(f"Results saved with filename: {result}")
        
        schedules_df = get_schedules(environ = env, trained_agents = RL_agents)
        
        schedule = N_OBJS.upper()+"-OBJECTIVE_"+"RampRate"+str(RR.upper())+"_ValvePointEffect"+str(VPE.upper())
        schedules_df.to_excel(schedule+".xlsx", index = False)
        print(f"Schedules saved with filename: {schedule}")
        
if __name__ == "__main__":
    main()
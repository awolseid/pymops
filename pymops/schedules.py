import numpy as np
import pandas as pd

def get_schedules(environ, trained_agents):
  commits_array = []
  loads_array = []
  max_capacity = []

  start_costs = [] 
  prod_costs = [] 
  shut_costs = [] 
  total_costs = []
  start_emiss = []
  shut_emiss = []
  prod_emiss = []
  total_emiss = []
  start_CO2s = [] 
  shut_CO2s = []
  prod_CO2s = []
  total_CO2s = []

  start_SO2s = [] 
  shut_SO2s = []
  prod_SO2s = []
  total_SO2s = []

  n_periods = 0
  done = False
  state_vec, _ = environ.reset()
  while not done:
    action_vec = trained_agents.act(state_vec)
    next_state_vec, _ , done, _, info_dict = environ.step(action_vec)
    commits_array.append(list(info_dict["action_vec"]))
    max_capacity.append(np.sum(info_dict["action_vec"] * environ.p_max_vec))
    loads_array.append(list(info_dict["loads"]))
    start_costs.append(info_dict["start_cost"])
    shut_costs.append(info_dict["shut_cost"])
    prod_costs.append(info_dict["prod_cost"])
    total_costs.append(info_dict["total_cost"])
    if environ.n_objs == "bi":
        start_emiss.append(info_dict["start_emis"])
        shut_emiss.append(info_dict["shut_emis"])
        prod_emiss.append(info_dict["prod_emis"])
        total_emiss.append(info_dict["total_emis"]) 
    elif environ.n_objs == "tri":
        start_CO2s.append(info_dict["start_CO2"])
        shut_CO2s.append(info_dict["shut_CO2"])
        prod_CO2s.append(info_dict["prod_CO2"])
        total_CO2s.append(info_dict["total_CO2"]) 
        
        start_SO2s.append(info_dict["start_SO2"])
        shut_SO2s.append(info_dict["shut_SO2"])
        prod_SO2s.append(info_dict["prod_SO2"])
        total_SO2s.append(info_dict["total_SO2"]) 
        
    
    n_periods += 1
  
  demands_vec = environ.demands_vec[ : n_periods]
  reserves_vec = ((max_capacity / demands_vec - 1) * 100).round(1).reshape(-1, 1)
  periods_vec = np.arange(1, n_periods + 1).reshape(-1, 1)

  n_units = environ.n_units
  commit_names = ["U" + str(i) for i in range(1, n_units + 1)]
  load_names = ["P_" + str(i) for i in range(1, n_units + 1)]
  if environ.n_objs == "bi":
      dispatch_df = pd.DataFrame(np.concatenate([periods_vec, demands_vec.reshape(-1, 1), 
                                                 np.array(commits_array), np.array(loads_array),
                                                 reserves_vec,
                                                 np.array(start_costs).reshape(-1, 1),
                                                 np.array(shut_costs).reshape(-1, 1),
                                                 np.array(prod_costs).reshape(-1, 1),
                                                 np.array(total_costs).reshape(-1, 1),
                                                 
                                                 np.array(start_emiss).reshape(-1, 1),
                                                 np.array(shut_emiss).reshape(-1, 1),
                                                 np.array(prod_emiss).reshape(-1, 1),
                                                 np.array(total_emiss).reshape(-1, 1)
                                                ], axis=1),
                                 columns = ["Hour","Demand", *commit_names, *load_names, "Reserve (%)",
                                            "Startup Cost", 
                                            "Shutdown Cost",
                                            "Production Cost", 
                                            "Total Cost",

                                            "Startup Emission", 
                                            "Shutdown Emission",
                                            "Production Emission",
                                            "Total Emission"
                                           ])
      dispatch_df[["Hour", *commit_names]] = dispatch_df[["Hour", *commit_names]].astype(int)
      print(f"Total cost = $ {round(np.sum(total_costs), 1)}/day.")
      print(f"Total emis = lb {round(np.sum(total_emiss), 1)}/day.")
  elif environ.n_objs == "tri":
      dispatch_df = pd.DataFrame(np.concatenate([periods_vec, demands_vec.reshape(-1, 1), 
                                                 np.array(commits_array), np.array(loads_array),
                                                 reserves_vec,
                                                 np.array(start_costs).reshape(-1, 1),
                                                 np.array(shut_costs).reshape(-1, 1),
                                                 np.array(prod_costs).reshape(-1, 1),
                                                 np.array(total_costs).reshape(-1, 1),
                                                 
                                                 np.array(start_CO2s).reshape(-1, 1),
                                                 np.array(shut_CO2s).reshape(-1, 1),
                                                 np.array(prod_CO2s).reshape(-1, 1),
                                                 np.array(total_CO2s).reshape(-1, 1),

                                                 np.array(start_SO2s).reshape(-1, 1),
                                                 np.array(shut_SO2s).reshape(-1, 1),
                                                 np.array(prod_SO2s).reshape(-1, 1),
                                                 np.array(total_SO2s).reshape(-1, 1),
                                                ], axis=1),
                                 columns = ["Hour","Demand", *commit_names, *load_names, "Reserve (%)",
                                            "Startup Cost", 
                                            "Shutdown Cost",
                                            "Production Cost", 
                                            "Total Cost",

                                            "Startup CO2", 
                                            "Shutdown CO2",
                                            "Production CO2",
                                            "Total CO2",
                                            
                                            "Startup SO2", 
                                            "Shutdown SO2",
                                            "Production SO2",
                                            "Total SO2"
                                           ])
      dispatch_df[["Hour", *commit_names]] = dispatch_df[["Hour", *commit_names]].astype(int)
      print(f"Total cost = $ {round(np.sum(total_costs), 1)}/day.")
      print(f"Total CO2 = lb {round(np.sum(total_CO2s), 1)}/day.")
      print(f"Total SO2 = lb {round(np.sum(total_SO2s), 1)}/day.")
    
  return (round(np.sum(total_costs), 1), 
          round(np.sum(total_emiss), 1) if environ.n_objs == "bi" else None,
          round(np.sum(total_CO2s), 1) if environ.n_objs == "tri" else None, 
          round(np.sum(total_SO2s), 1) if environ.n_objs == "tri" else None, 
          dispatch_df)
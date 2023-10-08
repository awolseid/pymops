import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint

class SimEnv:
  def __init__(self, 
               supply_df: pd.DataFrame, 
               demand_df: pd.DataFrame, 
               SR: float = 0.0,
               RR: str = None, 
               VPE: str = None,
               n_objs = None, 
               w = None,
               duplicates: int = None):
    if not isinstance(supply_df, pd.DataFrame): raise TypeError("Supply info must be a dataframe.")
    if not isinstance(demand_df, pd.DataFrame): raise TypeError("Demand profile must be a dataframe.")
        
    if not isinstance(float(SR), float): raise TypeError("Reserve percentage must be a number.")
    else: 
        if (SR< 0.0 or SR> 1.0): raise Exception("Reserve percentage must be between 0.0 and 1.0.")

    if RR is not None: 
        if str(RR).isalpha() == False: raise TypeError("Input for RR must be a non-numeric string.")
        else: 
            if RR.lower() not in ["yes", "no"]: 
                raise Exception("Input for RR must be 'Yes' or 'No' (default = None).")
    if VPE is not None: 
        if str(VPE).isalpha() == False: raise TypeError("Input for VPE must be a non-numeric string.")
        else: 
            if VPE.lower() not in ["yes", "no"]: 
                raise Exception("Input for VPE must be 'Yes' or 'No' (default = None).")
            
    if n_objs is None: n_objs = "bi"        
    else:
        if str(n_objs).isalpha() == False:
            raise Exception('Input for n_objs must be a non-numeric string.')
        else:
            if n_objs.lower() not in ["bi", "tri"]: 
                raise Exception('Input for n_objs must be "bi" for bijective or "tri" for trijective.')
    if n_objs.lower() == "bi":
        if w is None: w = [1, 0]
        elif type(w) != float: raise TypeError("Weight w for a bijective must be a scalar.")
        elif type(float(w)) != float: raise TypeError("Weight w for a bijective must be float.")
        elif (w < 0 or w > 1): raise TypeError("Weight w for a bijective must be in [0, 1].")
        else: w = [w, 1 - w]
    elif n_objs.lower() == "tri":
        if w is None: w = np.random.dirichlet(np.ones(3), size = 1)[0]
        elif type(w) == list: 
            if abs(sum(w) - 1) > 0.001: raise Exception("Sum of weights != 1.")
        elif type(float(w)) == float: raise TypeError("Weight w for a trijective must be a list.")
        
    if duplicates == None: duplicates = 1
    elif not isinstance(duplicates, int): raise TypeError("Duplicates must be an integer.")
    else:
        if (duplicates < 0.0): raise Exception("Duplicates must be a positive integer.")
            
    self.duplicates = duplicates
    self.supply_df = supply_df.reindex(supply_df.index.repeat(self.duplicates)).reset_index(drop=True)
    self.demand_df = demand_df * self.duplicates
    self.n_objs = n_objs.lower()
    self.SR = SR
    self.RR = RR.lower() if RR is not None else "no"
    self.VPE = VPE.lower() if VPE is not None else "no"
    self.w = w
    self.w_cost = self.w[0]
    
    self.n_units = self.supply_df.shape[0]
    self.n_timesteps = self.demand_df.shape[0]
    self.demands_vec = self.demand_df["Demand"].to_numpy()

    self.p_min_vec = self.supply_df["P_min"].to_numpy()
    self.p_max_vec = self.supply_df["P_max"].to_numpy()
    if self.RR == "yes":
        self.ramp_dn_vec = self.supply_df["Ramp_down"].to_numpy() 
        self.ramp_up_vec = self.supply_df["Ramp_up"].to_numpy()  
    self.dn_times_vec = self.supply_df["Down_time"].to_numpy()
    self.up_times_vec = self.supply_df["Up_time"].to_numpy()
    self.durations_vec = self.supply_df["Initial_Duration"].to_numpy()
    self.hot_costs_vec = self.supply_df["Hot_Cost"].to_numpy()
    self.cold_costs_vec = self.supply_df["Cold_Cost"].to_numpy()
    self.cold_times_vec = self.supply_df["Cold_Periods"].to_numpy()
    self.shut_costs_vec = (self.supply_df["Shut_Cost"].to_numpy() 
                               if "Shut_Cost" in self.supply_df.columns else np.zeros(self.n_units))
    self.ac_vec = self.supply_df["aCost"].to_numpy()
    self.bc_vec = self.supply_df["bCost"].to_numpy()
    self.cc_vec = self.supply_df["cCost"].to_numpy()
    if self.VPE == "yes":
        self.dc_vec = self.supply_df["dCost"].to_numpy()
        self.ec_vec = self.supply_df["eCost"].to_numpy()
    if self.n_objs == "bi":
        self.ae_vec = self.supply_df["aEmis"].to_numpy()
        self.be_vec = self.supply_df["bEmis"].to_numpy()
        self.ce_vec = self.supply_df["cEmis"].to_numpy()
        if self.VPE == "yes":
            self.de_vec = self.supply_df["dEmis"].to_numpy()
            self.ee_vec = self.supply_df["eEmis"].to_numpy()
        self.start_emiss_vec = (self.supply_df["Start_Emis"].to_numpy() 
                                 if "Start_Emis" in self.supply_df.columns else np.zeros(self.n_units))
        self.shut_emiss_vec = (self.supply_df["Shut_Emis"].to_numpy() 
                                 if "Shut_Emis" in self.supply_df.columns else np.zeros(self.n_units))
        self.w_emis = self.w[1]
 
    elif self.n_objs == "tri":
        self.aCO2_vec = self.supply_df["aCO2"].to_numpy()
        self.bCO2_vec = self.supply_df["bCO2"].to_numpy()
        self.cCO2_vec = self.supply_df["cCO2"].to_numpy()
        if self.VPE == "yes":
            self.dCO2_vec = self.supply_df["dCO2"].to_numpy()
            self.eCO2_vec = self.supply_df["eCO2"].to_numpy()
        self.start_CO2s_vec = (self.supply_df["Start_CO2"].to_numpy()
                              if "Start_CO2" in self.supply_df.columns else np.zeros(self.n_units))
        self.shut_CO2s_vec = (self.supply_df["Shut_CO2"].to_numpy() 
                             if "Shut_CO2" in self.supply_df.columns else np.zeros(self.n_units))

        self.aSO2_vec = self.supply_df["aSO2"].to_numpy()
        self.bSO2_vec = self.supply_df["bSO2"].to_numpy()
        self.cSO2_vec = self.supply_df["cSO2"].to_numpy()
        if self.VPE == "yes":
            self.dSO2_vec = self.supply_df["dSO2"].to_numpy()
            self.eSO2_vec = self.supply_df["eSO2"].to_numpy()
        self.start_SO2s_vec = (self.supply_df["Start_SO2"].to_numpy()
                               if "Start_SO2" in self.supply_df.columns else np.zeros(self.n_units))
        self.shut_SO2s_vec = (self.supply_df["Shut_SO2"].to_numpy()
                              if "Shut_SO2" in self.supply_df.columns else np.zeros(self.n_units))
        self.w_CO2 = self.w[1]
        self.w_SO2 = self.w[2]
        
    self.timestep = 0
    self.commits_vec = np.where(self.durations_vec > 0, 1, 0)
    self.startup_costs_and_emissions_if_ON()
    self.shutdown_costs_and_emissions_if_OFF()
    self.summarize_marginal_functions()
    self.cost_penalty_factors()
    self.penalties_for_incomplete_episodes()
    self.determine_priority_orders()
    self.identify_must_ON_and_must_OFF_units()

    self.incomplete_episode = False
    self.done = False
    self.probs_vec = np.array([(self.demands_vec[i] * (1 + self.SR)) / sum(self.p_max_vec) 
                               for i in range(self.n_timesteps)])

    if self.n_objs == "bi":
        if self.w_cost == 1: 
            print(f'"ECONOMIC COST" Optimization: RR = {self.RR}, VPE = {self.VPE}, SR = {self.SR}')
        elif self.w_emis == 1: 
            print(f'"ENVIRONMENTAL EMISSION" Optimization: RR = {self.RR}, VPE = {self.VPE}, SR = {self.SR}')
        else: 
            print(f"DUAL OBJECTIVES (Cost and Emission): w = {self.w} RR = {self.RR}, VPE = {self.VPE}, SR = {self.SR}")
    elif self.n_objs == "tri":
        if self.w_CO2 == 1: 
            print(f'"CO2 EMISSIONS" Optimization: RR = {self.RR}, VPE = {self.VPE}, Reserve = {self.SR}')
        elif self.w_SO2 == 1: 
            print(f'"SO2 EMISSIONS" Optimization: RR = {self.RR}, VPE = {self.VPE}, Reserve = {self.SR}')
        else: 
            print(f"TRI OBJECTIVES (Cost, CO2 and SO2): w = {self.w}, RR = {self.RR}, VPE = {self.VPE}, SR = {self.SR}")

                
  def startup_costs_and_emissions_if_ON(self):  
    OFF_times_vec = np.abs(np.minimum(self.durations_vec, 0))
    cold_OFF_times_vec = self.dn_times_vec + self.cold_times_vec
    is_hot_cost = np.logical_and(self.dn_times_vec <= OFF_times_vec, OFF_times_vec <= cold_OFF_times_vec)
    is_cold_cost = OFF_times_vec > cold_OFF_times_vec
    self.start_costs_vec = np.where(is_hot_cost, self.hot_costs_vec, np.where(is_cold_cost, self.cold_costs_vec, 0))
    if self.n_objs == "bi":
        self.start_emiss_vec *= np.where(OFF_times_vec > 0, 1, 0)
    elif self.n_objs == "tri":
        self.start_CO2s_vec *= np.where(OFF_times_vec > 0, 1, 0)
        self.start_SO2s_vec *= np.where(OFF_times_vec > 0, 1, 0)        

        
  def shutdown_costs_and_emissions_if_OFF(self):  
    ON_times_vec = np.maximum(self.durations_vec, 0)
    self.shut_costs_vec *= np.where(ON_times_vec > 0, 1, 0)
    if self.n_objs == "bi":
        self.shut_emiss_vec *= np.where(ON_times_vec > 0, 1, 0)
    elif self.n_objs == "tri":
        self.shut_CO2s_vec *= np.where(ON_times_vec > 0, 1, 0)
        self.shut_SO2s_vec *= np.where(ON_times_vec > 0, 1, 0)

  def prod_cost_funs(self, loads_vec: np.ndarray):
    if self.VPE == "yes":
        prod_cost_funs_vec = (self.ac_vec * loads_vec**2 + self.bc_vec * loads_vec + self.cc_vec +
                                            np.abs(self.dc_vec * np.sin(self.ec_vec * (self.p_min_vec - loads_vec))))
    else:
        prod_cost_funs_vec = self.ac_vec * loads_vec**2 + self.bc_vec * loads_vec + self.cc_vec
        
    return np.where(loads_vec > 0, 1, 0) * prod_cost_funs_vec

  def prod_emis_funs(self, loads_vec: np.ndarray, cost_to_emis_factors_vec: np.ndarray = 1):
    if self.VPE == "yes":
        prod_emis_funs_vec = (self.ae_vec * loads_vec**2 + self.be_vec * loads_vec + self.ce_vec +
                              self.de_vec * np.exp(self.ee_vec * loads_vec)) if self.n_objs == "bi" else None
    else:
        prod_emis_funs_vec = (
            self.ae_vec * loads_vec**2 + self.be_vec * loads_vec + self.ce_vec) if self.n_objs == "bi" else None       
        
    return np.where(loads_vec > 0, 1, 0) * cost_to_emis_factors_vec * prod_emis_funs_vec
    
  def prod_CO2_funs(self, loads_vec: np.ndarray, cost_to_CO2_factors_vec: np.ndarray = 1):
    if self.VPE == "yes":
        prod_CO2_funs_vec = (self.aCO2_vec * loads_vec**2 + self.bCO2_vec * loads_vec + self.cCO2_vec + 
        self.dCO2_vec * np.exp(self.eCO2_vec * loads_vec)) if self.n_objs == "tri" else None
    else:
        prod_CO2_funs_vec = (
            self.aCO2_vec * loads_vec**2 + self.bCO2_vec * loads_vec + self.cCO2_vec) if self.n_objs == "tri" else None         
    
    return np.where(loads_vec > 0, 1, 0) * cost_to_CO2_factors_vec * prod_CO2_funs_vec
        

  def prod_SO2_funs(self, loads_vec: np.ndarray, cost_to_SO2_factors_vec: np.ndarray = 1):
    if self.VPE == "yes":
        prod_SO2_funs_vec = (self.aSO2_vec * loads_vec**2 + self.bSO2_vec * loads_vec + self.cSO2_vec + 
        self.dSO2_vec * np.exp(self.eSO2_vec * loads_vec)) if self.n_objs == "tri" else None
    else:
        prod_SO2_funs_vec = (
            self.aSO2_vec * loads_vec**2 + self.bSO2_vec * loads_vec + self.cSO2_vec) if self.n_objs == "tri" else None         
    
    return np.where(loads_vec > 0, 1, 0) * cost_to_SO2_factors_vec * prod_SO2_funs_vec
    

  def summarize_marginal_functions(self):
    # The cost, CO2 and SO2 functions are all upward functions for all units.
    # But, there are some units whose "emission" functions are u shaped.
    # Hence, the min obj value is not necssarily at the min capacity, and 
    # the same is true for max obj values.
    # Thus, the necessity of the min and max value points are below are for this purpose. 
    # I will think of adjusting for non-convex/non smooth functions also.
    self.max_cap_prod_cost = np.sum(self.prod_cost_funs(self.p_max_vec)) # I do not include the time dependent SU costs
    min_prod_cost_points_vec = np.where(self.p_min_vec > (p_min_vec := -self.bc_vec / (2 * self.ac_vec)),
                                       self.p_min_vec, p_min_vec)
    max_prod_cost_points_vec = np.where(self.prod_cost_funs(self.p_min_vec) > self.prod_cost_funs(self.p_max_vec), 
                                   self.p_min_vec, self.p_max_vec)
    
    self.min_prod_costs_vec = self.prod_cost_funs(min_prod_cost_points_vec)
    self.max_prod_costs_vec = self.prod_cost_funs(max_prod_cost_points_vec)
    
    self.min_prod_costs_MW_vec = self.max_prod_costs_vec / self.p_max_vec
    self.max_prod_costs_MW_vec = self.min_prod_costs_vec / self.p_min_vec
    
    self.min_prod_cost = np.min(self.min_prod_costs_MW_vec * self.p_min_vec)
    self.max_prod_cost = np.sum(self.max_prod_costs_MW_vec * self.p_max_vec)
    
    
    if self.n_objs == "bi":
        self.max_cap_prod_emis = np.sum(self.prod_emis_funs(self.p_max_vec))
        min_prod_emis_points_vec = np.where(self.p_min_vec > (p_min_vec := -self.be_vec / (2 * self.ae_vec)),
                                       self.p_min_vec, p_min_vec)
        max_prod_emis_points_vec = np.where(self.prod_emis_funs(self.p_min_vec) > self.prod_emis_funs(self.p_max_vec), 
                                       self.p_min_vec, self.p_max_vec)
        
        self.min_prod_emiss_vec = self.prod_emis_funs(min_prod_emis_points_vec)
        self.max_prod_emiss_vec = self.prod_emis_funs(max_prod_emis_points_vec)
        # Here, I found that f(Pmax)/Pmax <= f(Pmin)/Pmin is not true for all units
        # It results a negative reward; so I made some adjustments
        min_prod_emiss_MW_vec = self.max_prod_emiss_vec / max_prod_emis_points_vec
        max_prod_emiss_MW_vec = self.min_prod_emiss_vec / min_prod_emis_points_vec
        
        self.min_prod_emiss_MW_vec = np.where(min_prod_emiss_MW_vec < max_prod_emiss_MW_vec, 
                                              min_prod_emiss_MW_vec, max_prod_emiss_MW_vec)
        self.max_prod_emiss_MW_vec = np.where(min_prod_emiss_MW_vec > max_prod_emiss_MW_vec, 
                                              min_prod_emiss_MW_vec, max_prod_emiss_MW_vec)
    
        self.min_prod_emis = np.min(self.min_prod_emiss_MW_vec * self.p_min_vec) 
        self.max_prod_emis = np.sum(self.max_prod_emiss_MW_vec * self.p_max_vec) 
        

    elif self.n_objs == "tri":
        self.max_cap_prod_CO2 = np.sum(self.prod_CO2_funs(self.p_max_vec))
        min_prod_CO2_points_vec = np.where(self.p_min_vec > (p_min_vec := -self.bCO2_vec / (2 * self.aCO2_vec)),
                                      self.p_min_vec, p_min_vec) # same as "self.p_min_vec", no difference
        max_prod_CO2_points_vec = np.where(self.prod_CO2_funs(self.p_min_vec) > self.prod_CO2_funs(self.p_max_vec),
                                      self.p_min_vec, self.p_max_vec) # same as "self.p_max_vec", no difference

        self.min_prod_CO2s_vec = self.prod_CO2_funs(min_prod_CO2_points_vec)
        self.max_prod_CO2s_vec = self.prod_CO2_funs(max_prod_CO2_points_vec)
        # So I made some adjustments as above
        min_prod_CO2s_MW_vec = self.max_prod_CO2s_vec / min_prod_CO2_points_vec
        max_prod_CO2s_MW_vec = self.min_prod_CO2s_vec / max_prod_CO2_points_vec 
        
        self.min_prod_CO2s_MW_vec = np.where(min_prod_CO2s_MW_vec < max_prod_CO2s_MW_vec, 
                                             min_prod_CO2s_MW_vec, max_prod_CO2s_MW_vec)
        self.max_prod_CO2s_MW_vec = np.where(min_prod_CO2s_MW_vec > max_prod_CO2s_MW_vec, 
                                             min_prod_CO2s_MW_vec, max_prod_CO2s_MW_vec)
        
        self.min_prod_CO2 = np.min(self.min_prod_CO2s_MW_vec * self.p_min_vec) 
        self.max_prod_CO2 = np.sum(self.max_prod_CO2s_MW_vec * self.p_max_vec)
        
        self.max_cap_prod_SO2 = np.sum(self.prod_SO2_funs(self.p_max_vec))
        min_prod_SO2_points_vec = np.where(self.p_min_vec > (p_min_vec := -self.bSO2_vec / (2 * self.aSO2_vec)),
                                      self.p_min_vec, p_min_vec)
        max_prod_SO2_points_vec = np.where(self.prod_SO2_funs(self.p_min_vec) > self.prod_SO2_funs(self.p_max_vec),
                                      self.p_min_vec, self.p_max_vec)
        self.min_prod_SO2s_vec = self.prod_SO2_funs(min_prod_SO2_points_vec)
        self.max_prod_SO2s_vec = self.prod_SO2_funs(max_prod_SO2_points_vec)
        # It is true that f(Pmax)/Pmax <= f(Pmin)/Pmin for all units
        # I just added it for consistency
        min_prod_SO2s_MW_vec = self.max_prod_SO2s_vec / self.p_max_vec
        max_prod_SO2s_MW_vec = self.min_prod_SO2s_vec / self.p_min_vec

        self.min_prod_SO2s_MW_vec = np.where(min_prod_SO2s_MW_vec < max_prod_SO2s_MW_vec, 
                                             min_prod_SO2s_MW_vec, max_prod_SO2s_MW_vec)
        self.max_prod_SO2s_MW_vec = np.where(min_prod_SO2s_MW_vec > max_prod_SO2s_MW_vec, 
                                             min_prod_SO2s_MW_vec, max_prod_SO2s_MW_vec)
        
        self.min_prod_SO2 = np.min(self.min_prod_SO2s_MW_vec * self.p_min_vec) 
        self.max_prod_SO2 = np.sum(self.max_prod_SO2s_MW_vec * self.p_max_vec)
        
    
  def cost_penalty_factors(self):
    max_cap_prod_costs_vec = self.prod_cost_funs(self.p_max_vec)
    min_cap_prod_costs_vec = self.prod_cost_funs(self.p_min_vec)
    
    if self.n_objs == "bi":
#         self.eta_min_min_vec = self.min_prod_costs_vec / self.min_prod_emiss_vec
#         self.eta_min_max_vec = self.min_prod_costs_vec / self.max_prod_emiss_vec
#         self.eta_max_min_vec = self.max_prod_costs_vec / self.min_prod_emiss_vec
#         self.eta_max_max_vec = self.max_prod_costs_vec / self.max_prod_emiss_vec
#         self.eta_mean_vec = (self.eta_min_min_vec + self.eta_min_max_vec + 
#                              self.eta_max_min_vec + self.eta_max_max_vec) / 4 
        
        max_cap_prod_emiss_vec = self.prod_emis_funs(self.p_max_vec)
        min_cap_prod_emiss_vec = self.prod_emis_funs(self.p_min_vec)
        emis_slopes_vec = (max_cap_prod_costs_vec - min_cap_prod_costs_vec) / (max_cap_prod_emiss_vec - min_cap_prod_emiss_vec)
        standardized_slopes_vec = emis_slopes_vec / (max(emis_slopes_vec) - min(emis_slopes_vec))
        self.cost_to_emis_factors_vec = np.exp(standardized_slopes_vec) 
    elif self.n_objs == "tri":
#         self.eta1_min_min_vec = self.min_prod_costs_vec / self.min_prod_CO2s_vec
#         self.eta1_min_max_vec = self.min_prod_costs_vec / self.max_prod_CO2s_vec
#         self.eta1_max_min_vec = self.max_prod_costs_vec / self.min_prod_CO2s_vec
#         self.eta1_max_max_vec = self.max_prod_costs_vec / self.max_prod_CO2s_vec
#         self.eta1_mean_vec = (self.eta1_min_min_vec + self.eta1_min_max_vec + 
#                               self.eta1_max_min_vec + self.eta1_max_max_vec) / 4 
#         self.eta1_common = np.sum(self.eta1_mean_vec) / self.n_units
        
        
        max_cap_prod_CO2s_vec = self.prod_CO2_funs(self.p_max_vec)
        min_cap_prod_CO2s_vec = self.prod_CO2_funs(self.p_min_vec)
        CO2_slopes_vec = (max_cap_prod_costs_vec - min_cap_prod_costs_vec)/ (max_cap_prod_CO2s_vec - min_cap_prod_CO2s_vec)
        standardized_CO2_slopes_vec = CO2_slopes_vec / (max(CO2_slopes_vec) - min(CO2_slopes_vec))
        self.cost_to_CO2_factors_vec = np.exp(standardized_CO2_slopes_vec)
        
#         self.eta2_min_min_vec = self.min_prod_costs_vec / self.min_prod_SO2s_vec
#         self.eta2_min_max_vec = self.min_prod_costs_vec / self.max_prod_SO2s_vec
#         self.eta2_max_min_vec = self.max_prod_costs_vec / self.min_prod_SO2s_vec
#         self.eta2_max_max_vec = self.max_prod_costs_vec / self.max_prod_SO2s_vec
#         self.eta2_mean_vec = (self.eta1_min_min_vec + self.eta1_min_max_vec + 
#                               self.eta1_max_min_vec + self.eta1_max_max_vec) / 4 
#         self.eta2_common = np.sum(self.eta1_mean_vec) / self.n_units
        
        max_cap_prod_SO2s_vec = self.prod_SO2_funs(self.p_max_vec)
        min_cap_prod_SO2s_vec = self.prod_SO2_funs(self.p_min_vec)
        SO2_slopes_vec = (max_cap_prod_costs_vec - min_cap_prod_costs_vec)/ (max_cap_prod_SO2s_vec - min_cap_prod_SO2s_vec)
        standardized_SO2_slopes_vec = SO2_slopes_vec / (max(SO2_slopes_vec) - min(SO2_slopes_vec))
        self.cost_to_SO2_factors_vec = np.exp(standardized_SO2_slopes_vec)  

  def penalties_for_incomplete_episodes(self):
    # startup values are not included
    self.cost_penalties_vec = np.linspace(self.max_prod_cost, self.max_cap_prod_cost, num = self.n_timesteps)
    if self.n_objs == "bi":  
        self.emis_penalties_vec = np.linspace(self.max_prod_emis, self.max_cap_prod_emis, num = self.n_timesteps)
    elif self.n_objs == "tri":
        self.CO2_penalties_vec = np.linspace(self.max_prod_CO2, self.max_cap_prod_CO2, num = self.n_timesteps)
        self.SO2_penalties_vec = np.linspace(self.max_prod_SO2, self.max_cap_prod_SO2, num = self.n_timesteps)
    
  def determine_priority_orders(self): 
    up_times_vec = np.maximum(self.up_times_vec, 0.001) # setting the minimum up time durations to 0.001
    dn_times_vec = np.maximum(self.dn_times_vec, 0.001) # setting the minimum off time durations to 0.001
    ON_costs_vec = (self.min_prod_costs_MW_vec + (self.start_costs_vec / self.p_max_vec)) / up_times_vec
    if self.n_objs == "bi":
        ON_emiss_vec = (self.min_prod_emiss_MW_vec + (self.start_emiss_vec / self.p_max_vec)) / up_times_vec
        if self.w_cost == 1: self.ON_priorities_vec = ON_costs_vec
        elif self.w_emis == 1: self.ON_priorities_vec = ON_emiss_vec
        else: self.ON_priorities_vec = (ON_costs_vec + ON_emiss_vec) / 2
        self.ON_priority_idx_vec = self.ON_priorities_vec.argsort() 
    elif self.n_objs =="tri":
        ON_CO2_vec = (self.min_prod_CO2s_MW_vec + (self.start_CO2s_vec / self.p_max_vec)) / up_times_vec
        ON_SO2_vec = (self.min_prod_SO2s_MW_vec + (self.start_SO2s_vec / self.p_max_vec)) / up_times_vec
        if self.w_cost == 1: self.ON_priorities_vec = ON_costs_vec
        elif self.w_CO2 == 1: self.ON_priorities_vec = ON_CO2_vec
        elif self.w_SO2 == 1: self.ON_priorities_vec = ON_SO2_vec
        else: self.ON_priorities_vec = (ON_costs_vec + ON_CO2_vec + ON_SO2_vec) / 3
        self.ON_priority_idx_vec = self.ON_priorities_vec.argsort()
        
  def identify_must_ON_and_must_OFF_units(self): 
    initial_durations_vec = self.supply_df["Initial_Duration"].to_numpy()
    initial_OFF_times_vec = np.where(initial_durations_vec < 0, np.abs(initial_durations_vec), 0)
    self.must_OFF_vec = np.logical_and(-self.dn_times_vec < self.durations_vec, self.durations_vec < 0)
    initial_ON_times_vec = np.where(initial_durations_vec > 0, initial_durations_vec, 0)
    self.must_ON_vec = np.logical_and(0 < self.durations_vec, self.durations_vec < self.up_times_vec)
    if np.any(self.commits_vec): # future demand satisfaction
        prev_ON_idx_vec = np.where(self.commits_vec == 1)[0]
        priority_idx_vec = np.array([i for i in self.ON_priority_idx_vec if i in prev_ON_idx_vec])
        demands_sr_vec = (1 + self.SR) * self.demands_vec
        for idx in priority_idx_vec:
            max_timestep = min(self.timestep + self.dn_times_vec[idx], self.n_timesteps)
            for timestep in range(self.timestep, max_timestep):
                act_vec = np.ones(self.n_units)
                act_vec[idx] = 0
                if np.sum(act_vec * self.p_max_vec) < demands_sr_vec[timestep]:
                    self.must_ON_vec[idx] = True
                    break
                    

  def step(self, action_vec: np.ndarray):
    if not isinstance(action_vec, np.ndarray):
        raise TypeError("Action vector must be a NumPy array.")
    state_flat, state_dict = self.get_current_state()
    self.demand = state_dict["demand"]
    demand_SR= round((1 + self.SR) * self.demand, 1)
    self.action_vec = self.ensure_action_legitimacy(demand_SR, action_vec)
    self.get_operation_costs_and_emissions(self.demand, self.action_vec)
    reward = self.evaluate_action_reward(demand_SR, self.action_vec)     
    dispatch_info = self.dispatch_info() 
    is_done = self.is_terminal()
    if self.timestep < self.n_timesteps - 1: 
        next_state_flat, next_state_dict = self.get_next_state(self.action_vec)
    else: next_state_flat, next_state_dict = self.reset()
        
    return next_state_flat, reward, is_done, next_state_dict, dispatch_info                                 

  def get_current_state(self):
    self.startup_costs_and_emissions_if_ON()
    self.shutdown_costs_and_emissions_if_OFF()
    self.determine_priority_orders()
    self.identify_must_ON_and_must_OFF_units()
    state_dict = {
        "timestep": self.timestep + 1,
        "demand": self.demands_vec[self.timestep],
        "min_capacities": self.p_min_vec,
        "max_capacities": self.p_max_vec, 
        "operating_statuses": self.durations_vec,
        "commitments": self.commits_vec,
        
        "start_costs_if_ON": self.start_costs_vec,
        "shut_costs_if_OFF": self.shut_costs_vec,
        
        "start_emiss_if_ON": self.start_emiss_vec if self.n_objs == "bi" else None,
        "shut_emiss_if_OFF": self.shut_emiss_vec if self.n_objs == "bi" else None,
        
        "start_CO2s_if_ON": self.start_CO2s_vec if self.n_objs == "tri" else None,
        "shut_CO2s_if_OFF": self.shut_CO2s_vec if self.n_objs == "tri" else None,
        
        "start_SO2s_if_ON": self.start_SO2s_vec if self.n_objs == "tri" else None,
        "shut_SO2s_if_OFF": self.shut_SO2s_vec if self.n_objs == "tri" else None, 
        }
    
#     if self.RR == "yes":
#         flat_state = np.concatenate(
#             (
#                 np.array([self.timestep + 1]),
#                 self.durations_vec,
#                 self.p_min_vec,
#                 self.p_max_vec, 
#                 np.array([self.demands_vec[self.timestep]])
#             ))
#     else:
    flat_state = np.concatenate(
            (
                np.array([self.timestep + 1]),
                self.durations_vec,
                np.array([self.demands_vec[self.timestep]])
            ))   
    return flat_state, state_dict

  def ensure_action_legitimacy(self, demand: float, action_vec: np.ndarray):  
    if self._is_action_illegal(action_vec): 
        action_vec = self._legalize_action(action_vec) 
    if np.sum(action_vec * self.p_max_vec) < demand: 
        action_vec = self._adjust_low_capacity(demand, action_vec)
    elif np.sum(action_vec * self.p_min_vec) > demand: 
        action_vec = self._adjust_excess_capacity(demand, action_vec)
    return action_vec

  def _is_action_illegal(self, action_vec: np.ndarray):
    any_illegal_ON = np.any(action_vec[self.must_ON_vec] == 0)              
    any_illegal_OFF = np.any(action_vec[self.must_OFF_vec] == 1)             
    return any([any_illegal_ON, any_illegal_OFF])

  def _legalize_action(self, action_vec: np.ndarray): 
    illegal_action_vec = action_vec.copy()                                   
    action_vec = np.array(np.logical_or(illegal_action_vec, self.must_ON_vec)
                          * np.logical_not(self.must_OFF_vec), dtype = int) 
    return action_vec

  def _adjust_low_capacity(self, demand: float, action_vec: np.ndarray):
    low_action_vec = action_vec.copy()
    already_OFF_idx_vec = np.where(action_vec == 0)[0]
    must_not_OFF_idx_vec = np.where(self.must_OFF_vec == False)[0]
    can_ON_idx_vec = np.intersect1d(already_OFF_idx_vec, must_not_OFF_idx_vec)
    if len(can_ON_idx_vec) > 0:
        priority_idx_vec = np.array([i for i in self.ON_priority_idx_vec if i in can_ON_idx_vec])
        remaining_supply = demand - np.sum(action_vec * self.p_max_vec)
        for idx in priority_idx_vec:
            action_vec[idx] = 1
            remaining_supply = remaining_supply - self.p_max_vec[idx]
            if remaining_supply <= 0.0001: break           
    return action_vec

  def _adjust_excess_capacity(self, demand: float, action_vec: np.ndarray):
    excess_action_vec = action_vec.copy()
    already_ON_idx_vec = np.where(action_vec == 1)[0]
    must_not_ON_idx_vec = np.where(self.must_ON_vec == False)[0]
    can_OFF_idx_vec = np.intersect1d(already_ON_idx_vec, must_not_ON_idx_vec)
    if len(can_OFF_idx_vec) > 0:
        OFF_priority_idx_vec = np.array([i for i in self.ON_priority_idx_vec[::-1] if i in can_OFF_idx_vec])
        excess_supply = np.sum(action_vec * self.p_min_vec) - demand
        for idx in OFF_priority_idx_vec:
            action_vec[idx] = 0
            excess_supply -= self.p_min_vec[idx]
            if excess_supply <= 0.0001:
                if np.sum(action_vec * self.p_max_vec) < demand:
                    action_vec[idx] = 1
                    break
                break            
    return action_vec

  def get_operation_costs_and_emissions(self, demand: float, action_vec: np.ndarray):
    self._get_startup_results(action_vec)
    self._get_shutdown_results(action_vec)
    self._get_production_results(demand, action_vec)

    self.total_cost = self.start_cost + self.shut_cost + self.prod_cost
    if self.n_objs == "bi":
        self.total_emis = self.start_emis + self.shut_emis + self.prod_emis
    elif self.n_objs == "tri":
        self.total_CO2 = self.start_CO2 + self.shut_CO2 + self.prod_CO2 
        self.total_SO2 = self.start_SO2 + self.shut_SO2 + self.prod_SO2

  def _get_startup_results(self, action_vec: np.ndarray):
    self.start_cost = np.sum(action_vec * (1 - self.commits_vec) * self.start_costs_vec)
    if self.n_objs == "bi":
        self.start_emis = np.sum(action_vec * (1 - self.commits_vec) * self.start_emiss_vec)

    elif self.n_objs == "tri":
        self.start_CO2 = np.sum(action_vec * (1 - self.commits_vec) * self.start_CO2s_vec)
        self.start_SO2 = np.sum(action_vec * (1 - self.commits_vec) * self.start_SO2s_vec)
        
  def _get_shutdown_results(self, action_vec: np.ndarray):
    self.shut_cost = np.sum((1 - action_vec) * self.commits_vec * self.shut_costs_vec)
    if self.n_objs == "bi":
        self.shut_emis = np.sum((1 - action_vec) * self.commits_vec * self.shut_emiss_vec)
        self.shut_emis_penalty = np.sum((1 - action_vec) * self.commits_vec * 
                                        self.cost_to_emis_factors_vec * self.shut_emiss_vec)
    elif self.n_objs == "tri":
        self.shut_CO2 = np.sum((1 - action_vec) * self.commits_vec * self.shut_CO2s_vec)
        self.shut_SO2 = np.sum((1 - action_vec) * self.commits_vec * self.shut_SO2s_vec)

        
  def _get_production_results(self, demand: float, action_vec: np.ndarray):
    if np.sum(action_vec * self.p_max_vec) < demand:
        self.incomplete_episode = True 
        loads_vec = action_vec * self.p_max_vec
        prod_cost = self.cost_penalties_vec[self.timestep] 
        if self.n_objs == "bi":
            prod_emis = self.emis_penalties_vec[self.timestep]
        elif self.n_objs == "tri":
            prod_CO2 = self.CO2_penalties_vec[self.timestep]
            prod_SO2 = self.SO2_penalties_vec[self.timestep]
    elif np.sum(action_vec * self.p_min_vec) > demand:
        self.incomplete_episode = True 
        loads_vec = action_vec * self.p_min_vec
        prod_cost = self.cost_penalties_vec[self.timestep] 
        if self.n_objs == "bi":
            prod_emis = self.emis_penalties_vec[self.timestep]
        elif self.n_objs == "tri":
            prod_CO2 = self.CO2_penalties_vec[self.timestep]
            prod_SO2 = self.SO2_penalties_vec[self.timestep]
    else:
        EC_EM_D = self.optimize_production(action_vec, demand)     
        loads_vec = EC_EM_D["loads_vec"]
        prod_cost = np.sum(self.prod_cost_funs(loads_vec))
        if self.n_objs == "bi":
            prod_emis = np.sum(self.prod_emis_funs(loads_vec))
        elif self.n_objs == "tri":
            prod_CO2 = np.sum(self.prod_CO2_funs(loads_vec))
            prod_SO2 = np.sum(self.prod_SO2_funs(loads_vec))

    self.loads_vec = loads_vec
    self.prod_cost = prod_cost
    if self.n_objs == "bi":
        self.prod_emis = prod_emis
    elif self.n_objs == "tri":
        self.prod_CO2 = prod_CO2
        self.prod_SO2 = prod_SO2

  def optimize_production(self, action_vec: np.ndarray, demand: float):
        idx = np.where(action_vec == 1)[0] 
        n_ON_units = len(idx)
        p_min_vec = self.p_min_vec[idx].reshape(-1, 1)
        p_max_vec = self.p_max_vec[idx].reshape(-1, 1)        
        def objective_function(p_vec):
            if self.VPE == "yes":
                cost_obj_values_vec = (self.ac_vec[idx] * p_vec**2 + self.bc_vec[idx] * p_vec + 
                                       self.cc_vec[idx] + np.abs(self.dc_vec[idx] * 
                                                                 np.sin(self.ec_vec[idx] * (p_min_vec - p_vec))))
                emis_obj_values_vec = (self.ae_vec[idx] * p_vec**2 + self.be_vec[idx] * p_vec + 
                                       self.ce_vec[idx] + self.de_vec[idx] * 
                                       np.exp(self.ee_vec[idx] * p_vec)) if self.n_objs == "bi" else None
                CO2_obj_values_vec = (self.aCO2_vec[idx] * p_vec**2 + self.bCO2_vec[idx] * p_vec + 
                                      self.cCO2_vec[idx] + self.dCO2_vec[idx] * 
                                      np.exp(self.eCO2_vec[idx] * p_vec)) if self.n_objs == "tri" else None
                SO2_obj_values_vec = (self.aSO2_vec[idx] * p_vec**2 + self.bSO2_vec[idx] * p_vec + 
                                      self.cSO2_vec[idx] + self.dSO2_vec[idx] * 
                                      np.exp(self.eSO2_vec[idx] * p_vec)) if self.n_objs == "tri" else None
            else:
                cost_obj_values_vec = (self.ac_vec[idx] * p_vec**2 + self.bc_vec[idx] * p_vec + 
                                       self.cc_vec[idx])
                emis_obj_values_vec = (self.ae_vec[idx] * p_vec**2 + self.be_vec[idx] * p_vec + 
                                       self.ce_vec[idx]) if self.n_objs == "bi" else None
                CO2_obj_values_vec = (self.aCO2_vec[idx] * p_vec**2 + self.bCO2_vec[idx] * p_vec + 
                                      self.cCO2_vec[idx]) if self.n_objs == "tri" else None
                SO2_obj_values_vec = (self.aSO2_vec[idx] * p_vec**2 + self.bSO2_vec[idx] * p_vec + 
                                      self.cSO2_vec[idx]) if self.n_objs == "tri" else None  
            
            if self.n_objs == "bi":
                combined_obj_value = np.sum(self.w_cost * cost_obj_values_vec 
                                            + (self.w_emis * self.cost_to_emis_factors_vec[idx] * emis_obj_values_vec
                                               if self.w_emis != 1 else emis_obj_values_vec)
                                           )
            elif self.n_objs == "tri":
                combined_obj_value = np.sum(self.w_cost * cost_obj_values_vec 
                                            + (self.w_CO2 * self.cost_to_CO2_factors_vec[idx] * CO2_obj_values_vec 
                                               if self.w_CO2 != 1 else CO2_obj_values_vec)
                                            + (self.w_SO2 * self.cost_to_SO2_factors_vec[idx] * SO2_obj_values_vec
                                               if self.w_SO2 != 1 else SO2_obj_values_vec)
                                           )

            if self.n_objs == "bi":
                if 0 <= self.w_cost < 1 and emis_obj_values_vec is None:
                    raise Exception(f"Cost weight = {self.w_cost} but emission obj. values vector is None.")
            elif self.n_objs == "tri":
                if 0 <= self.w_CO2 < 1:
                    if CO2_obj_values_vec is None:
                        raise Exception(f"CO2 weight = {self.w_CO2} but CO2 obj. values vector is None.")
                if 0 <= self.w_SO2 < 1:
                    if SO2_obj_values_vec is None:
                        raise Exception(f"SO2 weight = {self.w_SO2} but SO2 obj. values vector is None.")
            return combined_obj_value

        load_bounds = np.concatenate([p_min_vec, p_max_vec], axis=1)
        constraint = LinearConstraint(np.ones(n_ON_units), lb = demand, ub = demand) 
        optimal_results = minimize(objective_function, x0 = np.random.uniform(p_min_vec, p_max_vec),
                                   method = 'SLSQP', constraints = constraint, bounds = load_bounds) 
        loads_vec = np.zeros(len(action_vec))
        loads_vec[idx] = optimal_results["x"]
        obj_value = optimal_results["fun"]
        return {"loads_vec": loads_vec, "obj_value": obj_value}

  def dispatch_info(self):
    info_dict = {
        "timestep": self.timestep + 1,
        "commitments": self.commits_vec,
        "demand": self.demands_vec[self.timestep],
        "action_vec": self.action_vec, 
        "loads": self.loads_vec.round(1),
        "start_cost": round(self.start_cost, 1),
        "shut_cost": round(self.shut_cost, 1),
        "prod_cost": round(self.prod_cost, 1),
        "total_cost": round(self.total_cost, 1),
        
        "start_emis": round(self.start_emis, 1) if self.n_objs == "bi" else None,
        "shut_emis": round(self.shut_emis, 1) if self.n_objs == "bi" else None,
        "prod_emis": round(self.prod_emis, 1) if self.n_objs == "bi" else None,
        "total_emis": round(self.total_emis, 1) if self.n_objs == "bi" else None,
        
        "start_CO2": round(self.start_CO2, 1) if self.n_objs == "tri" else None,
        "shut_CO2": round(self.shut_CO2, 1) if self.n_objs == "tri" else None,
        "prod_CO2": round(self.prod_CO2, 1) if self.n_objs == "tri" else None,
        "total_CO2": round(self.total_CO2, 1) if self.n_objs == "tri" else None,

        "start_SO2": round(self.start_SO2, 1) if self.n_objs == "tri" else None,
        "shut_SO2": round(self.shut_SO2, 1) if self.n_objs == "tri" else None,
        "prod_SO2": round(self.prod_SO2, 1) if self.n_objs == "tri" else None,
        "total_SO2": round(self.total_SO2, 1) if self.n_objs == "tri" else None,
        }
    return info_dict

  def evaluate_action_reward(self, demand: float, action_vec: np.ndarray):
    CPI = (self.total_cost - self.min_prod_cost) / (self.max_prod_cost - self.min_prod_cost)
    if self.n_objs == "bi":
        EPI = (self.total_emis - self.min_prod_emis) / (self.max_prod_emis - self.min_prod_emis)
        PI = (self.w_cost == 1) * CPI + (self.w_emis == 1) * EPI + (0 < self.w_cost < 1) * (CPI + EPI) / 2
    elif self.n_objs == "tri":
        CO2PI = (self.total_CO2 - self.min_prod_CO2) / (self.max_prod_CO2 - self.min_prod_CO2)
        SO2PI = (self.total_SO2 - self.min_prod_SO2) / (self.max_prod_SO2 - self.min_prod_SO2)
        if self.w_cost == 1: PI = CPI
        elif self.w_CO2 == 1: PI = CO2PI
        elif self.w_SO2 == 1: PI = SO2PI
        else: PI = (CPI + CO2PI + SO2PI) / 3

    reward = 1 / PI

    if reward < 0: raise Exception(f"Reward becomes negative: Reward = {reward}!")
    
    return reward

  def get_next_state(self, action_vec: np.ndarray):
    self.timestep += 1        
    self._update_operating_statuses(action_vec)
    if self.RR == "yes": self._update_production_capacities(action_vec)    
    self.commits_vec = action_vec 
    next_state_dict = self.get_current_state()   
    return next_state_dict


  def _update_production_capacities(self, action_vec: np.ndarray):  
    p_min_vec = self.supply_df["P_min"].to_numpy()
    p_max_vec = self.supply_df["P_max"].to_numpy() 
    self.p_min_vec = np.maximum(p_min_vec, self.commits_vec * action_vec * (self.loads_vec - self.ramp_dn_vec))
    self.p_max_vec = np.minimum(p_max_vec, self.commits_vec * action_vec * (self.loads_vec + self.ramp_up_vec) +
                                np.where((self.commits_vec * action_vec) == 0, 1, 0) * p_max_vec)   
    
    if np.any(self.p_min_vec > self.p_max_vec) == True: raise Exception("Min capacity > Max capacity.")
        

  def _update_operating_statuses(self, action_vec: np.ndarray):
    self.durations_vec = np.array([(self.durations_vec[i] + 1 if action_vec[i] == 1 else -1)
                                  if self.durations_vec[i] > 0 else (1 if action_vec[i] == 1 
                                                                    else self.durations_vec[i] - 1)
                                  for i in range(self.n_units)])

  def is_terminal(self):
    if (self.incomplete_episode == True) or (self.timestep == self.n_timesteps - 1): 
        self.done = True
    return self.done


  def reset(self):
    self.p_min_vec = self.supply_df["P_min"].to_numpy()
    self.p_max_vec = self.supply_df["P_max"].to_numpy()
    self.durations_vec = self.supply_df["Initial_Duration"].to_numpy()
    self.commits_vec = np.where(self.durations_vec > 0, 1, 0)
    self.timestep = 0
    self.incomplete_episode = False
    self.done = False
    inital_state_flat, inital_state_dict = self.get_current_state()

    return inital_state_flat, inital_state_dict
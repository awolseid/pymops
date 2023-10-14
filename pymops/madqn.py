import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from timeit import default_timer
from tqdm.auto import tqdm

class DQNAgents:
    def __init__(self, 
                 environ, 
                 model, 
                 epsilon_max = 1.0, 
                 epsilon_min = 0.0, 
                 epsilon_decay = 1.0, 
                 lr = 0.003, 
                 lr_gamma = 1.0, 
                 gamma = 0.99):
        
        self.environ = environ
        self.n_units = self.environ.n_units
        self.model = model
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.lr_gamma = lr_gamma
        self.lr_sheduler = lr_scheduler.ExponentialLR(self.optimizer, self.lr_gamma)
        self.epsilon_max = epsilon_max  
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.criterion = nn.SmoothL1Loss()
    
    def act(self, state_vec):
        if np.random.rand() <= self.epsilon_max:
            action_vec = np.array(np.random.choice([1, 0], size = self.n_units))
        else:
            q_values_vec = self.model(state_vec)
            q_values_mat = q_values_vec.reshape(self.n_units, 2)
            action_vec = q_values_mat.argmax(axis = 1).detach().numpy()   
        return action_vec
                       
    def update(self, memory, batch_size):       
        sampled_experiences_dict = memory.sample(batch_size)
        sampled_states_mat = torch.Tensor(sampled_experiences_dict["states_mat"]) # batch_size x input_nodes
        sampled_actions_mat = sampled_experiences_dict["actions_mat"] #  batch_size x n_units
        sampled_rewards_vec = sampled_experiences_dict["rewards_vec"]                      # batch_size x 1   
        sampled_next_states_mat = torch.Tensor(sampled_experiences_dict["next_states_mat"]) # batch_size x input_nodes  
        
        q_preds_vec = self.model(sampled_states_mat)                      # batch_size x output_nodes
        q_preds_mat = q_preds_vec.reshape(batch_size, self.n_units, 2) # batch_size x n_units  x 2

        b_size, a_dim =  sampled_actions_mat.shape
        b_idx_mat, u_idx_mat = np.ogrid[ : b_size, : a_dim] # Use actions [batch_size, n_units] to index Q-values
        q_preds_mat = q_preds_mat[b_idx_mat, u_idx_mat, sampled_actions_mat] # batch_size x n_units

        next_q_preds_vec = self.model(sampled_next_states_mat)                      # batch_size x output_nodes
        next_q_preds_mat = next_q_preds_vec.reshape(batch_size, self.n_units, 2) # batch_size x n_units  x 2

        rewards_mat = np.broadcast_to(sampled_rewards_vec, (self.n_units, batch_size)).T # reshape to n_units x batch_size 
        rewards_mat = torch.as_tensor(rewards_mat.copy()).float() 

        next_actions_mat = next_q_preds_mat.argmax(axis=2).detach().numpy()              # batch_size x n_units 
        b_size, a_dim =  next_actions_mat.shape
        b_idx_mat, u_idx_mat = np.ogrid[ : b_size, : a_dim]  # Use actions [batch_size, n_units] to index Q-values
        next_q_preds_mat = next_q_preds_mat[b_idx_mat, u_idx_mat, next_actions_mat] # batch_size x n_units = same shape as reward_vec

        target_q_values_mat = rewards_mat + self.gamma * next_q_preds_mat
        return {"predicted_q_values": q_preds_mat, "target_q_values": target_q_values_mat}
    
    def learn_step(self, memory, batch_size = None):        

        episode_timesteps = []
        episode_rewards = []
        while memory.is_full() == False:
          done = False
          state_vec, _ = self.environ.reset()
          num_timesteps = 0
          total_reward = 0
          while not done:
            action_vec = self.act(state_vec)
            next_state_vec, reward, done, _, info_dict = self.environ.step(action_vec)
            corr_action_vec = info_dict["action_vec"]
            memory.store(state_vec, action_vec, reward, next_state_vec)
            state_vec = next_state_vec
            if memory.is_full(): break
            num_timesteps += 1
            total_reward += reward
            if done:
                episode_timesteps.append(num_timesteps)
                episode_rewards.append(total_reward)

        if batch_size == None: batch_size = memory.buffer_size
        elif batch_size > memory.buffer_size: raise Exception("Batch size > Buffer size.")

        q_values_dict = self.update(memory, batch_size)
        q_pred_mat = q_values_dict["predicted_q_values"]
        q_target_mat = q_values_dict["target_q_values"]
        loss = self.criterion(q_pred_mat, q_target_mat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_sheduler.step()
        memory.reset()
        return np.mean(episode_timesteps), np.mean(episode_rewards)

    def test_step(self, test_environ = None):
        if test_environ == None: test_environ = self.environ
        start_cost = 0
        shut_cost = 0
        prod_cost = 0
        total_cost = 0
        start_emis = 0
        shut_emis = 0
        prod_emis = 0
        total_emis = 0
        start_emis1 = 0
        shut_emis1 = 0
        prod_emis1 = 0
        total_emis1 = 0
        start_emis2 = 0
        shut_emis2 = 0
        prod_emis2 = 0
        total_emis2 = 0

        period = 0
        done = False
        state_vec, _ = test_environ.reset()
        while not done:
            action_vec = self.act(state_vec)
            next_state_vec, _ , done, _, info_dict = test_environ.step(action_vec)
            start_cost = start_cost + info_dict["start_cost"]
            shut_cost = shut_cost + info_dict["shut_cost"]
            prod_cost = prod_cost + info_dict["prod_cost"]
            total_cost = total_cost + info_dict["total_cost"]
            if test_environ.n_objs == "bi":
                start_emis = start_emis + info_dict["start_emis"]
                shut_emis = shut_emis + info_dict["shut_emis"]
                prod_emis = prod_emis + info_dict["prod_emis"]
                total_emis = total_emis + info_dict["total_emis"]
            elif test_environ.n_objs == "tri":
                start_emis1 = start_emis1 + info_dict["start_emis1"]
                shut_emis1 = shut_emis1 + info_dict["shut_emis1"]
                prod_emis1 = prod_emis1 + info_dict["prod_emis1"]
                total_emis1 = total_emis1 + info_dict["total_emis1"]
                
                start_emis2 = start_emis2 + info_dict["start_emis2"]
                shut_emis2 = shut_emis2 + info_dict["shut_emis2"]
                prod_emis2 = prod_emis2 + info_dict["prod_emis2"]
                total_emis2 = total_emis2 + info_dict["total_emis2"]

            state_vec = next_state_vec
            period += 1
        return {"start_cost": start_cost, 
                "shut_cost": shut_cost, 
                "prod_cost": prod_cost,  
                "total_cost": total_cost,
                "start_emis": start_emis if test_environ.n_objs == "bi" else None, 
                "shut_emis": shut_emis if test_environ.n_objs == "bi" else None, 
                "prod_emis": prod_emis if test_environ.n_objs == "bi" else None,  
                "total_emis": total_emis if test_environ.n_objs == "bi" else None,
                
                "start_emis1": start_emis1 if test_environ.n_objs == "tri" else None, 
                "shut_emis1": shut_emis1 if test_environ.n_objs == "tri" else None, 
                "prod_emis1": prod_emis1 if test_environ.n_objs == "tri" else None,  
                "total_emis1": total_emis1 if test_environ.n_objs == "tri" else None,

                "start_emis2": start_emis2 if test_environ.n_objs == "tri" else None, 
                "shut_emis2": shut_emis2 if test_environ.n_objs == "tri" else None, 
                "prod_emis2": prod_emis2 if test_environ.n_objs == "tri" else None,  
                "total_emis2": total_emis2 if test_environ.n_objs == "tri" else None,
               }

    def train(self, memory, test_environ = None, batch_size = None, num_episodes = 100):

      start_time = default_timer()
      mean_timesteps = []
      mean_rewards = []

      start_costs = []
      shut_costs = []
      prod_costs = []
      total_costs = []

      start_emiss = []
      shut_emiss = []
      prod_emiss = []
      total_emiss = []
    
      start_emis1s = []
      shut_emis1s = []
      prod_emis1s = []
      total_emis1s = []
    
      start_emis2s = []
      shut_emis2s = []
      prod_emis2s = []
      total_emis2s = []

      for episode in tqdm(range(1, num_episodes + 1)):
        mean_timestep, mean_reward = self.learn_step(memory, batch_size)
        mean_timesteps.append(mean_timestep)
        mean_rewards.append(mean_reward)

        results_dict = self.test_step(test_environ)
        start_costs.append(results_dict["start_cost"])
        shut_costs.append(results_dict["shut_cost"])
        prod_costs.append(results_dict["prod_cost"])
        total_costs.append(results_dict["total_cost"])
        if self.environ.n_objs == "bi":
            start_emiss.append(results_dict["start_emis"])
            shut_emiss.append(results_dict["shut_emis"])
            prod_emiss.append(results_dict["prod_emis"])
            total_emiss.append(results_dict["total_emis"])
        elif self.environ.n_objs == "tri":
            start_emis1s.append(results_dict["start_emis1"])
            shut_emis1s.append(results_dict["shut_emis1"])
            prod_emis1s.append(results_dict["prod_emis1"])
            total_emis1s.append(results_dict["total_emis1"])

            start_emis2s.append(results_dict["start_emis2"])
            shut_emis2s.append(results_dict["shut_emis2"])
            prod_emis2s.append(results_dict["prod_emis2"])
            total_emis2s.append(results_dict["total_emis2"])

        if episode % (num_episodes / 10) == 0:

            if self.environ.n_objs == "bi":    
                print(f"Episode: {episode} | "
                      f"Steps = {mean_timestep:.1f} | "
                      f"Rewards = {mean_reward:.2f} | "
                      f"Eps. = {self.epsilon_max:.3f} | "
                      f"LR = {self.optimizer.param_groups[0]['lr']:.3f} | "
                      f"Cost = {results_dict['total_cost']:.1f} | "
                      f"Emis. = {results_dict['total_emis']:.1f} | "
                     )
            elif self.environ.n_objs == "tri":
                print(f"Episode: {episode} | "
                      f"Steps = {mean_timestep:.1f} | "
                      f"Rewards = {mean_reward:.2f} | "
                      f"Eps. = {self.epsilon_max:.3f} | "
                      f"LR = {self.optimizer.param_groups[0]['lr']:.3f} | "
                      f"Cost = {results_dict['total_cost']:.1f} | "
                      f"emis1 = {results_dict['total_emis1']:.1f} | "
                      f"emis2 = {results_dict['total_emis2']:.1f} | "
                     )   

        if self.epsilon_max > self.epsilon_min: self.epsilon_max *= self.epsilon_decay
      if self.environ.n_objs == "bi":
          training_results_df = pd.DataFrame(np.concatenate([
              np.array(np.arange(1, num_episodes + 1).reshape(-1, 1)), 
              np.array(mean_timesteps).reshape(-1, 1), np.array(mean_rewards).reshape(-1, 1), 
              np.array(start_costs).reshape(-1, 1), 
              np.array(shut_costs).reshape(-1, 1), 
              np.array(prod_costs).reshape(-1, 1),
              np.array(total_costs).reshape(-1, 1),

              np.array(start_emiss).reshape(-1, 1),
              np.array(shut_emiss).reshape(-1, 1),
              np.array(prod_emiss).reshape(-1, 1),
              np.array(total_emiss).reshape(-1, 1)
          ], axis = 1),
                                             columns=["Episode", "Timesteps", "Rewards", "Startup Cost",
                                             "Shutdown Cost", "Production Cost", "Total Cost",
                                             "Startup Emission", "Shutup Emission", 
                                             "Production Emission", "Total Emission"])
      elif self.environ.n_objs == "tri": 
          training_results_df = pd.DataFrame(np.concatenate([
              np.array(np.arange(1, num_episodes + 1).reshape(-1, 1)), 
              np.array(mean_timesteps).reshape(-1, 1), np.array(mean_rewards).reshape(-1, 1), 
              np.array(start_costs).reshape(-1, 1), 
              np.array(shut_costs).reshape(-1, 1), 
              np.array(prod_costs).reshape(-1, 1),
              np.array(total_costs).reshape(-1, 1),

              np.array(start_emis1s).reshape(-1, 1),
              np.array(shut_emis1s).reshape(-1, 1),
              np.array(prod_emis1s).reshape(-1, 1),
              np.array(total_emis1s).reshape(-1, 1),
              
              np.array(start_emis2s).reshape(-1, 1),
              np.array(shut_emis2s).reshape(-1, 1),
              np.array(prod_emis2s).reshape(-1, 1),
              np.array(total_emis2s).reshape(-1, 1)
          ], axis = 1),
                                             columns=["Episode", "Timesteps", "Rewards", "Startup Cost",
                                             "Shutdown Cost", "Production Cost", "Total Cost",
                                             "Startup Emission1", "Shutup Emission1", 
                                             "Production Emission1", "Total Emission1",
                                             "Startup Emission2", "Shutup Emission2", 
                                             "Production Emission2", "Total Emission2"                                                     
                                                     ])        
    
      training_results_df[["Timesteps"]] = training_results_df[["Timesteps"]].astype(int)  
      end_time = default_timer()
      print(f"Training completed: time = {(end_time - start_time):.3f} seconds.!") 
      return training_results_df
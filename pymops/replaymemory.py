import numpy as np

class ReplayMemory:
  def __init__(self, environ, buffer_size = 32):
    self.buffer_size = buffer_size
    self.state_vec, _ = environ.reset()
    self.input_dim = len(self.state_vec)
    self.action_dim = environ.n_units
    
    self.states_buffer_mat = np.zeros((self.buffer_size, self.input_dim))
    self.actions_buffer_mat = np.zeros((self.buffer_size, self.action_dim))
    self.rewards_buffer_vec = np.zeros(self.buffer_size)
    self.next_states_buffer_mat = np.zeros((self.buffer_size, self.input_dim))
    
    self.num_memory_used = 0
    
  def store(self, state, action, reward, next_state):
    idx = self.num_memory_used % self.buffer_size
    
    self.states_buffer_mat[idx] = state
    self.actions_buffer_mat[idx] = action
    self.rewards_buffer_vec[idx] = reward
    self.next_states_buffer_mat[idx] = next_state
    
    self.num_memory_used += 1

  def sample(self, batch_size):
    idx = np.random.choice(np.arange(self.buffer_size), size = batch_size, replace = False)
    
    experiences_dict = {
        'states_mat': self.states_buffer_mat[idx],
        'actions_mat': self.actions_buffer_mat[idx],
        'rewards_vec': self.rewards_buffer_vec[idx],
        'next_states_mat': self.next_states_buffer_mat[idx]
    }
    return experiences_dict
    
  def is_full(self):
    return (self.num_memory_used >= self.buffer_size)
    
  def reset(self):
    self.num_memory_used = 0 
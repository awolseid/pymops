import torch
import torch.nn as nn

class DQNet(nn.Module):
    def __init__(self, environ, hidden_nodes = 32):
        super(DQNet, self).__init__()
        self.environ = environ
        self.state_vec, _ = self.environ.reset()
        self.input_nodes = len(self.state_vec)
        self.output_nodes = 2 * self.environ.n_units
        
        self.model = nn.Sequential(
            nn.Linear(self.input_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, self.output_nodes),
            nn.ReLU()
        )
    
    def forward(self, state_vec):
        return self.model(torch.as_tensor(state_vec).float())
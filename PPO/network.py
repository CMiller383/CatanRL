import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for the PPO agent
    """
    def __init__(self, input_dim, action_dim, hidden_size=256):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        base_output = self.base(x)
        
        # Get action probabilities
        action_logits = self.actor(base_output)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Get state value
        state_value = self.critic(base_output)
        
        return action_probs, state_value
    
    def evaluate(self, x, action):
        action_probs, state_value = self.forward(x)
        
        # Get action log probabilities
        action_log_probs = torch.log(action_probs + 1e-10)
        action_log_probs = action_log_probs.gather(1, action)
        
        # Calculate entropy
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1).mean()
        
        return action_log_probs, state_value, entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _init_weights(m):
    """
    Initialize weights for Linear and Norm layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)


class CatanNetwork(nn.Module):
    """
    Neural network for AlphaZero-style Catan agent.
    Takes a state representation and outputs:
    1. Policy (action logits)
    2. Value (expected outcome)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CatanNetwork, self).__init__()
        
        # Shared representation layers
        self.representation = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head - outputs action logits
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Value head - outputs state value estimate
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        # Initialize weights
        self.apply(_init_weights)
    
    def forward(self, x):
        representation = self.representation(x)
        policy_logits = self.policy_head(representation)
        value = self.value_head(representation)
        return policy_logits, value
    
    def predict(self, state, valid_actions=None):
        # Ensure tensor and batch dim
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        policy_logits, value = self(state)
        
        if valid_actions is not None:
            # Create a mask of -inf on same device/dtype
            mask = torch.full_like(policy_logits, -float('inf'))
            if isinstance(valid_actions, list):
                mask[0, valid_actions] = 0.0
            else:
                valid = valid_actions.to(device=policy_logits.device)
                mask = torch.where(valid, torch.tensor(0.0, device=policy_logits.device), torch.tensor(-float('inf'), device=policy_logits.device))
            policy_logits = policy_logits + mask
        
        policy = F.softmax(policy_logits, dim=-1)
        return policy.squeeze(0), value.item()
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))


class ResBlock(nn.Module):
    """
    Residual block for deeper network architectures.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.norm2(self.fc2(x))
        x = F.relu(x + residual)
        return x


class DeepCatanNetwork(nn.Module):
    """
    Deeper neural network architecture with residual blocks.
    Can be used for larger state representations and more complex learning.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_blocks=10):
        super(DeepCatanNetwork, self).__init__()
        
        # Initial processing
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        # Initialize weights
        self.apply(_init_weights)
    
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
    
    def predict(self, state, valid_actions=None):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        policy_logits, value = self(state)
        
        if valid_actions is not None:
            mask = torch.full_like(policy_logits, -float('inf'))
            if isinstance(valid_actions, list):
                mask[0, valid_actions] = 0.0
            else:
                valid = valid_actions.to(device=policy_logits.device)
                mask = torch.where(valid, torch.tensor(0.0, device=policy_logits.device), torch.tensor(-float('inf'), device=policy_logits.device))
            policy_logits = policy_logits + mask
        
        policy = F.softmax(policy_logits, dim=-1)
        return policy.squeeze(0), value.item()

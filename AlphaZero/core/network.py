"""
Neural network architecture for AlphaZero Catan.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CatanNetwork(nn.Module):
    """
    Neural network for AlphaZero-style Catan agent.
    Takes a state representation and outputs:
    1. Policy (action probabilities)
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
        
        # Policy head - outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
            # Note: we apply softmax after masking invalid actions
        )
        
        # Value head - outputs state value estimate
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: State representation tensor
            
        Returns:
            policy_logits: Unnormalized action probabilities
            value: Predicted state value
        """
        # Get shared representation
        representation = self.representation(x)
        
        # Get policy logits (unnormalized action probabilities)
        policy_logits = self.policy_head(representation)
        
        # Get value prediction
        value = self.value_head(representation)
        
        return policy_logits, value
    
    def predict(self, state, valid_actions=None):
        """
        Make a prediction for a given state
        
        Args:
            state: State tensor
            valid_actions: List of valid action indices, or mask of valid actions
            
        Returns:
            policy: Normalized action probabilities (masked for valid actions)
            value: Predicted state value
        """
        # Ensure state is a torch tensor with batch dimension
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Forward pass
        policy_logits, value = self(state)
        
        # Mask invalid actions if provided
        if valid_actions is not None:
            # If valid_actions is a list of indices
            if isinstance(valid_actions, list):
                # Create a mask of invalid actions
                mask = torch.ones_like(policy_logits) * float('-inf')
                mask[0, valid_actions] = 0
            else:
                # Assume it's already a boolean mask
                mask = torch.where(valid_actions, 0.0, float('-inf'))
            
            # Apply mask
            policy_logits = policy_logits + mask
            
        # Convert logits to probabilities
        policy = F.softmax(policy_logits, dim=1)
        
        return policy.squeeze(0), value.item()
    
    def save(self, path):
        """Save the model to the specified path"""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load the model from the specified path"""
        self.load_state_dict(torch.load(path))


class ResBlock(nn.Module):
    """
    Residual block for deeper network architectures.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Initial layer
        x = self.input_layer(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
    
    def predict(self, state, valid_actions=None):
        """Same prediction interface as CatanNetwork"""
        # Ensure state is a torch tensor with batch dimension
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Forward pass
        policy_logits, value = self(state)
        
        # Mask invalid actions if provided
        if valid_actions is not None:
            # If valid_actions is a list of indices
            if isinstance(valid_actions, list):
                # Create a mask of invalid actions
                mask = torch.ones_like(policy_logits) * float('-inf')
                mask[0, valid_actions] = 0
            else:
                # Assume it's already a boolean mask
                mask = torch.where(valid_actions, 0.0, float('-inf'))
            
            # Apply mask
            policy_logits = policy_logits + mask
            
        # Convert logits to probabilities
        policy = F.softmax(policy_logits, dim=1)
        
        return policy.squeeze(0), value.item()
import torch
import numpy as np


class Memory:
    """
    Memory buffer for storing trajectories
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.action_probs = []
        self.values = []
        self.dones = []
        
    def add(self, state, action, reward, next_state, action_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.action_probs.append(action_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.action_probs = []
        self.values = []
        self.dones = []
    
    def get_all(self):
        return (
            torch.tensor(np.array(self.states), dtype=torch.float32),
            torch.tensor(np.array(self.actions), dtype=torch.long),
            torch.tensor(np.array(self.rewards), dtype=torch.float32),
            torch.tensor(np.array(self.next_states), dtype=torch.float32),
            torch.tensor(np.array(self.action_probs), dtype=torch.float32),
            torch.tensor(np.array(self.values), dtype=torch.float32),
            torch.tensor(np.array(self.dones), dtype=torch.bool)
        )
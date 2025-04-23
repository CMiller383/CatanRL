# agent/base.py
from enum import Enum
from AlphaZero.utils.config import get_config, get_eval_config
import torch
class AgentType(Enum):
    HUMAN = 0
    RANDOM = 1
    HEURISTIC = 2
    ALPHAZERO = 3

    # Future agent types here

class Agent:
    """Base agent class that all agents should inherit from"""
    def __init__(self, player_id, agent_type):
        self.player_id = player_id
        self.agent_type = agent_type
    
    def is_human(self):
        return self.agent_type == AgentType.HUMAN
    
    def is_AlphaZero(self):
        return self.agent_type == AgentType.ALPHAZERO
    
    def get_action(self, game_state):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_action method")
    

def create_agent(player_id, agent_type, model_path=None):
    """Factory function to create appropriate agent type"""
    if agent_type == AgentType.HUMAN:
        from agent.human_agent import HumanAgent
        return HumanAgent(player_id)
    elif agent_type == AgentType.RANDOM:
        from agent.random_agent import RandomAgent
        return RandomAgent(player_id)
    elif agent_type == AgentType.HEURISTIC:
        from agent.simple_heuristic_agent import SimpleHeuristicAgent
        return SimpleHeuristicAgent(player_id)
    elif agent_type == AgentType.ALPHAZERO:
        import os
        if model_path:
            # Load a trained model
            from AlphaZero.utils.alphazero_utils import load_alphazero_agent
            agent = load_alphazero_agent(player_id, model_path, config=get_eval_config())
            if get_eval_config()['use_placement_network']:
                # Load the placement network if specified
                placement_model_path = 'models/placement_model.pth'
                if os.path.exists(placement_model_path):
                    agent.load_placement_network(placement_model_path)
                else:
                    print(f"Placement model not found at {placement_model_path}.")
            return agent
        
                
        else:
            # Create a new untrained agent for training
            from AlphaZero.agent.alpha_agent import create_alpha_agent
            return create_alpha_agent(player_id)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
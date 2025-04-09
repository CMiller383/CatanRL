# agent/base.py
from enum import Enum

class AgentType(Enum):
    HUMAN = 0
    RANDOM = 1
    HEURISTIC = 2

    # Future agent types here

class Agent:
    """Base agent class that all agents should inherit from"""
    def __init__(self, player_id, agent_type):
        self.player_id = player_id
        self.agent_type = agent_type
    
    def is_human(self):
        return self.agent_type == AgentType.HUMAN
    
    def get_action(self, game_state):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_action method")

def create_agent(player_id, agent_type):
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
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
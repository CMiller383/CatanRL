# agent/base.py
from enum import Enum

class AgentType(Enum):
    HUMAN = 0
    RANDOM = 1

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

class HumanAgent(Agent):
    """Human player agent - actions are driven by UI input"""
    def __init__(self, player_id):
        super().__init__(player_id, AgentType.HUMAN)
    
    def get_action(self, game_state):
        # Human actions are handled through the UI, not here
        return None

def create_agent(player_id, agent_type):
    """Factory function to create appropriate agent type"""
    if agent_type == AgentType.HUMAN:
        return HumanAgent(player_id)
    elif agent_type == AgentType.RANDOM:
        # Import here to avoid circular imports
        from agent.random_agent import RandomAgent
        return RandomAgent(player_id)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
from .base import Agent, AgentType
class HumanAgent(Agent):
    """Human player agent - actions are driven by UI input"""
    def __init__(self, player_id):
        super().__init__(player_id, AgentType.HUMAN)
    
    def get_action(self, game_state):
        # Human actions are handled through the UI, not here
        return None

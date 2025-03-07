# agent/random_agent.py
import random
from .base import Agent, AgentType

class RandomAgent(Agent):
    """Random agent that makes valid random moves"""
    
    def __init__(self, player_id):
        super().__init__(player_id, AgentType.RANDOM)
    
    def get_initial_settlement(self, game_logic):
        """Choose a random valid spot for initial settlement placement"""
        valid_spots = []
        for spot_id, spot in game_logic.board.spots.items():
            if game_logic.is_valid_initial_settlement(spot_id):
                valid_spots.append(spot_id)
        
        if valid_spots:
            return random.choice(valid_spots)
        return None
    
    def get_initial_road(self, game_logic, settlement_id):
        """Choose a random valid road connected to the settlement"""
        valid_roads = []
        for road_id, road in game_logic.board.roads.items():
            if game_logic.is_valid_initial_road(road_id, settlement_id):
                valid_roads.append(road_id)
        
        if valid_roads:
            return random.choice(valid_roads)
        return None
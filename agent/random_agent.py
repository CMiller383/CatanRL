# agent/random_agent.py
import random

from game.action import Action
from game.rules import is_valid_initial_road
from game.setup import is_valid_initial_settlement
from .base import Agent, AgentType
from game.enums import ActionType, Resource

class RandomAgent(Agent):
    """Random agent that makes valid random moves"""
    
    def __init__(self, player_id):
        super().__init__(player_id, AgentType.RANDOM)
    
    def get_initial_settlement(self, state):
        """Choose a random valid spot for initial settlement placement"""
        valid_spots = []
        for spot_id, spot in state.board.spots.items():
            if is_valid_initial_settlement(state, spot_id):
                valid_spots.append(spot_id)
        
        if valid_spots:
            return random.choice(valid_spots)
        return None
    
    def get_initial_road(self, state, settlement_id):
        """Choose a random valid road connected to the settlement"""
        valid_roads = []
        for road_id, road in state.board.roads.items():
            if is_valid_initial_road(state, road_id, settlement_id):
                valid_roads.append(road_id)
        
        if valid_roads:
            return random.choice(valid_roads)
        return None
    
    def get_action(self, state):
        possible_actions = list(state.possible_actions)

        if not possible_actions:
            # If no moves, force an end turn
            return Action(ActionType.END_TURN)
        
        return random.choice(possible_actions)
    
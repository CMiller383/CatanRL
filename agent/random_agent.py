# agent/random_agent.py
import random

from game.rules import is_valid_initial_road
from game.setup import is_valid_initial_settlement
from .base import Agent, AgentType
from game.enums import Resource

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
        print("getting action")
        possible_moves = list(state.possible_actions)
        print("possible moves")
        print(possible_moves)

        if not possible_moves:
            # If no moves, force an end turn
            return "end_turn"
        
        if "roll_dice" in possible_moves:
            return "roll_dice"
        
        if state.awaiting_robber_placement:
            valid_hexes = [hex_id for hex_id in state.board.hexes.keys()
                           if hex_id != state.robber_hex_id]
            if valid_hexes:
                chosen_hex = random.choice(valid_hexes)
                return ("move_robber", chosen_hex)
            
        if state.awaiting_resource_selection:
            resources = [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]
            chosen_resource = random.choice(resources)
            return ("select_resource", chosen_resource)
        
        if state.awaiting_monopoly_selection:
            resources = [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]
            chosen_resource = random.choice(resources)
            return ("select_monopoly", chosen_resource)
        
        if 0 < state.road_building_roads_placed < 2:
            free_road_moves = [move for move in possible_moves 
                               if isinstance(move, tuple) and move[0] == "free_road"]
            if free_road_moves:
                return random.choice(free_road_moves)
        
        # Otherwise, choose from build moves (filter out roll_dice and end_turn)
        build_moves = [move for move in possible_moves if move not in ("end_turn", "roll_dice")]
        if build_moves:
            return random.choice(build_moves)
        
        # Fallback: if no build moves, end turn
        if "end_turn" in possible_moves:
            return "end_turn"
        
        return None
# agent/random_agent.py
import random
from .base import Agent, AgentType
from game.resource import Resource



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
    
    def get_move(self, game_logic):
        possible_moves = list(game_logic.possible_moves)

        if not possible_moves:
            # If no moves, force an end turn
            return "end_turn"
        
        if "roll_dice" in possible_moves:
            return "roll_dice"
        
        if game_logic.awaiting_robber_placement:
            valid_hexes = [hex_id for hex_id in game_logic.board.hexes.keys()
                           if hex_id != game_logic.robber_hex_id]
            if valid_hexes:
                chosen_hex = random.choice(valid_hexes)
                return ("move_robber", chosen_hex)
        if game_logic.awaiting_resource_selection:
            resources = [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]
            chosen_resource = random.choice(resources)
            return ("select_resource", chosen_resource)
        if game_logic.awaiting_monopoly_selection:
            resources = [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]
            chosen_resource = random.choice(resources)
            return ("select_monopoly", chosen_resource)
        if 0 < game_logic.road_building_roads_placed < 2:
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
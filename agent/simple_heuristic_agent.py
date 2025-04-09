import random

from game.rules import is_valid_initial_road
from game.setup import is_valid_initial_settlement
from .base import Agent, AgentType
from game.enums import Resource

class SimpleHeuristicAgent(Agent):
    """A simple heuristic agent:
       - For initial settlement placement, it picks the valid spot whose adjacent hexes have the highest sum of dice numbers (ignoring desert).
       - In regular play, it prioritizes upgrading settlements, then building roads, then buying a development card, and finally ending its turn.
    """
    def __init__(self, player_id):
        super().__init__(player_id, AgentType.HEURISTIC)
    
    def get_initial_settlement(self, state):
        # Mapping from dice number to pip count
        pip_values = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        best_spot = None
        best_score = -1
        for spot_id, spot in state.board.spots.items():
            if is_valid_initial_settlement(state, spot_id):
                score = 0
                for hex_id in spot.adjacent_hex_ids:
                    hex_obj = state.board.get_hex(hex_id)
                    if hex_obj.resource == Resource.DESERT:
                        continue
                    pip = pip_values.get(hex_obj.number, 0)
                    score += pip
                    
                if score > best_score:
                    best_score = score
                    best_spot = spot_id
        return best_spot

    
    def get_initial_road(self, state, settlement_id):
        valid_roads = [
            road_id for road_id, road in state.board.roads.items() 
            if is_valid_initial_road(state, road_id, settlement_id)
        ]
        if valid_roads:
            return random.choice(valid_roads)
        return None

    def get_action(self, state):
        # If dice haven't been rolled yet, roll them.
        if "roll_dice" in state.possible_actions:
            return "roll_dice"
        
        # If a robber move is required, pick one.
        if state.awaiting_robber_placement:
            valid_hexes = [hex_id for hex_id in state.board.hexes.keys()
                           if hex_id != state.robber_hex_id]
            if valid_hexes:
                return ("move_robber", random.choice(valid_hexes))
        
        # If resource selection is required (Year of Plenty), pick one at random.
        if state.awaiting_resource_selection:
            resources = [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]
            return ("select_resource", random.choice(resources))
        
        # If monopoly selection is required, choose a resource at random.
        if state.awaiting_monopoly_selection:
            resources = [Resource.WOOD, Resource.BRICK, Resource.WHEAT, Resource.SHEEP, Resource.ORE]
            return ("select_monopoly", random.choice(resources))
        
        # During road building via a dev card, if free road moves exist.
        if 0 < state.road_building_roads_placed < 2:
            free_road_moves = [
                move for move in state.possible_actions
                if isinstance(move, tuple) and move[0] == "free_road"
            ]
            if free_road_moves:
                return random.choice(free_road_moves)
        
        # Regular build moves in order of priority:
        # 1. Upgrade a settlement to a city.
        upgrade_moves = [
            move for move in state.possible_actions
            if isinstance(move, tuple) and move[0] == "upgrade_city"
        ]
        if upgrade_moves:
            return upgrade_moves[0]
        
        # 2. Build a road.
        road_moves = [
            move for move in state.possible_actions
            if isinstance(move, tuple) and move[0] == "road"
        ]
        if road_moves:
            return random.choice(road_moves)
        
        # 3. Buy a development card.
        if "buy_dev_card" in state.possible_actions:
            return "buy_dev_card"
        
        # 4. If none of the above, try ending the turn.
        if "end_turn" in state.possible_actions:
            return "end_turn"
        
        # pick a random move worst case
        moves = list(state.possible_actions)
        if moves:
            return random.choice(moves)
        return None

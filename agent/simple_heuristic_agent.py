import random
from collections import Counter
from game.action import Action
from game.rules import is_valid_initial_road
from game.setup import is_valid_initial_settlement
from .base import Agent, AgentType
from game.enums import ActionType, Resource

class SimpleHeuristicAgent(Agent):
    def __init__(self, player_id):
        super().__init__(player_id, AgentType.HEURISTIC)

    def get_initial_settlement(self, state):
        pip_values = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        best_spot = None
        best_score = -1

        for spot_id, spot in state.board.spots.items():
            if not is_valid_initial_settlement(state, spot_id):
                continue

            score = 0
            resources = set()
            for hex_id in spot.adjacent_hex_ids:
                hex_obj = state.board.get_hex(hex_id)
                if hex_obj.resource != Resource.DESERT:
                    pip = pip_values.get(hex_obj.number, 0)
                    score += pip
                    resources.add(hex_obj.resource)

            diversity_bonus = len(resources) * 0.5
            total_score = score + diversity_bonus

            if total_score > best_score:
                best_score = total_score
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

    def evaluate_action(self, action, state):
        if action.type == ActionType.UPGRADE_TO_CITY:
            return 10
        if action.type == ActionType.BUILD_SETTLEMENT:
            return 9
        if action.type == ActionType.BUILD_ROAD:
            # Prefer roads leading to buildable settlement spots
            player = state.get_current_player()
            for spot_id, spot in state.board.spots.items():
                road = state.board.roads[action.payload]
                if spot.player_idx is None and spot_id in ([road.spot1_id, road.spot2_id]):
                    return 6
            return 4
        if action.type == ActionType.BUY_DEV_CARD:
            return 3
        if action.type == ActionType.END_TURN:
            return 0
        return 1

    def get_action(self, state):
        # Handle special conditions
        if state.awaiting_robber_placement:
            valid_hexes = [hex_id for hex_id in state.board.hexes.keys()
                           if hex_id != state.robber_hex_id]
            if valid_hexes:
                return Action(ActionType.MOVE_ROBBER, payload=random.choice(valid_hexes))

        if state.awaiting_resource_selection:
            return Action(ActionType.SELECT_YEAR_OF_PLENTY_RESOURCE, payload=random.choice(list(Resource)))

        if state.awaiting_monopoly_selection:
            return Action(ActionType.SELECT_MONOPOLY_RESOURCE, payload=random.choice(list(Resource)))

        if 0 < state.road_building_roads_placed < 2:
            free_roads = [a for a in state.possible_actions
                          if a.type == ActionType.PLACE_FREE_ROAD]
            if free_roads:
                return random.choice(free_roads)

        # Score all actions and pick the best
        best_score = -1
        best_action = None
        for action in state.possible_actions:
            score = self.evaluate_action(action, state)
            if score > best_score:
                best_score = score
                best_action = action

        # Fallback: pick random legal action
        if best_action:
            return best_action

        if state.possible_actions:
            return random.choice(list(state.possible_actions))

        return None

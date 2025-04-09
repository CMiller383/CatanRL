"""
Maps between game actions and neural network action indices.
"""
import numpy as np
from game.action import Action
from game.enums import ActionType

class ActionMapper:
    """
    Maps between Catan game actions and neural network action indices.
    Provides bidirectional conversion between Action objects and integer indices.
    """
    def __init__(self, max_actions=200):
        """
        Initialize the action mapper
        
        Args:
            max_actions: Maximum number of possible actions
        """
        self.max_actions = max_actions
        self.action_space = self._build_action_space()
    
    def _build_action_space(self):
        """
        Build a structured action space for mapping
        
        Returns:
            action_space: Dictionary mapping action types to index ranges
        """
        # Define index ranges for different action types
        action_space = {
            # Simple actions without payloads
            ActionType.ROLL_DICE: 0,
            ActionType.END_TURN: 1,
            ActionType.BUY_DEV_CARD: 2,
            ActionType.PLAY_KNIGHT_CARD: 3,
            ActionType.PLAY_ROAD_BUILDING_CARD: 4,
            ActionType.PLAY_YEAR_OF_PLENTY_CARD: 5,
            ActionType.PLAY_MONOPOLY_CARD: 6,
            
            # Actions with spot_id payload (range: 7-61)
            "build_settlement": (7, 61),  # 54 possible spots
            "upgrade_city": (7, 61),      # Same range as settlements
            
            # Actions with road_id payload (range: 62-134)
            "build_road": (62, 134),      # 72 possible roads
            "place_free_road": (62, 134), # Same range as regular roads
            
            # Actions with hex_id payload (range: 135-154)
            "move_robber": (135, 154),    # 19 possible hexes
            
            # Actions with resource selection (range: 155-159)
            "select_resource": (155, 159), # 5 resource types
            
            # Actions with player selection (range: 160-163)
            "steal_resource": (160, 163),  # 4 possible players
        }
        
        return action_space
    
    def action_to_index(self, action):
        """
        Convert an Action object to an index for the neural network
        
        Args:
            action: Action object
            
        Returns:
            index: Integer index representing the action
        """
        # Handle simple actions without payloads
        if action.type in [ActionType.ROLL_DICE, ActionType.END_TURN, 
                          ActionType.BUY_DEV_CARD, ActionType.PLAY_KNIGHT_CARD,
                          ActionType.PLAY_ROAD_BUILDING_CARD, ActionType.PLAY_YEAR_OF_PLENTY_CARD,
                          ActionType.PLAY_MONOPOLY_CARD]:
            return self.action_space[action.type]
        
        # Handle settlement and city actions (with spot_id payload)
        elif action.type == ActionType.BUILD_SETTLEMENT:
            start, _ = self.action_space["build_settlement"]
            # Map spot_id to index range, with bounds checking
            offset = min(action.payload, 54) - 1  # Assuming spot_ids start at 1
            return start + offset
        
        elif action.type == ActionType.UPGRADE_TO_CITY:
            start, _ = self.action_space["upgrade_city"]
            offset = min(action.payload, 54) - 1
            return start + offset
        
        # Handle road actions (with road_id payload)
        elif action.type == ActionType.BUILD_ROAD:
            start, _ = self.action_space["build_road"]
            offset = min(action.payload, 72) - 1  # Assuming road_ids start at 1
            return start + offset
        
        elif action.type == ActionType.PLACE_FREE_ROAD:
            start, _ = self.action_space["place_free_road"]
            offset = min(action.payload, 72) - 1
            return start + offset
        
        # Handle robber movement (with hex_id payload)
        elif action.type == ActionType.MOVE_ROBBER:
            start, _ = self.action_space["move_robber"]
            offset = min(action.payload, 19) - 1  # Assuming hex_ids start at 1
            return start + offset
        
        # Handle resource selection
        elif action.type in [ActionType.SELECT_YEAR_OF_PLENTY_RESOURCE, ActionType.SELECT_MONOPOLY_RESOURCE]:
            start, _ = self.action_space["select_resource"]
            # Map resource enum to index, ensuring bounds
            resource_idx = min(action.payload.value, 5) if hasattr(action.payload, 'value') else 0
            return start + resource_idx
        
        # Handle stealing (with player_id payload)
        elif action.type == ActionType.STEAL:
            start, _ = self.action_space["steal_resource"]
            offset = min(action.payload, 4)  # Player index 0-3
            return start + offset
        
        # Fallback for unrecognized actions
        return self.max_actions - 1  # Last index as fallback
    
    def index_to_action(self, index, game_state=None):
        """
        Convert an index to an Action object
        
        Args:
            index: Integer index
            game_state: Optional game state for context
            
        Returns:
            action: Action object
        """
        # Handle simple actions without payloads
        if index == self.action_space[ActionType.ROLL_DICE]:
            return Action(ActionType.ROLL_DICE)
        elif index == self.action_space[ActionType.END_TURN]:
            return Action(ActionType.END_TURN)
        elif index == self.action_space[ActionType.BUY_DEV_CARD]:
            return Action(ActionType.BUY_DEV_CARD)
        elif index == self.action_space[ActionType.PLAY_KNIGHT_CARD]:
            return Action(ActionType.PLAY_KNIGHT_CARD)
        elif index == self.action_space[ActionType.PLAY_ROAD_BUILDING_CARD]:
            return Action(ActionType.PLAY_ROAD_BUILDING_CARD)
        elif index == self.action_space[ActionType.PLAY_YEAR_OF_PLENTY_CARD]:
            return Action(ActionType.PLAY_YEAR_OF_PLENTY_CARD)
        elif index == self.action_space[ActionType.PLAY_MONOPOLY_CARD]:
            return Action(ActionType.PLAY_MONOPOLY_CARD)
        
        # Handle settlement actions
        elif self.action_space["build_settlement"][0] <= index <= self.action_space["build_settlement"][1]:
            spot_id = index - self.action_space["build_settlement"][0] + 1
            return Action(ActionType.BUILD_SETTLEMENT, payload=spot_id)
        
        # Handle city upgrade actions
        elif self.action_space["upgrade_city"][0] <= index <= self.action_space["upgrade_city"][1]:
            spot_id = index - self.action_space["upgrade_city"][0] + 1
            return Action(ActionType.UPGRADE_TO_CITY, payload=spot_id)
        
        # Handle road building actions
        elif self.action_space["build_road"][0] <= index <= self.action_space["build_road"][1]:
            road_id = index - self.action_space["build_road"][0] + 1
            return Action(ActionType.BUILD_ROAD, payload=road_id)
        
        # Handle free road placement
        elif self.action_space["place_free_road"][0] <= index <= self.action_space["place_free_road"][1]:
            road_id = index - self.action_space["place_free_road"][0] + 1
            return Action(ActionType.PLACE_FREE_ROAD, payload=road_id)
        
        # Handle robber movement
        elif self.action_space["move_robber"][0] <= index <= self.action_space["move_robber"][1]:
            hex_id = index - self.action_space["move_robber"][0] + 1
            return Action(ActionType.MOVE_ROBBER, payload=hex_id)
        
        # Handle resource selection
        elif self.action_space["select_resource"][0] <= index <= self.action_space["select_resource"][1]:
            resource_idx = index - self.action_space["select_resource"][0]
            # Map index to Resource enum
            from game.enums import Resource
            resources = [Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE]
            resource = resources[min(resource_idx, len(resources)-1)]
            
            # Check game state to decide which resource selection action to use
            if game_state and game_state.awaiting_monopoly_selection:
                return Action(ActionType.SELECT_MONOPOLY_RESOURCE, payload=resource)
            else:
                return Action(ActionType.SELECT_YEAR_OF_PLENTY_RESOURCE, payload=resource)
        
        # Handle stealing
        elif self.action_space["steal_resource"][0] <= index <= self.action_space["steal_resource"][1]:
            player_idx = index - self.action_space["steal_resource"][0]
            return Action(ActionType.STEAL, payload=player_idx)
        
        # If we get here, the index is invalid or not mapped
        # Return END_TURN as a safe default
        return Action(ActionType.END_TURN)
    
    def get_action_space_size(self):
        """
        Get the total size of the action space
        
        Returns:
            size: Number of possible action indices
        """
        return self.max_actions
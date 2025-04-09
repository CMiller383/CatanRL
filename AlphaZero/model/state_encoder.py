"""
State encoder for Catan game states.
Converts the game state into a format suitable for the neural network.
"""
import numpy as np
import torch
from game.enums import Resource, SettlementType, GamePhase

class StateEncoder:
    """
    Encodes the game state into a tensor representation for the neural network.
    """
    def __init__(self, max_actions=200):
        """
        Initialize the state encoder
        
        Args:
            max_actions: Maximum number of possible actions
        """
        self.max_actions = max_actions
        
        # Calculate the size of the state representation
        self._calculate_state_dimensions()
    
    def _calculate_state_dimensions(self):
        """Calculate the dimensions of the state vector"""
        # This is an estimated size - adjust based on your specific game representation
        self.state_dim = 0
        
        # Player resources (5 types per player * 4 players)
        self.state_dim += 5 * 4
        
        # Board state: for each hex (19 hexes)
        # - Resource type (6 possible types -> one-hot encoding)
        # - Dice number (2-12, normalized)
        # - Robber presence (binary)
        self.state_dim += 19 * (6 + 1 + 1)
        
        # For each spot (54 spots)
        # - Owner (4 players or none -> one-hot encoding)
        # - Settlement type (none, settlement, city -> one-hot encoding)
        self.state_dim += 54 * (5 + 3)
        
        # For each road (72 roads)
        # - Owner (4 players or none -> one-hot encoding)
        self.state_dim += 72 * 5
        
        # Game phase
        self.state_dim += 3  # one-hot for 3 phases
        
        # Current player
        self.state_dim += 4  # one-hot encoding
        
        # Last dice roll
        self.state_dim += 1
        
        # Development cards (simplified)
        self.state_dim += 5 * 4  # 5 types for 4 players
        
        print(f"Estimated state dimension: {self.state_dim}")
    
    def encode_state(self, game_state):
        """
        Convert a game state to a vector representation
        
        Args:
            game_state: The GameState object
            
        Returns:
            state_vector: Numpy array representation of the state
        """
        state = []
        
        # Current player indicator
        current_player_idx = game_state.current_player_idx
        player_one_hot = [0] * 4
        player_one_hot[current_player_idx] = 1
        state.extend(player_one_hot)
        
        # Player resources
        for player in game_state.players:
            for resource in [Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE]:
                state.append(player.resources[resource] / 10.0)  # Normalize resource count
        
        # Game phase
        phase_one_hot = [0] * 3
        phase_one_hot[game_state.current_phase.value] = 1
        state.extend(phase_one_hot)
        
        # Board state - hexes
        for hex_id, hex_obj in game_state.board.hexes.items():
            # Resource type (one-hot encoding)
            resource_one_hot = [0] * 6  # 6 resource types including desert
            resource_index = [r for r in Resource].index(hex_obj.resource)
            resource_one_hot[resource_index] = 1
            state.extend(resource_one_hot)
            
            # Dice number (normalized)
            state.append(hex_obj.number / 12.0 if hex_obj.number > 0 else 0)
            
            # Robber presence
            state.append(1.0 if game_state.robber_hex_id == hex_id else 0.0)
        
        # Board state - spots (settlements/cities)
        for spot_id, spot in game_state.board.spots.items():
            # Owner (one-hot encoding)
            owner_one_hot = [0] * 5  # 4 players + no owner
            if spot.player_idx is not None:
                owner_one_hot[spot.player_idx] = 1
            else:
                owner_one_hot[4] = 1  # No owner
            state.extend(owner_one_hot)
            
            # Settlement type (one-hot encoding)
            settlement_type_one_hot = [0] * 3
            settlement_type_one_hot[spot.settlement_type.value] = 1
            state.extend(settlement_type_one_hot)
        
        # Board state - roads
        for road_id, road in game_state.board.roads.items():
            # Owner (one-hot encoding)
            road_owner_one_hot = [0] * 5  # 4 players + no owner
            if road.owner is not None:
                road_owner_one_hot[road.owner] = 1
            else:
                road_owner_one_hot[4] = 1  # No owner
            state.extend(road_owner_one_hot)
        
        # Last dice roll
        if game_state.dice1_roll and game_state.dice2_roll:
            state.append((game_state.dice1_roll + game_state.dice2_roll) / 12.0)
        else:
            state.append(0.0)
        
        # Development cards (simplified for now)
        for player in game_state.players:
            # Count each type of dev card
            knight_count = sum(1 for card in player.dev_cards if card.card_type.name == "KNIGHT")
            vp_count = sum(1 for card in player.dev_cards if card.card_type.name == "VICTORY_POINT")
            road_count = sum(1 for card in player.dev_cards if card.card_type.name == "ROAD_BUILDING")
            plenty_count = sum(1 for card in player.dev_cards if card.card_type.name == "YEAR_OF_PLENTY")
            monopoly_count = sum(1 for card in player.dev_cards if card.card_type.name == "MONOPOLY")
            
            # Normalize and add to state
            state.extend([
                knight_count / 5.0,
                vp_count / 5.0,
                road_count / 2.0,
                plenty_count / 2.0,
                monopoly_count / 2.0
            ])
        
        # Convert to numpy array
        return np.array(state, dtype=np.float32)
    
    def get_valid_action_mask(self, game_state):
        """
        Create a mask of valid actions
        
        Args:
            game_state: The GameState object
            
        Returns:
            valid_action_mask: Boolean array of valid actions
        """
        # Import action mapper
        from AlphaZero.model.action_mapper import ActionMapper
        
        # Get the action mapper
        action_mapper = ActionMapper(self.max_actions)
        
        # Get possible actions from the game state
        possible_actions = list(game_state.possible_actions)
        
        # Create a mask initialized with all False
        mask = np.zeros(self.max_actions, dtype=bool)
        
        # Set True for valid actions
        for action in possible_actions:
            # Convert action to an index
            action_idx = action_mapper.action_to_index(action)
            if action_idx < self.max_actions:
                mask[action_idx] = True
        
        return mask
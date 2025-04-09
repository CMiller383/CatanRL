"""
AlphaZero-style agent for playing Settlers of Catan.
"""
import random
import torch
import numpy as np
from agent.base import Agent, AgentType
from game.setup import is_valid_initial_settlement
from game.rules import is_valid_initial_road

class AlphaZeroAgent(Agent):
    """
    AlphaZero-style agent that uses MCTS and a neural network to play Catan
    """
    def __init__(self, player_id, network, state_encoder, action_mapper, mcts):
        """
        Initialize the AlphaZero agent
        
        Args:
            player_id: The player ID
            network: Neural network for state evaluation
            state_encoder: State encoder for the network
            action_mapper: Converts between game actions and network indices
            mcts: MCTS instance for action selection
        """
        super().__init__(player_id, AgentType.ALPHAZERO)
        self.network = network
        self.state_encoder = state_encoder
        self.action_mapper = action_mapper
        self.mcts = mcts
        self.training_mode = False  # Whether the agent is in training mode
        
        # Training data collection
        self.game_history = []
    
    def set_training_mode(self, training_mode=True):
        """Set whether the agent is in training mode"""
        self.training_mode = training_mode
        if not training_mode:
            # Clear game history when not in training mode
            self.game_history = []
    
    def get_initial_settlement(self, state):
        """
        Choose a spot for initial settlement placement
        In early development, we'll use a heuristic approach for setup
        
        Args:
            state: The game state
            
        Returns:
            spot_id: ID of the selected spot
        """
        # For simplicity, we'll use a heuristic for initial placement
        # Mapping from dice number to pip count (probability)
        pip_values = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        
        best_spot = None
        best_score = -1
        
        for spot_id, spot in state.board.spots.items():
            if is_valid_initial_settlement(state, spot_id):
                score = 0
                resources = set()
                
                for hex_id in spot.adjacent_hex_ids:
                    hex_obj = state.board.get_hex(hex_id)
                    resources.add(hex_obj.resource)
                    
                    if hex_obj.number > 0:  
                        score += pip_values.get(hex_obj.number, 0)
                
                score += len(resources) * 2
                
                if score > best_score:
                    best_score = score
                    best_spot = spot_id
        
        return best_spot
    
    def get_initial_road(self, state, settlement_id):
        """
        Choose a road connected to the initial settlement
        
        Args:
            state: The game state
            settlement_id: ID of the settlement to connect to
            
        Returns:
            road_id: ID of the selected road
        """
        valid_roads = []
        
        for road_id, road in state.board.roads.items():
            if is_valid_initial_road(state, road_id, settlement_id):
                valid_roads.append(road_id)
        
        if valid_roads:
            # random for now whatever
            return random.choice(valid_roads)
        
        return None
    
    def get_action(self, state):
        """
        Get an action for the current game state
        
        Args:
            state: The game state
            
        Returns:
            action: The selected action
        """
        # If no valid actions, return None
        if not state.possible_actions:
            return None
        
        # If only one valid action, take it (common for must-move situations)
        if len(state.possible_actions) == 1:
            return list(state.possible_actions)[0]
        
        # Use MCTS to find the best action
        action_probs, value_estimate = self.mcts.search(state)
        # print(f"AlphaZero action probabilities: {action_probs}")
        # Record state and policy for training
        if self.training_mode:
            state_tensor = self.state_encoder.encode_state(state)
            self.game_history.append({
                'state': state_tensor,
                'player': state.current_player_idx,
                'action_probs': action_probs,
                'value': value_estimate,
                'reward': None  # need to make this
            })
        
        # Select the action based on the policy
        if self.training_mode and random.random() < 0.1:
            # Occasionally explore random actions during training
            return random.choice(list(state.possible_actions))
        else:
            # Choose the action with the highest probability
            if action_probs:
                action = max(action_probs.items(), key=lambda x: x[1])[0]
                print(f"AlphaZero chose action: {action}")
                return action
            else:
                # Fallback to random action if MCTS failed
                # print(f"MTCS failed, choosing random action")
                return random.choice(list(state.possible_actions))
    
    def record_game_result(self, final_reward):
        """
        Record the final result of the game for all states in the game history
        
        Args:
            final_reward: The final reward for this agent
        """
        if not self.training_mode:
            return
        
        # Update all states with the final reward
        for step in self.game_history:
            step['reward'] = final_reward
        
        # Game history can now be used for training
        # In a full implementation, we would pass this to a training coordinator
    
    def get_game_history(self):
        """Get the recorded game history for training"""
        return self.game_history
    
    def clear_game_history(self):
        """Clear the recorded game history"""
        self.game_history = []


def create_alpha_agent(player_id, state_dim=1000, action_dim=200, hidden_dim=256):
    """
    Factory function to create an AlphaZero agent with initialized components
    
    Args:
        player_id: The player ID
        state_dim: State dimension for the network
        action_dim: Action dimension for the network
        hidden_dim: Hidden dimension for the network
        
    Returns:
        agent: AlphaZeroAgent instance
    """
    # Import required components
    from AlphaZero.core.network import CatanNetwork
    from AlphaZero.model.state_encoder import StateEncoder
    from AlphaZero.model.action_mapper import ActionMapper
    from AlphaZero.core.mcts import MCTS
    
    # Create the network
    network = CatanNetwork(state_dim, action_dim, hidden_dim)
    
    # Create the state encoder
    state_encoder = StateEncoder(max_actions=action_dim)
    
    # Create the action mapper
    action_mapper = ActionMapper(max_actions=action_dim)
    
    # Create the MCTS
    mcts = MCTS(network, state_encoder, action_mapper)
    
    # Create and return the agent
    return AlphaZeroAgent(player_id, network, state_encoder, action_mapper, mcts)
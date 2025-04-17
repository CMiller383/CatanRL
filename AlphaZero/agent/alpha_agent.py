"""
AlphaZero-style agent for playing Settlers of Catan.
"""
import random
import torch
import numpy as np
from agent.base import Agent, AgentType
from game.setup import is_valid_initial_settlement
from game.rules import is_valid_initial_road
from game.action import Action
from game.enums import ActionType

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
        super().__init__(player_id, AgentType.ALPHAZERO)  # Use HEURISTIC as a placeholder
        self.network = network
        self.state_encoder = state_encoder
        self.action_mapper = action_mapper
        self.mcts = mcts
        self.training_mode = False  # Whether the agent is in training mode
        
        # Training data collection
        self.game_history = []
        
        # Debug flag
        self.debug = True
        self.inactivity_count = 0
    
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
                # Calculate a score based on resource diversity and probability
                score = 0
                resources = set()
                
                for hex_id in spot.adjacent_hex_ids:
                    hex_obj = state.board.get_hex(hex_id)
                    resources.add(hex_obj.resource)
                    
                    # Add score based on dice probability
                    if hex_obj.number > 0:  # Skip desert
                        score += pip_values.get(hex_obj.number, 0)
                
                # Bonus for resource diversity
                score += len(resources) * 2
                
                if score > best_score:
                    best_score = score
                    best_spot = spot_id
        
        if self.debug:
            print(f"AlphaZero chose initial settlement: {best_spot}")
        
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
            # For now, just choose randomly
            road_id = random.choice(valid_roads)
            if self.debug:
                print(f"AlphaZero chose initial road: {road_id}")
            return road_id
        
        if self.debug:
            print("Warning: No valid initial roads found!")
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
            if self.debug:
                print("Warning: No possible actions available!")
            return Action(ActionType.END_TURN)  # End turn if no actions
        
        if self.debug:
            print(f"\nCurrent player: {state.current_player_idx} (AlphaZero is player 0)")
            print(f"Possible actions: {len(state.possible_actions)}")
            for i, act in enumerate(state.possible_actions):
                if i < 5:  # Show only first 5 actions to avoid clutter
                    print(f"  - {act}")
                elif i == 5:
                    print(f"  - ... and {len(state.possible_actions) - 5} more")
            
            # Print player resources
            player = state.get_current_player()
            print(f"Resources: {dict(player.resources)}")
            print(f"Settlements: {player.settlements}")
            print(f"Cities: {player.cities}")
            print(f"Roads: {player.roads}")
            trade_moves = [act for act in state.possible_actions if act.type == ActionType.TRADE_RESOURCES]
            if trade_moves:
                print(f"Available trades: {len(trade_moves)}")
                for trade in trade_moves[:3]:  # Show just a few
                    give, get = trade.payload
                    print(f"  - Trade 4 {give.name} for 1 {get.name}")
        
        # If only one valid action, take it (common for must-move situations)
        if len(state.possible_actions) == 1:
            action = list(state.possible_actions)[0]
            if self.debug:
                print(f"Only one possible action: {action}")
            return action
        
        # FALLBACK: If there are issues with MCTS, fall back to random agent behavior
        if self.inactivity_count > 3:
            if self.debug:
                print("FALLBACK: Using random selection due to past inactivity")
            action = random.choice(list(state.possible_actions))
            self.inactivity_count = 0
            return action
            
        try:
            # Use MCTS to find the best action
            action_probs, value_estimate = self.mcts.search(state)
            
            if self.debug:
                print(f"MCTS value estimate: {value_estimate:.4f}")
                if action_probs:
                    sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
                    print(f"Top actions from MCTS:")
                    for i, (act, prob) in enumerate(sorted_actions[:3]):
                        print(f"  {i+1}. {act} with probability {prob:.4f}")
            
            # Record state and policy for training
            if self.training_mode:
                state_tensor = self.state_encoder.encode_state(state)
                self.game_history.append({
                    'state': state_tensor,
                    'player': state.current_player_idx,
                    'action_probs': action_probs,
                    'value': value_estimate,
                    'reward': None  # To be filled in later
                })
            
            # Select the action based on the policy
            if self.training_mode and random.random() < 0.1:
                # Occasionally explore random actions during training
                action = random.choice(list(state.possible_actions))
                if self.debug:
                    print(f"Exploration mode: randomly selected {action}")
            else:
                # Choose the action with the highest probability
                if action_probs:
                    action = max(action_probs.items(), key=lambda x: x[1])[0]
                    if self.debug:
                        print(f"AlphaZero chose action: {action}")
                else:
                    # Fallback to random action if MCTS failed
                    if self.debug:
                        print("Warning: MCTS returned no action probs, falling back to random")
                    action = random.choice(list(state.possible_actions))
                    # Increment inactivity counter
                    self.inactivity_count += 1
            
            # Verify the action is in possible_actions
            if action not in state.possible_actions:
                if self.debug:
                    print(f"Warning: Selected action {action} not in possible_actions!")
                    action_type_matches = [a for a in state.possible_actions if a.type == action.type]
                    if action_type_matches:
                        print(f"Found {len(action_type_matches)} actions with same type. Using first one.")
                        action = action_type_matches[0]
                    else:
                        print("No action with matching type found. Using random action.")
                        action = random.choice(list(state.possible_actions))
            
            return action
            
        except Exception as e:
            if self.debug:
                print(f"Error in get_action: {e}")
                import traceback
                traceback.print_exc()
            
            # Fallback to random action if there's an error
            self.inactivity_count += 1
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

def create_alpha_agent(player_id, config=None, network=None):
    """
    Factory function to create an AlphaZero agent with initialized components
    
    Args:
        player_id: The player ID
        config: Configuration dictionary with hyperparameters
        network: Optional pre-created network to share (used for self-play with same network)
        
    Returns:
        agent: AlphaZeroAgent instance
    """
    # Import required components
    from AlphaZero.core.network import CatanNetwork
    from AlphaZero.model.state_encoder import StateEncoder
    from AlphaZero.model.action_mapper import ActionMapper
    from AlphaZero.core.mcts import MCTS
    
    # Use defaults if no config provided
    if config is None:
        from AlphaZero.utils.config import get_config
        config = get_config()
    
    # Create or use the provided network
    if network is None:
        network = CatanNetwork(
            state_dim=config.get('state_dim', 992),
            action_dim=config.get('action_dim', 200),
            hidden_dim=config.get('hidden_dim', 256)
        )
    
    # Create the state encoder
    state_encoder = StateEncoder(max_actions=config.get('action_dim', 200))
    
    # Create the action mapper
    action_mapper = ActionMapper(max_actions=config.get('action_dim', 200))
    
    # Create the MCTS with config parameters
    mcts = MCTS(
        network=network,
        state_encoder=state_encoder,
        action_mapper=action_mapper,
        num_simulations=config.get('num_simulations', 100),
        c_puct=config.get('c_puct', 1.5),
        batch_size=config.get('batch_size', 8)
    )
    
    # Create and return the agent
    return AlphaZeroAgent(player_id, network, state_encoder, action_mapper, mcts)

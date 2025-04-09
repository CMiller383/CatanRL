"""
Monte Carlo Tree Search implementation for AlphaZero-style Catan agent.
"""
import math
import numpy as np
import copy
import random
from collections import defaultdict
import time

class MCTSNode:
    """
    Node in the MCTS tree representing a game state
    """
    def __init__(self, game_state, parent=None, prior=0.0, action=None):
        """
        Initialize a new MCTS Node
        
        Args:
            game_state: The game state this node represents
            parent: The parent node
            prior: Prior probability of this action (from policy network)
            action: The action that led to this state
        """
        self.game_state = game_state
        self.parent = parent
        self.prior = prior
        self.action = action
        self.children = {}  # Action -> MCTSNode
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
        # Debug info
        self.ucb_scores = {}  # Stores UCB scores for child selection debugging
    
    def value(self):
        """Get the current value estimate for this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct=1.5):
        """
        Select a child node using the PUCT algorithm
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            action: The selected action
            child: The selected child node
        """
        # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        # Clear previous UCB scores
        self.ucb_scores = {}
        
        for action, child in self.children.items():
            # Exploitation term
            q_value = child.value()
            
            # Exploration term
            u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            
            # Combined score
            score = q_value + u_value
            
            # Store for debugging
            self.ucb_scores[action] = {
                'q_value': q_value,
                'u_value': u_value,
                'ucb_score': score,
                'visit_count': child.visit_count,
                'prior': child.prior
            }
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_priors):
        """
        Expand this node with the given action probabilities
        
        Args:
            action_priors: List of (action, prior) tuples
        """
        self.is_expanded = True
        expansion_count = 0
        
        for action, prior in action_priors:
            if action not in self.children:
                # Create a copy of the game state
                next_state = copy.deepcopy(self.game_state)
                
                # Apply the action to get the next state
                try:
                    success = self._apply_action(next_state, action)
                    
                    if success:
                        # Create a new child node
                        self.children[action] = MCTSNode(
                            game_state=next_state,
                            parent=self,
                            prior=prior,
                            action=action
                        )
                        expansion_count += 1
                except Exception as e:
                    print(f"Error in node expansion: {e} for action {action}")
        
        return expansion_count
    
    def _apply_action(self, state, action):
        """
        Apply an action to a state
        
        Args:
            state: The game state to modify
            action: The action to apply
            
        Returns:
            success: Whether the action was applied successfully
        """
        # Check if the action is in the possible actions for this state
        if action not in state.possible_actions:
            return False
            
        # Apply the action directly
        from game.game_logic import GameLogic
        
        # Create a temporary game logic object to handle the action
        temp_game = GameLogic(state.board)
        temp_game.state = state
        
        # Apply the action
        return temp_game.do_action(action)
    
    def update(self, value):
        """
        Update this node with a value
        
        Args:
            value: The value to update with
        """
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    """
    Monte Carlo Tree Search algorithm for Catan
    """
    def __init__(self, network, state_encoder, action_mapper, num_simulations=800, c_puct=1.5):
        """
        Initialize the MCTS
        
        Args:
            network: Neural network for state evaluation
            state_encoder: Encoder to convert game states to network inputs
            action_mapper: Mapper between actions and indices
            num_simulations: Number of simulations per move
            c_puct: Exploration constant for UCB formula
        """
        self.network = network
        self.state_encoder = state_encoder
        self.action_mapper = action_mapper
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
        # Debug flag
        self.debug = False
    
    def search(self, game_state):
        """
        Perform MCTS search from the given game state
        
        Args:
            game_state: The root game state to search from
            
        Returns:
            action_probs: Action probabilities based on visit counts
            value: Value estimate of the root state
        """
        # Create the root node
        start_time = time.time()
        root = MCTSNode(game_state=copy.deepcopy(game_state))
        
        # First, evaluate the root state
        state_tensor = self.state_encoder.encode_state(game_state)
        valid_action_mask = self.state_encoder.get_valid_action_mask(game_state)
        
        import torch
        state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0)  # Add batch dimension
        
        try:
            # Get policy and value from the network
            with torch.no_grad():
                policy, value = self.network(state_tensor)
                
            # Convert to numpy for processing
            policy = policy[0].numpy()  # Remove batch dimension
            
            # Create action priors for valid actions
            action_priors = []
            for i, valid in enumerate(valid_action_mask):
                if valid:
                    # Convert index back to action
                    action = self.action_mapper.index_to_action(i, game_state)
                    # Check if the action is valid in the current state
                    if action in game_state.possible_actions:
                        action_priors.append((action, policy[i]))
            
            # Expand the root node with these action priors
            expansion_count = root.expand(action_priors)
            
            if self.debug:
                print(f"Root node expanded with {expansion_count} children")
                print(f"Root state value estimate: {value.item():.4f}")
        
        except Exception as e:
            if self.debug:
                print(f"Error in root node evaluation: {e}")
                import traceback
                traceback.print_exc()
            # Return an empty policy if root evaluation fails
            return {}, 0.0
        
        # Run simulations
        simulation_count = 0
        for _ in range(self.num_simulations):
            try:
                success = self._simulate(root)
                if success:
                    simulation_count += 1
            except Exception as e:
                if self.debug:
                    print(f"Error in simulation: {e}")
        
        if self.debug:
            print(f"Completed {simulation_count} successful simulations out of {self.num_simulations} attempts")
            print(f"MCTS search took {time.time() - start_time:.2f} seconds")
        
        # Calculate action probabilities based on visit counts
        action_probs = self._get_action_probs(root)
        
        # Return action probabilities and the value estimate
        return action_probs, root.value()
    
    def _simulate(self, root):
        """
        Run a single MCTS simulation
        
        Args:
            root: The root node to start simulation from
            
        Returns:
            success: Whether the simulation was successful
        """
        # Selection phase: traverse the tree to a leaf node
        node = root
        search_path = [node]
        
        while node.is_expanded and node.children:
            action, node = node.select_child(self.c_puct)
            if node is None:
                if self.debug:
                    print("Warning: select_child returned None")
                return False
            search_path.append(node)
        
        # Expansion and evaluation phase
        # Only expand nodes that are not terminal
        if not self._is_terminal(node.game_state):
            # Encode the state for the neural network
            state_tensor = self.state_encoder.encode_state(node.game_state)
            valid_action_mask = self.state_encoder.get_valid_action_mask(node.game_state)
            
            # Get policy and value from the network
            import torch
            state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0)
            
            try:
                with torch.no_grad():
                    policy, value = self.network(state_tensor)
                
                # Convert to numpy for processing
                policy = policy[0].numpy()
                value = value.item()
                
                # Create action priors for valid actions
                action_priors = []
                for i, valid in enumerate(valid_action_mask):
                    if valid:
                        # Convert index back to action
                        action = self.action_mapper.index_to_action(i, node.game_state)
                        # Only include actions that are valid in the current state
                        if action in node.game_state.possible_actions:
                            action_priors.append((action, policy[i]))
                
                # Expand the node with these action priors
                expansion_count = node.expand(action_priors)
                if self.debug and expansion_count == 0:
                    print(f"Warning: Node expanded with 0 children")
            
            except Exception as e:
                if self.debug:
                    print(f"Error in node evaluation: {e}")
                return False
            
        else:
            # Terminal node
            value = self._get_game_outcome(node.game_state)
        
        # Backpropagation phase
        self._backpropagate(search_path, value)
        return True
    
    def _backpropagate(self, search_path, value):
        """
        Backpropagate the value through the search path
        
        Args:
            search_path: List of nodes in the search path
            value: The value to backpropagate
        """
        # For multi-player Catan, we need to adjust the value for each player's perspective
        for node in reversed(search_path):
            node.update(value)
            # Negate the value for opposing players in a zero-sum perspective
            # For Catan, this might need adjustment for multiple players
            value = -value
    
    def _get_action_probs(self, root):
        """
        Calculate action probabilities based on visit counts
        
        Args:
            root: The root node
            
        Returns:
            action_probs: Dictionary mapping actions to probabilities
        """
        action_probs = {}
        
        # Calculate the sum of visit counts
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits > 0:
            # Calculate probabilities based on visit counts
            for action, child in root.children.items():
                action_probs[action] = child.visit_count / total_visits
                
                if self.debug and child.visit_count > 0:
                    print(f"Action {action} - Visits: {child.visit_count}, Value: {child.value():.4f}, Prob: {action_probs[action]:.4f}")
        
        return action_probs
    
    def _is_terminal(self, game_state):
        """
        Check if a game state is terminal
        
        Args:
            game_state: The game state to check
            
        Returns:
            is_terminal: Whether the state is terminal
        """
        # Check if any player has 10 victory points
        from game.game_state import check_game_over
        if game_state.winner is not None:
            return True
        return check_game_over(game_state)
    
    def _get_game_outcome(self, game_state):
        """
        Get the outcome of a terminal game state
        
        Args:
            game_state: The terminal game state
            
        Returns:
            outcome: The outcome from the current player's perspective
        """
        current_player = game_state.get_current_player()
        if game_state.winner is not None:
            if game_state.winner == current_player.player_idx:
                return 1.0
            else:
                return -1.0
            
        max_points = 0
        winner = None
        
        # Find the player with the most victory points
        for player in game_state.players:
            points = player.victory_points
            if points > max_points:
                max_points = points
                winner = player
        
        # Return outcome from current player's perspective
        if winner.player_idx == current_player.player_idx:
            return 1.0  # Win
        else:
            return -1.0  # Loss
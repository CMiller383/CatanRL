"""
Monte Carlo Tree Search implementation for AlphaZero-style Catan agent.
"""
import math
import numpy as np
import copy
import random
from collections import defaultdict

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
        # UCB forrmula Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) https://www.turing.com/kb/guide-on-upper-confidence-bound-algorithm-in-reinforced-learning
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # Exploitation term
            q_value = child.value()
            
            # Exploration term
            u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            
            # Combined score
            score = q_value + u_value
            
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
        
        for action, prior in action_priors:
            if action not in self.children:
                # Create a copy of the game state
                next_state = copy.deepcopy(self.game_state)
                
                # Apply the action to get the next state
                success = self._apply_action(next_state, action)
                
                if success:
                    # Create a new child node
                    self.children[action] = MCTSNode(
                        game_state=next_state,
                        parent=self,
                        prior=prior,
                        action=action
                    )
    
    def _apply_action(self, state, action):
        """
        Apply an action to a state
        
        Args:
            state: The game state to modify
            action: The action to apply
            
        Returns:
            success: Whether the action was applied successfully
        """
        from game.game_logic import GameLogic
        
        # Create a temporary game logic object to handle the action
        game_logic = GameLogic(state.board)
        game_logic.state = state
        
        # Apply the action
        return game_logic.do_action(action)
    
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
        root = MCTSNode(game_state=copy.deepcopy(game_state))
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # Calculate action probabilities based on visit counts
        action_probs = self._get_action_probs(root)
        
        # Return action probabilities and the value estimate
        return action_probs, root.value()
    
    def _simulate(self, root):
        """
        Run a single MCTS simulation
        
        Args:
            root: The root node to start simulation from
        """
        # Selection phase: traverse the tree to a leaf node
        node = root
        search_path = [node]
        
        while node.is_expanded and node.children:
            action, node = node.select_child(self.c_puct)
            search_path.append(node)
        
        # Check if we've reached a terminal state
        if self._is_terminal(node.game_state):
            # Use the game outcome as the value
            value = self._get_game_outcome(node.game_state)
        else:
            # Expansion phase
            # Encode the state for the neural network
            state_tensor = self.state_encoder.encode_state(node.game_state)
            valid_action_mask = self.state_encoder.get_valid_action_mask(node.game_state)
            
            # Get policy and value from the network
            import torch
            state_tensor = torch.FloatTensor(state_tensor)
            policy, value = self.network.predict(state_tensor, valid_action_mask)
            
            # Convert to numpy for processing
            policy = policy.detach().numpy()
            
            # Create action priors for valid actions
            action_priors = []
            for i, valid in enumerate(valid_action_mask):
                if valid:
                    # Convert index back to action
                    action = self.action_mapper.index_to_action(i, node.game_state)
                    action_priors.append((action, policy[i]))
            
            # Expand the node with these action priors
            node.expand(action_priors)
        
        # Backpropagation phase
        self._backpropagate(search_path, value)
    
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
            # Negate the value for opposing players in a two-player zero-sum perspective
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
        for player in game_state.players:
            if player.victory_points >= 10:
                return True
        
        return False
    
    def _get_game_outcome(self, game_state):
        """
        Get the outcome of a terminal game state
        
        Args:
            game_state: The terminal game state
            
        Returns:
            outcome: The outcome from the current player's perspective
        """
        current_player = game_state.get_current_player()
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
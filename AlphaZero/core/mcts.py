"""
Monte Carlo Tree Search implementation for AlphaZero-style Catan agent.
"""
import math
import numpy as np
import copy
import random
from collections import defaultdict
import time
import torch

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
        self.virtual_loss = 0
        
        # Debug info
        self.ucb_scores = {}  # Stores UCB scores for child selection debugging
    
    def value(self):
        """Get the current value estimate for this node"""
        if self.visit_count == 0:
            return 0.0
        # Apply virtual loss adjustment for better parallelization
        virtual_loss_adjustment = -1.0 * self.virtual_loss / max(1, self.visit_count)
        return self.value_sum / self.visit_count + virtual_loss_adjustment
    
    def add_virtual_loss(self):
        """Add virtual loss to discourage threads from exploring the same nodes"""
        self.virtual_loss += 1
    
    def remove_virtual_loss(self):
        """Remove virtual loss after the evaluation is complete"""
        self.virtual_loss = max(0, self.virtual_loss - 1)
    
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
        
        # Create a single deep copy of the state to reuse for efficiency
        base_state = copy.deepcopy(self.game_state)
        
        for action, prior in action_priors:
            if action not in self.children:
                # Create a lightweight copy of the base state
                next_state = self._light_copy_state(base_state)
                
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
    
    def _light_copy_state(self, state):
        """Create a lightweight copy of state with shared immutable components"""
        # Shallow copy the state
        new_state = copy.copy(state)
        
        # Only deep copy the mutable parts that will change
        new_state.current_player_idx = state.current_player_idx
        new_state.possible_actions = state.possible_actions.copy() if hasattr(state, 'possible_actions') else set()
        
        # Don't copy the board structure - only the ownership attributes
        new_state.board = copy.copy(state.board)
        
        # Shallow copies of players (since we're careful with mutations)
        new_state.players = [copy.copy(player) for player in state.players]
        
        return new_state
    
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
        if hasattr(state, 'possible_actions') and action not in state.possible_actions:
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
    def __init__(self, network, state_encoder, action_mapper, num_simulations=800, c_puct=1.5, batch_size=8):
        """
        Initialize the MCTS
        
        Args:
            network: Neural network for state evaluation
            state_encoder: Encoder to convert game states to network inputs
            action_mapper: Mapper between actions and indices
            num_simulations: Number of simulations per move
            c_puct: Exploration constant for UCB formula
            batch_size: Size of batches for network evaluation
        """
        self.network = network
        self.state_encoder = state_encoder
        self.action_mapper = action_mapper
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        
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
        self.root = root
        
        # First, evaluate the root state
        state_tensor = self.state_encoder.encode_state(game_state)
        valid_action_mask = self.state_encoder.get_valid_action_mask(game_state)
        # print(f"→ valid_action_mask has {valid_action_mask.sum()} Trues out of {len(valid_action_mask)} slots")
        if hasattr(self.network, 'parameters'):
          device_str = str(next(self.network.parameters()).device)
        if self.debug:
            print(f"MCTS using network on device: {device_str}")
        
        # Ensure we're on CUDA if available
        if torch.cuda.is_available() and not device_str.startswith('cuda'):
            if self.debug:
                print(f"Moving network to CUDA during search")
            self.network = self.network.to('cuda')
        try:
            # Get policy and value from the network
            with torch.no_grad():
              # Convert to tensor
              state_tensor = torch.FloatTensor(state_tensor)
              
              # Move to same device as network
              if hasattr(self.network, 'parameters'):
                  device = next(self.network.parameters()).device
                  state_tensor = state_tensor.to(device)
              
              policy, value = self.network(state_tensor.unsqueeze(0))
                
            # Convert to numpy for processing
            policy = policy[0].detach().cpu().numpy()
            
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
        
        # Prepare arrays for batch processing
        states_to_evaluate = []
        nodes_pending_evaluation = []
        
        # Run simulations
        simulation_count = 0
        for _ in range(self.num_simulations):
            # Select a node to evaluate
            node, path = self._select_node_for_evaluation(root)
            if node and node.game_state:
                # If we found a node to evaluate, add it to our batch
                try:
                    states_to_evaluate.append(self.state_encoder.encode_state(node.game_state))
                    nodes_pending_evaluation.append((node, path))
                except Exception as e:
                    if self.debug:
                        print(f"Error encoding state: {e}")
                    # Remove virtual loss
                    for n in path:
                        n.remove_virtual_loss()
                    continue
                
                # If we have enough nodes or this is the last simulation, evaluate them
                if len(states_to_evaluate) >= self.batch_size or _ == self.num_simulations - 1:
                    if states_to_evaluate:
                        # Batch evaluate the states
                        try:
                            # Convert list of arrays to a single numpy array first (fixes warning)
                            states_np_array = np.array(states_to_evaluate, dtype=np.float32)
                            batch_tensor = torch.FloatTensor(states_np_array)
                            if hasattr(self.network, 'parameters'):
                                device = next(self.network.parameters()).device
                                batch_tensor = batch_tensor.to(device)

                            with torch.no_grad():
                                batch_policies, batch_values = self.network(batch_tensor)
                            
                            # Process each result and backpropagate
                            for i, (eval_node, eval_path) in enumerate(nodes_pending_evaluation):
                                try:
                                    # Complete the evaluation with the network results
                                    policy = batch_policies[i].detach().cpu().numpy()
                                    value = batch_values[i].item()
                                    
                                    # Process the evaluation
                                    self._process_evaluation(eval_node, eval_path, policy, value)
                                    simulation_count += 1
                                except Exception as e:
                                    if self.debug:
                                        print(f"Error processing evaluation: {e}")
                                    # Remove virtual loss
                                    for n in eval_path:
                                        n.remove_virtual_loss()
                        except Exception as e:
                            if self.debug:
                                print(f"Error in batch evaluation: {e}")
                            # Remove virtual loss from all nodes
                            for _, path in nodes_pending_evaluation:
                                for n in path:
                                    n.remove_virtual_loss()
                        
                        # Clear batches for next round
                        states_to_evaluate = []
                        nodes_pending_evaluation = []
        
        if self.debug:
            print(f"Completed {simulation_count} successful simulations out of {self.num_simulations} attempts")
            print(f"MCTS search took {time.time() - start_time:.2f} seconds")
        
        # Calculate action probabilities based on visit counts
        action_probs = self._get_action_probs(root)
        
        # Return action probabilities and the value estimate
        return action_probs, root.value()
    
    def _select_node_for_evaluation(self, root):
        """Select a node that needs evaluation and return its search path"""
        node = root
        search_path = [node]
        
        # Add virtual loss during selection to discourage other threads
        node.add_virtual_loss()
        
        # Selection phase - traverse down the tree
        while node.is_expanded and node.children:
            # Check if any children have zero visits - prioritize them
            unexplored = [(a, n) for a, n in node.children.items() if n.visit_count == 0]
            if unexplored:
                action, node = random.choice(unexplored)
            else:
                action, node = node.select_child(self.c_puct)
            
            if node is None:
                # Remove virtual loss and return none
                for n in search_path:
                    n.remove_virtual_loss()
                return None, []
            
            # Add virtual loss
            node.add_virtual_loss()
            search_path.append(node)
            
            # Early pruning - skip nodes in heavily explored paths that have very poor value
            if len(search_path) > 2 and search_path[-2].visit_count > 10:
                if node.visit_count > 3 and node.value() < -0.8:
                    # Remove virtual loss
                    for n in search_path:
                        n.remove_virtual_loss()
                    return None, []  # Skip this path
        
        return node, search_path
    
    def _process_evaluation(self, node, search_path, policy, value):
        """Process network evaluation for a node"""
        try:
            # Get valid actions
            valid_action_mask = self.state_encoder.get_valid_action_mask(node.game_state)
            
            # Create action priors for valid actions
            action_priors = []
            for i, valid in enumerate(valid_action_mask):
                if valid:
                    # Convert index back to action
                    action = self.action_mapper.index_to_action(i, node.game_state)
                    # Check if the action is valid in the current state
                    if action in node.game_state.possible_actions:
                        action_priors.append((action, policy[i]))
            
            # Expand the node with these action priors
            node.expand(action_priors)
            
            # Backpropagate
            self._backpropagate(search_path, value)
        finally:
            # Always remove virtual loss
            for n in search_path:
                n.remove_virtual_loss()
    
    def _backpropagate(self, search_path, value):
        """
        Backpropagate the value through the search path
        
        Args:
            search_path: List of nodes in the search path
            value: The value to backpropagate
        """
        # Get the perspective of the player at the root node
        if not search_path:
            return
            
        root_player_idx = search_path[0].game_state.current_player_idx
        
        for node in reversed(search_path):
            # Get the player of the current node
            node_player_idx = node.game_state.current_player_idx
            
            # Adjust value based on player perspective
            if node_player_idx == root_player_idx:
                # Same player - use the value directly
                node_value = value
            else:
                # Different player - use a transformed value
                # For Catan, we use a softer transformation than pure negation
                # This reflects that in Catan, other players doing well doesn't always
                # mean you're doing poorly to the same degree
                node_value = -value * 0.8  # Dampen the negative effect for opponents
            
            # Update the node with the perspective-adjusted value
            node.update(node_value)
    
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
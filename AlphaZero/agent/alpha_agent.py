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
from game.enums import ActionType, Resource
from game.enums import GamePhase

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
        self.training_mode = True  # Whether the agent is in training mode
        self.temperature = 1.0  # Temperature for action selection
        
        # Training data collection
        self.game_history = []
        
        # Debug flag
        self.debug = False
        self.inactivity_count = 0
    
    def set_training_mode(self, training_mode=True):
        """Set whether the agent is in training mode"""
        self.training_mode = training_mode
        if not training_mode:
            # Clear game history when not in training mode
            self.game_history = []
    
    def _score_spot(self, state, spot_id):
        """
        Heuristic score for placing a settlement on spot_id, now with port logic.
        In SETUP_PHASE_2, this also biases toward resources you lack.
        """
        pip = {2:1,3:2,4:3,5:4,6:5,8:5,9:4,10:3,11:2,12:1}
        # base weight per resource
        res_w = {
            Resource.WHEAT: 1.5,
            Resource.BRICK: 1.3,
            Resource.WOOD: 1.2,
            Resource.ORE:   1.2,
            Resource.SHEEP: 1.0
        }
        diversity_bonus   = 2
        duplicate_penalty = 1
        port_bonus        = 3

        spot = state.board.spots[spot_id]
        nums, ress = [], []
        for hid in spot.adjacent_hex_ids:
            h = state.board.get_hex(hid)
            if h.number > 0:
                nums.append(h.number)
                ress.append(h.resource)

        player_inv = {}
        if state.current_phase == GamePhase.SETUP_PHASE_2:
            player = state.get_current_player()
            player_inv = player.resources  # Resource → count

        score = 0.0
        # 1) pip × base weight, scaled in phase 2 by 1/(1 + have_of_that_resource)
        for n, r in zip(nums, ress):
            base = pip[n] * res_w.get(r, 1.0)
            if player_inv:
                score += base / (1 + player_inv.get(r, 0))
            else:
                score += base

        # 2) diversity bonus
        uniq = set(ress)
        score += diversity_bonus * len(uniq)

        # 3) duplicate penalty
        score -= duplicate_penalty * (len(ress) - len(uniq))

        # 4) expansion potential: count valid initial roads off this spot
        valid_roads = 0
        for rid, road in state.board.roads.items():
            if (road.spot1_id == spot_id or road.spot2_id == spot_id) \
               and is_valid_initial_road(state, rid, spot_id):
                valid_roads += 1
        score += valid_roads

        # 5) port bonus
        if spot.has_port:
            score += port_bonus

        return score

    def get_initial_settlement(self, state):
        """Phase‑aware two‑spot lookahead in SETUP_PHASE_1, single‑spot in PHASE_2."""
        alpha = 0.8  # how much to discount your second spot

        # Phase 2: simple best‐spot fill‑gaps heuristic
        if state.current_phase == GamePhase.SETUP_PHASE_2:
            best, best_s = None, -float('inf')
            for sid, spot in state.board.spots.items():
                if not is_valid_initial_settlement(state, sid):
                    continue
                s = self._score_spot(state, sid)
                if s > best_s:
                    best_s, best = s, sid
            if self.debug:
                print(f"[Phase 2] Chose settlement {best} (score {best_s:.1f})")
            return best

        # Phase 1: two‑spot lookahead
        bestA, best_val = None, -float('inf')
        me = self.player_id

        for A, spotA in state.board.spots.items():
            if not is_valid_initial_settlement(state, A):
                continue

            sA = self._score_spot(state, A)

            # pretend we’ve taken A
            spotA.player_idx = me
            # collect valid second spots
            candidates = [
                B for B, sp in state.board.spots.items()
                if is_valid_initial_settlement(state, B)
            ]
            sB = max((self._score_spot(state, B) for B in candidates), default=0.0)
            # undo
            spotA.player_idx = None

            total = sA + alpha * sB
            if total > best_val:
                best_val, bestA = total, A

        if self.debug:
            print(f"[Phase 1] Chose settlement {bestA} (A + α·B = {best_val:.1f})")
        return bestA

    def get_initial_road(self, state, settlement_id):
        best_road, best_score = None, -float('inf')
        for road_id, road in state.board.roads.items():
            if not is_valid_initial_road(state, road_id, settlement_id):
                continue

            a, b = road.spot1_id, road.spot2_id
            next_spot = b if a == settlement_id else a

            s = self._score_spot(state, next_spot)
            if s > best_score:
                best_score, best_road = s, road_id

        if best_road is None:
            if self.debug:
                print("Warning: no valid initial road found, falling back to random")
            candidates = [
                rid for rid in state.board.roads
                if is_valid_initial_road(state, rid, settlement_id)
            ]
            return random.choice(candidates) if candidates else None

        if self.debug:
            print(f"AlphaZero initial road → road {best_road}  (leads to spot score={best_score:.1f})")
        return best_road

    def improved_action_selection(self, action_probs, state, mcts_root=None):
        """
        Select an action based on MCTS results with configurable selection method
        
        Args:
            action_probs: Dictionary mapping actions to probabilities
            mcts_root: Root node of the MCTS search tree (if available)
            
        Returns:
            The selected action
        """
        # Check if we should use deterministic selection (using visit counts)
        if hasattr(self, 'deterministic') and self.deterministic and mcts_root and mcts_root.children:
            # Select action with highest visit count
            action = max(mcts_root.children.items(), key=lambda x: x[1].visit_count)[0]
            if self.debug:
                print(f"Using deterministic selection (highest visit count): {action}")
            return action
        
        # Otherwise use temperature-based selection
        if self.training_mode and random.random() < 0.1:
            # Occasionally explore random actions during training
            action = random.choice(list(state.possible_actions))
            if self.debug:
                print(f"Exploration mode: randomly selected {action}")
        else:
            # Choose action based on probability distribution and temperature
            if action_probs:
                if hasattr(self, 'temperature') and self.temperature > 0:
                    # Temperature-based sampling
                    actions = list(action_probs.keys())
                    probs = np.array([action_probs[a] for a in actions])
                    
                    # Apply temperature
                    if self.temperature != 1.0:
                        probs = probs ** (1.0 / self.temperature)
                        probs = probs / np.sum(probs)
                    
                    # Sample from the distribution
                    idx = np.random.choice(len(actions), p=probs)
                    action = actions[idx]
                    
                    if self.debug:
                        print(f"Selected {action} using temperature {self.temperature}")
                else:
                    # Greedy selection (equivalent to temperature → 0)
                    action = max(action_probs.items(), key=lambda x: x[1])[0]
                    if self.debug:
                        print(f"Selected {action} (greedy)")
            else:
                # Fallback to random action if MCTS failed
                if self.debug:
                    print("Warning: MCTS returned no action probs, falling back to random")
                action = random.choice(list(state.possible_actions))
                # Increment inactivity counter
                self.inactivity_count += 1
        
        return action
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
              print("❯❯❯ MCTS returned", len(action_probs), "actions with total mass",
                    sum(action_probs.values()), action_probs)
            mcts_root = None
            if hasattr(self.mcts, 'root'):
                mcts_root = self.mcts.root
            action = self.improved_action_selection(action_probs, state, mcts_root)
            
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
                # print(state_tensor)
                self.game_history.append({
                    'state': state_tensor,
                    'player': state.current_player_idx,
                    'action_probs': action_probs,
                    'value': value_estimate,
                    'reward': None  # To be filled in later
                })
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
    from AlphaZero.model.state_encoder import StateEncoder
    from AlphaZero.model.action_mapper import ActionMapper
    from AlphaZero.core.mcts import MCTS
    
    # Use defaults if no config provided
    if config is None:
        from AlphaZero.utils.config import get_config
        config = get_config()
        
    
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
    agent = AlphaZeroAgent(player_id, network, state_encoder, action_mapper, mcts)
    if agent.training_mode:
        agent.temperature = config.get('temperature', 1.0)  # Set temperature for exploration
    else:
        agent.temperature = 0.5  # Set a lower temperature for evaluation
    return agent 
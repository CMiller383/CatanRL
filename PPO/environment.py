import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional

from game.board import Board
from game.player import Player
from game.resource import Resource
from game.spot import SettlementType
from game.game_logic import GameLogic
from agent.base import AgentType


class CatanEnvironment:
    """
    A reinforcement learning environment wrapper for the Catan game.
    Follows a gym-like interface.
    """
    def __init__(self, num_players=4, agent_types=None):
        self.board = Board()
        self.game = GameLogic(self.board, num_human_players=0, agent_types=agent_types)
        self.current_player_idx = 0  # Track our PPO agent's player index
        self.episode_rewards = 0

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state"""
        self.board = Board()
        self.game = GameLogic(self.board, num_human_players=0, 
                              agent_types=[AgentType.PPO if i == self.current_player_idx else AgentType.RANDOM 
                                          for i in range(4)])
        self.episode_rewards = 0
        
        # Handle setup phase automatically
        self._handle_setup_phase()
        
        return self._get_observation()
    
    def _handle_setup_phase(self):
        """Automatically handle the setup phase for all players"""
        while not self.game.is_setup_complete():
            # If it's our agent's turn during setup, we'll skip for now
            # In a real implementation, the agent should make this decision
            if self.game.current_player_idx == self.current_player_idx:
                # For now, just choose random valid actions for the setup phase
                self._random_setup_action()
            else:
                # For other agents, use the game's built-in AI
                self.game.process_ai_turn()

    def _random_setup_action(self):
        """Take random setup actions for the PPO agent"""
        game = self.game
        if not game.setup_phase_settlement_placed:
            # Choose a random valid settlement location
            valid_spots = [spot_id for spot_id in game.board.spots 
                          if game.is_valid_initial_settlement(spot_id)]
            if valid_spots:
                spot_id = random.choice(valid_spots)
                game.place_initial_settlement(spot_id)
                game.last_settlement_placed = spot_id
        else:
            # Choose a random valid road location
            valid_roads = [road_id for road_id in game.board.roads 
                         if game.is_valid_initial_road(road_id, game.last_settlement_placed)]
            if valid_roads:
                road_id = random.choice(valid_roads)
                game.place_initial_road(road_id, game.last_settlement_placed)

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment
        
        Args:
            action: The action to take
            
        Returns:
            observation: The new state
            reward: The reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Wait until it's our agent's turn
        self._advance_to_agent_turn()
        
        # Take the selected action
        valid_move = self._take_action(action)
        
        # Process other agents' turns until it's our turn again or game is over
        while self.game.current_player_idx != self.current_player_idx and not self._is_game_over():
            self.game.process_ai_turn()
        
        # Get new state, reward, and done flag
        new_state = self._get_observation()
        reward = self._calculate_reward(valid_move)
        done = self._is_game_over()
        info = {'victory_points': self._get_victory_points()}
        
        self.episode_rewards += reward
        
        if done:
            info['episode'] = {'r': self.episode_rewards}
        
        return new_state, reward, done, info
    
    def _advance_to_agent_turn(self):
        """Advance the game until it's our agent's turn"""
        while self.game.current_player_idx != self.current_player_idx and not self._is_game_over():
            self.game.process_ai_turn()
    
    def _take_action(self, action_idx) -> bool:
        """Translate the action index to a game action and execute it"""
        possible_moves = list(self.game.possible_moves)
        if not possible_moves:
            return False
            
        if action_idx >= len(possible_moves):
            return False  # Invalid action
            
        selected_move = possible_moves[action_idx]
        return self.game.do_move(selected_move)
    
    def _get_observation(self) -> np.ndarray:
        """Convert the game state to a numerical observation vector"""
        # This is a simplified version - a real implementation would need more detail
        obs = []
        player = self.game.players[self.current_player_idx]
        
        # Player resources
        for resource in Resource:
            if resource != Resource.DESERT:
                obs.append(player.resources[resource])
        
        # Player buildings
        obs.append(len(player.settlements))
        obs.append(len(player.cities))
        obs.append(len(player.roads))
        
        # Development cards (simplified)
        obs.append(len(player.dev_cards))
        
        # Victory points
        obs.append(player.get_victory_points())
        
        # Game board state (simplified)
        # For each spot, is it owned by our player, another player, or empty
        for spot_id, spot in self.game.board.spots.items():
            if spot.player is None:
                obs.extend([0, 0])  # [is_mine, is_opponent]
            elif spot.player == player.player_id:
                obs.extend([1, 0])  # It's mine
            else:
                obs.extend([0, 1])  # It's an opponent's
                
            # Settlement type (0=none, 1=settlement, 2=city)
            obs.append(spot.settlement_type.value)
        
        # For each road, is it owned by our player, another player, or empty
        for road_id, road in self.game.board.roads.items():
            if road.owner is None:
                obs.extend([0, 0])  # [is_mine, is_opponent]
            elif road.owner == player.player_id:
                obs.extend([1, 0])  # It's mine
            else:
                obs.extend([0, 1])  # It's an opponent's
                
        # Dice probabilities for resources
        # This represents the probability of getting each resource based on board state
        resource_probs = self._calculate_resource_probabilities()
        obs.extend(resource_probs)
        
        # Game phase
        obs.append(self.game.current_phase.value)
        
        # Current dice roll
        if self.game.last_dice1_roll and self.game.last_dice2_roll:
            obs.append(self.game.last_dice1_roll + self.game.last_dice2_roll)
        else:
            obs.append(0)
            
        return np.array(obs, dtype=np.float32)
    
    def _calculate_resource_probabilities(self) -> List[float]:
        """Calculate probability of gaining each resource on a dice roll"""
        # Map of dice roll (2-12) to probability
        dice_probs = {
            2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36, 
            7: 6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
        }
        
        player = self.game.players[self.current_player_idx]
        resource_probs = {r: 0.0 for r in Resource if r != Resource.DESERT}
        
        # For each settlement/city
        for spot_id in player.settlements + player.cities:
            spot = self.game.board.get_spot(spot_id)
            # Check adjacent hexes
            for hex_id in spot.adjacent_hex_ids:
                hex_obj = self.game.board.get_hex(hex_id)
                if hex_obj.resource != Resource.DESERT and hex_obj.hex_id != self.game.robber_hex_id:
                    # Multiply by 2 for cities
                    multiplier = 2 if spot_id in player.cities else 1
                    resource_probs[hex_obj.resource] += dice_probs[hex_obj.number] * multiplier
        
        return list(resource_probs.values())
    
    def _calculate_reward(self, valid_move: bool) -> float:
        """Calculate the reward for the current state"""
        if not valid_move:
            return -0.1  # Penalty for invalid move
            
        player = self.game.players[self.current_player_idx]
        
        # Main reward based on victory points
        vp_reward = player.get_victory_points() * 0.5
        
        # Additional rewards for good game tactics
        resource_reward = sum(player.resources.values()) * 0.01
        
        # Check if we've built something this turn
        built_reward = 0.0
        # Add more sophisticated reward shaping here
        
        return vp_reward + resource_reward + built_reward
    
    def _is_game_over(self) -> bool:
        """Check if the game is over (someone has reached 10 victory points)"""
        for player in self.game.players:
            total_vp = self._calculate_total_vp(player)
            if total_vp >= 10:
                return True
        return False
    
    def _calculate_total_vp(self, player: Player) -> int:
        """Calculate total victory points including longest road and largest army"""
        vp = player.get_victory_points()
        
        # Add longest road (2 points)
        if self.game.longest_road_player == player.player_id:
            vp += 2
            
        # Add largest army (2 points)
        if self.game.largest_army_player == player.player_id:
            vp += 2
            
        return vp
    
    def _get_victory_points(self) -> int:
        """Get the current victory points for our agent"""
        player = self.game.players[self.current_player_idx]
        return self._calculate_total_vp(player)
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space"""
        # This is a maximum estimate - we'll mask invalid actions during training
        return 100  # Placeholder - should be sized based on all possible moves
    
    def get_observation_space_size(self) -> int:
        """Get the size of the observation space"""
        # Run a test to get the observation vector length
        test_obs = self._get_observation()
        return len(test_obs)
    
    def render(self):
        """Render the current game state (can use built-in visualization)"""
        from visualization.board_graph import visualize_board
        visualize_board(self.board)
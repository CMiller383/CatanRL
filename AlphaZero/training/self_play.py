from tqdm import tqdm

class SelfPlayWorker:
    """
    Worker that generates self-play games for training
    """
    def __init__(self, game_creator, agent_creator, config):
        """
        Initialize the self-play worker
        
        Args:
            game_creator: Function that creates a new game instance
            agent_creator: Function that creates an agent
            config: Configuration dictionary
        """
        self.game_creator = game_creator
        self.agent_creator = agent_creator
        self.config = config
    
    def generate_games(self, num_games):
        """
        Generate self-play games
        
        Args:
            num_games: Number of games to generate
            
        Returns:
            game_data: List of game data
        """
        all_game_data = []
        
        for game_idx in tqdm(range(num_games), desc="Self-play games"):
            # Create a new game
            game = self.game_creator()
            
            # Create AlphaZero agents for all players
            for player_idx in range(len(game.agents)):
                agent = self.agent_creator(player_id=player_idx)
                agent.set_training_mode(True)  # Enable training mode
                game.agents[player_idx] = agent
            
            # Handle the setup phase
            while not game.is_setup_complete():
                game.process_ai_turn()
                
            # Play the game
            move_count = 0
            max_moves = self.config.get('max_moves', 200)
            
            # Main game loop
            while not self._is_game_over(game.state) and move_count < max_moves:
                move_count += 1
                game.process_ai_turn()
            
            # Calculate rewards for all agents
            winner = self._get_winner(game.state)
            
            # Collect game data from all agents
            for player_idx, agent in enumerate(game.agents):
                # Calculate reward for this player
                reward = self._calculate_reward(game.state, player_id=player_idx)
                
                # Record final reward in agent's game history
                agent.record_game_result(reward)
                
                # Get and add game history data
                game_data = agent.get_game_history()
                all_game_data.extend(game_data)
            
            # Log game results
            winner = self._get_winner(game.state)
            print(f"Game {game_idx+1}: Player {winner} won with "
                f"{game.state.players[winner].victory_points} VP")
        
        return all_game_data
    
    def _is_game_over(self, game_state):
        """Check if a game is over"""
        # Check if any player has 10+ victory points
        for player in game_state.players:
            if player.victory_points >= 10:
                return True
        return False
    
    def _get_winner(self, game_state):
        """Get the winner's player index"""
        max_vp = -1
        winner = None
        
        for i, player in enumerate(game_state.players):
            vp = player.victory_points
            if vp > max_vp:
                max_vp = vp
                winner = i
        
        return winner
    
    def _calculate_reward(self, game_state, player_id):
        """
        Calculate a more sophisticated reward for the AlphaZero agent
        
        Args:
            game_state: The game state
            player_id: The player ID for which to calculate the reward
            
        Returns:
            reward: The calculated reward
        """
        # Get the player and their victory points
        player = game_state.players[player_id]
        player_vp = player.victory_points
        
        # Get the winner and maximum VP
        winner = self._get_winner(game_state)
        max_vp = game_state.players[winner].victory_points
        
        # Base reward component: win/loss
        if winner == player_id:
            base_reward = 1.0  # Win
        else:
            base_reward = -1.0  # Loss
        
        # VP-based component: reward based on VP difference from winner
        # This helps distinguish between close losses and big losses
        vp_diff = player_vp - max_vp
        vp_component = vp_diff / 10.0  # Normalize by max VP (10)
        
        # Progress component: reward progress toward victory (settlements, cities, etc.)
        progress_score = 0.0
        
        # Add 0.1 for each settlement
        progress_score += len(player.settlements) * 0.1
        
        # Add 0.2 for each city
        progress_score += len(player.cities) * 0.2
        
        # Add 0.05 for each road
        progress_score += len(player.roads) * 0.05
        
        # Add 0.1 for each development card
        progress_score += len(player.dev_cards) * 0.1
        
        # Normalize progress score
        progress_component = progress_score / 5.0  # Max possible around 5
        
        # Resource diversity component: reward having diverse resources
        resource_count = sum(1 for r, count in player.resources.items() if count > 0)
        diversity_component = resource_count / 5.0  # 5 resource types
        
        # Combine components with weights
        reward = (
            0.6 * base_reward +      # Win/loss is the most important
            0.2 * vp_component +     # VP difference matters
            0.15 * progress_component + # Progress toward victory
            0.05 * diversity_component  # Resource diversity
        )
        
        return reward
    


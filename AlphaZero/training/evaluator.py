
from tqdm import tqdm

class Evaluator:
    """
    Evaluates the AlphaZero agent against baseline agents
    """
    def __init__(self, game_creator, agent_creator, config):
        """
        Initialize the evaluator
        
        Args:
            game_creator: Function that creates a new game instance
            agent_creator: Function that creates an agent
            config: Configuration dictionary
        """
        self.game_creator = game_creator
        self.agent_creator = agent_creator
        self.config = config
    
    def evaluate(self, num_games=None):
        """
        Evaluate the agent
        
        Args:
            num_games: Number of evaluation games
            
        Returns:
            results: Evaluation results
        """
        if num_games is None:
            num_games = self.config.get('eval_games', 20)
        
        # Results tracking
        wins = 0
        vp_total = 0
        game_lengths = []
        
        # Create evaluation agent
        agent = self.agent_creator(player_id=0)
        agent.set_training_mode(False)  # Turn off training mode for evaluation
        
        for game_idx in tqdm(range(num_games), desc="Evaluation games"):
            # Create a new game
            game = self.game_creator()
            game.agents[0] = agent
            
            # Handle the setup phase
            while not game.is_setup_complete():
                game.process_ai_turn()
            
            # Track game length
            moves = 0
            max_moves = self.config.get('max_moves', 200)
            
            # Main game loop
            while not self._is_game_over(game.state) and moves < max_moves:
                moves += 1
                
                if game.state.current_player_idx == 0:
                    # AlphaZero agent's turn
                    action = agent.get_action(game.state)
                    game.do_action(action)
                else:
                    # Other agents' turns
                    game.process_ai_turn()
            
            # Record results
            game_lengths.append(moves)
            vp_total += game.state.players[0].victory_points
            
            # Check if our agent won
            winner = self._get_winner(game.state)
            if winner == 0:
                wins += 1
            
            print(f"Game {game_idx+1}: Player {winner} won with "
                  f"{game.state.players[winner].victory_points} VP "
                  f"(Our agent: {game.state.players[0].victory_points} VP)")
        
        # Calculate statistics
        win_rate = wins / num_games
        avg_vp = vp_total / num_games
        avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        
        results = {
            'win_rate': win_rate,
            'avg_vp': avg_vp,
            'avg_game_length': avg_length,
            'num_games': num_games
        }
        
        print(f"Evaluation results:")
        print(f"  Win rate: {win_rate:.2f}")
        print(f"  Average VP: {avg_vp:.2f}")
        print(f"  Average game length: {avg_length:.2f} moves")
        
        return results
    
    def _is_game_over(self, game_state):
        """Check if the game is over"""
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

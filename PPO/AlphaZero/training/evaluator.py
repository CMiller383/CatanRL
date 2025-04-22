import time
import psutil
import torch
from tqdm import tqdm

class Evaluator:
    """
    Evaluates the AlphaZero agent against baseline agents
    """
    def __init__(self, game_creator, agent_creator, config, log_fn=None):
        """
        Initialize the evaluator

        Args:
            game_creator: Function that creates a new game instance
            agent_creator: Function that creates an agent
            config: Configuration dictionary
            log_fn: Logging function (defaults to print)
        """
        self.game_creator = game_creator
        self.agent_creator = agent_creator
        self.config = config
        self.log = log_fn or print
    
    def evaluate(self, num_games=None):
        """
        Evaluate the agent

        Args:
            num_games: Number of evaluation games

        Returns:
            results: Evaluation results including win rate, average victory points,
                    average game length, and total moves
        """
        if num_games is None:
            num_games = self.config.get('eval_games', 20)
        
        wins = 0
        vp_total = 0
        game_lengths = []
        per_game_times = []
        per_game_vps = []

        # Create evaluation agent
        agent = self.agent_creator(player_id=0)
        agent.set_training_mode(False)  # Turn off training mode for evaluation

        eval_start = time.time()
        for game_idx in tqdm(range(num_games), desc="Evaluation games"):
            game_start = time.time()
            # Create a new game
            game = self.game_creator()
            game.agents[0] = agent
            
            # Handle the setup phase
            while not game.is_setup_complete():
                game.process_ai_turn()
            
            # Main game loop
            moves = 0
            max_moves = self.config.get('max_moves', 200)
            while not self._is_game_over(game.state) and moves < max_moves:
                moves += 1
                if game.state.current_player_idx == 0:
                    action = agent.get_action(game.state)
                    game.do_action(action)
                else:
                    game.process_ai_turn()
            
            # Collect metrics
            duration = time.time() - game_start
            per_game_times.append(duration)
            vp = game.state.players[0].victory_points
            per_game_vps.append(vp)
            game_lengths.append(moves)
            vp_total += vp

            # Determine winner
            winner = self._get_winner(game.state)
            winner_vp = game.state.players[winner].victory_points
            self.log(f"Game {game_idx+1}: duration={duration:.2f}s, moves={moves}, our_VP={vp}, winner={winner} VP={winner_vp}")
            if winner == 0:
                wins += 1
            
        total_eval_time = time.time() - eval_start
        win_rate = wins / num_games
        avg_vp = vp_total / num_games
        avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        total_moves = sum(game_lengths)
        rate = num_games / total_eval_time if total_eval_time > 0 else 0

        # Summary logging
        self.log(f"Evaluated {num_games} games in {total_eval_time:.2f}s ({rate:.2f} games/s)")
        self.log(f"Evaluation results: win_rate={win_rate:.2f}, avg_vp={avg_vp:.2f}, avg_length={avg_length:.2f}, total_moves={total_moves}")

        # Resource usage logging
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            self.log(f"Eval resource usage: CPU {cpu:.1f}%, RAM {ram:.1f}%, GPU peak memory {gpu_mem:.2f} GB")
        else:
            self.log(f"Eval resource usage: CPU {cpu:.1f}%, RAM {ram:.1f}%")

        results = {
            'win_rate': win_rate,
            'avg_vp': avg_vp,
            'avg_game_length': avg_length,
            'total_moves': total_moves,
            'num_games': num_games,
            'per_game_times': per_game_times,
            'per_game_vps': per_game_vps
        }

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
            if player.victory_points > max_vp:
                max_vp = player.victory_points
                winner = i
        return winner

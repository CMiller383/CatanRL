#!/usr/bin/env python
"""
Standalone evaluation script for AlphaZero Catan.
Evaluates a trained model against random bots.

Usage:
    python evaluate_model.py [--model MODEL_PATH] [--games NUM_GAMES] [--debug]

Example:
    python evaluate_model.py --model models/best_model.pt --games 50
"""

import argparse
import os
import time
import random
import numpy as np
import torch
from tqdm import tqdm
import json
from datetime import datetime

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Import game components
from game.board import Board
from game.game_logic import GameLogic
from game.enums import Resource, SettlementType
from agent.base import AgentType
from game.game_state import check_game_over
from AlphaZero.utils.alphazero_utils import load_alphazero_agent

class ModelEvaluator:
    """Evaluates a trained AlphaZero model against random agents."""
    def __init__(self, model_path="models/best_model.pt", num_games=20, debug=False, 
                 num_simulations=100, c_puct=1.5, batch_size=8):
        self.model_path = model_path
        self.num_games = num_games
        self.debug = debug
        self.forced_num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        
        # Results tracking
        self.wins = 0
        self.total_vp = 0
        self.game_lengths = []
        self.resource_stats = {r: {'collected': 0, 'used': 0} for r in Resource if r != Resource.DESERT}
        self.build_stats = {
            'settlements': 0,
            'cities': 0,
            'roads': 0,
            'dev_cards': 0
        }
        self.per_game_stats = []
        
        # Detailed tracking per game
        self.action_types_per_game = []
        
        # Load the model and configuration
        self.load_model_config()
    
    def load_model_config(self):
        """Load configuration from the model file."""
        try:
            # Set device appropriately
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(self.model_path, map_location=device)
            self.config = checkpoint.get('config', {})
            self.iteration = checkpoint.get('iteration', 'unknown')
            
            # Override with explicit parameters
            self.config['num_simulations'] = self.forced_num_simulations
            self.config['c_puct'] = self.c_puct
            self.config['mcts_batch_size'] = self.batch_size
            
            if self.debug:
                print(f"Loaded configuration from {self.model_path}")
                print(f"Model iteration: {self.iteration}")
                print(f"MCTS simulations: {self.config['num_simulations']}")
                print(f"c_puct: {self.config['c_puct']}")
                print(f"MCTS batch size: {self.config['mcts_batch_size']}")
        except Exception as e:
            print(f"Couldn't load config from model: {e}")
            from AlphaZero.utils.config import get_config
            self.config = get_config()
            # Apply override parameters
            self.config['num_simulations'] = self.forced_num_simulations
            self.config['c_puct'] = self.c_puct
            self.config['mcts_batch_size'] = self.batch_size
            print("Using default configuration with overrides")
    
    def create_game(self):
        """Create a new game with AlphaZero at position 0 and random agents elsewhere."""
        board = Board()
        agent_types = [AgentType.ALPHAZERO] + [AgentType.RANDOM] * 3
        game = GameLogic(board, agent_types=agent_types)
        
        # Custom agent creation to ensure proper configuration
        from AlphaZero.core.network import DeepCatanNetwork
        from AlphaZero.model.state_encoder import StateEncoder
        from AlphaZero.model.action_mapper import ActionMapper
        from AlphaZero.core.mcts import MCTS
        from AlphaZero.agent.alpha_agent import AlphaZeroAgent
        
        # Load the checkpoint directly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_path, map_location=device)
        
        # Get configuration values
        state_dim = self.config.get('state_dim', 992)
        action_dim = self.config.get('action_dim', 200)
        hidden_dim = self.config.get('hidden_dim', 256)
        num_simulations = self.config.get('num_simulations', 100)
        c_puct = self.config.get('c_puct', 1.5)
        batch_size = self.config.get('mcts_batch_size', 8)
        
        if self.debug:
            print(f"Creating AlphaZero agent with {num_simulations} MCTS simulations")
            print(f"c_puct: {c_puct}")
            print(f"Batch size: {batch_size}")
        
        # Always use DeepCatanNetwork as that's what was used in training
        network = DeepCatanNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
            
        # Load the trained weights
        network.load_state_dict(checkpoint['network_state_dict'])
        network.eval()  # Set to evaluation mode
        network.to(device)
        
        # Create state encoder and action mapper
        state_encoder = StateEncoder(max_actions=action_dim)
        action_mapper = ActionMapper(max_actions=action_dim)
        
        # Create MCTS with specific parameters
        mcts = MCTS(
            network=network, 
            state_encoder=state_encoder, 
            action_mapper=action_mapper, 
            num_simulations=num_simulations,
            c_puct=c_puct,
            batch_size=batch_size
        )
        
        # Create and configure agent
        alpha_agent = AlphaZeroAgent(
            player_id=0, 
            network=network, 
            state_encoder=state_encoder, 
            action_mapper=action_mapper, 
            mcts=mcts
        )
        alpha_agent.set_training_mode(False)  # Make sure training mode is off
        alpha_agent.temperature = 1.5
        alpha_agent.game_state = game.state
        
        if self.debug:
            # Set debug flag on agent
            alpha_agent.debug = True
        
        # Replace the default agent with our custom one
        game.agents[0] = alpha_agent
        
        return game
    
    def _get_winner(self, game_state):
        """Get the winner's player index."""
        if game_state.winner is not None:
            return game_state.winner
            
        max_vp = -1
        winner = None
        
        for i, player in enumerate(game_state.players):
            if player.victory_points > max_vp:
                max_vp = player.victory_points
                winner = i
        
        return winner
    
    def _is_game_over(self, game_state):
        """Check if the game is over."""
        if game_state.winner is not None:
            return True
        return check_game_over(game_state)

    def _track_actions(self, player, action_counts):
        """Track the action types used by the player."""
        if action_counts is None:
            return {}
        return action_counts
    
    def evaluate(self):
        """Run the evaluation process."""
        print(f"=== Evaluating AlphaZero Catan Model ===")
        print(f"Model: {self.model_path}")
        print(f"Games: {self.num_games}")
        print(f"MCTS Simulations: {self.config.get('num_simulations', 100)}")
        print(f"C_puct: {self.config.get('c_puct', 1.5)}")
        
        start_time = time.time()
        
        for game_idx in tqdm(range(self.num_games), desc="Evaluation games"):
            game_start = time.time()
            
            # Create a new game
            game = self.create_game()
            
            # Handle the setup phase
            while not game.is_setup_complete():
                game.process_ai_turn()
            
            # Track actions per turn for AlphaZero player
            action_counts = {}
            
            # Main game loop
            moves = 0
            max_moves = self.config.get('max_moves', 200)
            
            while not self._is_game_over(game.state) and moves < max_moves:
                moves += 1
                
                if game.state.current_player_idx == 0:
                    # It's AlphaZero's turn
                    action = game.get_current_agent().get_action(game.state)
                    
                    # Track action type
                    action_type = action.type.name if hasattr(action, 'type') else str(action)
                    action_counts[action_type] = action_counts.get(action_type, 0) + 1
                    
                    # Execute action
                    game.do_action(action)
                else:
                    # Other player's turn
                    game.process_ai_turn()
            
            # Collect game results
            game_duration = time.time() - game_start
            
            # Get AlphaZero player stats
            player = game.state.players[0]
            vp = player.victory_points
            
            # Get settlements, cities, roads, dev cards
            settlements = len(player.settlements)
            cities = len(player.cities)
            roads = len(player.roads)
            dev_cards = len(player.dev_cards) + len(player.played_dev_cards)
            
            # Resources collected and used
            resources_collected = {}
            for r in Resource:
                if r != Resource.DESERT:
                    # This is an approximation
                    current = player.resources[r]
                    used_for_building = 0  # Would need more detailed tracking
                    
                    # Update global stats
                    self.resource_stats[r]['collected'] += current
            
            # Determine winner
            winner = self._get_winner(game.state)
            winner_vp = game.state.players[winner].victory_points
            
            if winner == 0:
                self.wins += 1
            
            # Update statistics
            self.total_vp += vp
            self.game_lengths.append(moves)
            self.build_stats['settlements'] += settlements
            self.build_stats['cities'] += cities
            self.build_stats['roads'] += roads
            self.build_stats['dev_cards'] += dev_cards
            
            # Store per-game stats
            game_stats = {
                'game_id': game_idx,
                'result': 'win' if winner == 0 else 'loss',
                'victory_points': vp,
                'winner': winner,
                'winner_vp': winner_vp,
                'moves': moves,
                'duration_seconds': game_duration,
                'buildings': {
                    'settlements': settlements,
                    'cities': cities,
                    'roads': roads,
                    'dev_cards': dev_cards
                },
                'action_distribution': action_counts
            }
            self.per_game_stats.append(game_stats)
            
            # Action types tracking
            self.action_types_per_game.append(self._track_actions(player, action_counts))
            
            if self.debug:
                print(f"\nGame {game_idx+1} result: {'Win' if winner == 0 else 'Loss'}")
                print(f"  AlphaZero VP: {vp}, Winner: Player {winner} with {winner_vp} VP")
                print(f"  Moves: {moves}, Duration: {game_duration:.2f}s")
                print(f"  Buildings: {settlements} settlements, {cities} cities, {roads} roads, {dev_cards} dev cards")
        
        # Calculate final statistics
        total_duration = time.time() - start_time
        win_rate = self.wins / self.num_games
        avg_vp = self.total_vp / self.num_games
        avg_game_length = sum(self.game_lengths) / self.num_games
        
        # Aggregate action statistics
        action_distribution = {}
        for game_actions in self.action_types_per_game:
            for action_type, count in game_actions.items():
                action_distribution[action_type] = action_distribution.get(action_type, 0) + count
        
        # Sort by frequency
        action_distribution = {k: v for k, v in sorted(
            action_distribution.items(), key=lambda item: item[1], reverse=True
        )}
        
        # Print summary
        print("\n=== Evaluation Results ===")
        print(f"Games played: {self.num_games}")
        print(f"Total time: {total_duration:.2f}s ({self.num_games/total_duration:.2f} games/s)")
        print(f"Win rate: {win_rate:.2f} ({self.wins}/{self.num_games})")
        print(f"Average victory points: {avg_vp:.2f}")
        print(f"Average game length: {avg_game_length:.2f} moves")
        
        print("\nBuilding statistics (average per game):")
        for building, count in self.build_stats.items():
            print(f"  {building}: {count/self.num_games:.2f}")
        
        print("\nAction type distribution:")
        total_actions = sum(action_distribution.values())
        for action_type, count in action_distribution.items():
            print(f"  {action_type}: {count} ({count/total_actions*100:.1f}%)")
        
        # Save results to file
        self.save_results()
        
        return {
            'win_rate': win_rate,
            'avg_vp': avg_vp,
            'avg_game_length': avg_game_length,
            'num_games': self.num_games,
            'action_distribution': action_distribution,
            'per_game_stats': self.per_game_stats
        }
    
    def save_results(self):
        """Save evaluation results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        model_name = os.path.basename(self.model_path).replace('.pt', '')
        filename = f"{results_dir}/eval_{model_name}_{timestamp}.json"
        
        # Prepare results
        results = {
            'model': self.model_path,
            'timestamp': timestamp,
            'num_games': self.num_games,
            'win_rate': self.wins / self.num_games,
            'avg_vp': self.total_vp / self.num_games,
            'avg_game_length': sum(self.game_lengths) / self.num_games,
            'build_stats': {k: v/self.num_games for k, v in self.build_stats.items()},
            'per_game_stats': self.per_game_stats
        }
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero Catan model against random agents")
    parser.add_argument("--model", type=str, default="models/best_model.pt",
                        help="Path to the model file (default: models/best_model.pt)")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of evaluation games to play (default: 20)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--simulations", type=int, default=100,
                        help="Override MCTS simulation count (default: 100)")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="Exploration constant for MCTS (default: 1.5)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for MCTS (default: 8)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with training evaluator implementation")
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(
        model_path=args.model,
        num_games=args.games,
        debug=args.debug,
        num_simulations=args.simulations,
        c_puct=args.c_puct,
        batch_size=args.batch_size
    )
    
    results = evaluator.evaluate()
    
    # If requested, compare with the training evaluator implementation
    if args.compare:
        print("\n=== Comparing with Training Evaluator ===")
        try:
            from AlphaZero.training.evaluator import Evaluator
            from AlphaZero.utils.alphazero_utils import load_alphazero_agent
            
            # Create a game factory function
            def create_game():
                board = Board()
                return GameLogic(board, agent_types=[AgentType.ALPHAZERO] + [AgentType.RANDOM] * 3)
            
            # Create an agent factory function
            def create_agent(player_id):
                return load_alphazero_agent(player_id=player_id, model_path=args.model)
            
            # Load configuration
            from AlphaZero.utils.config import get_config
            config = get_config()
            if args.simulations:
                config['num_simulations'] = args.simulations
            
            # Create and run the training evaluator
            print(f"Running {args.games} games with training evaluator...")
            training_evaluator = Evaluator(create_game, create_agent, config)
            training_results = training_evaluator.evaluate(args.games)
            
            print("\nResults comparison:")
            print(f"Custom evaluator:   Win rate = {results['win_rate']:.2f}, Avg VP = {results['avg_vp']:.2f}")
            print(f"Training evaluator: Win rate = {training_results['win_rate']:.2f}, Avg VP = {training_results['avg_vp']:.2f}")
            
        except Exception as e:
            print(f"Failed to run training evaluator comparison: {e}")
    

if __name__ == "__main__":
    main()
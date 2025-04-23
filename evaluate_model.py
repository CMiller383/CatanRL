"""
Enhanced evaluation script for AlphaZero Catan.
Evaluates a trained model against random agents or other trained models.

Usage:
    python evaluate_model.py [--model MODEL_PATH] [--games NUM_GAMES] [--debug]
    python evaluate_model.py --model models/best_model.pt --opponent models/model_iter_10.pt

Advanced options:
    --config {default,eval,strong}  # Predefined configurations
    --agents "A,R,H,R"              # Agent types: A=AlphaZero, R=Random, H=Heuristic
    --opponents "model1.pt,model2.pt,model3.pt" # Model paths for opponent AlphaZero agents
    --deterministic                 # Use deterministic MCTS (no temperature)
    --temperature TEMP              # Temperature for MCTS action selection
    --placement-network PATH        # Path to placement network model to use
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
import copy

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
from game.enums import Resource, SettlementType, ActionType, GamePhase
from agent.base import AgentType, create_agent
from game.game_state import check_game_over
from AlphaZero.utils.alphazero_utils import load_alphazero_agent
from AlphaZero.utils.config import get_eval_config

class ModelEvaluator:
    """Evaluates trained AlphaZero models against various opponents."""
    def __init__(self, 
                 model_path="models/best_model.pt",
                 agent_spec="A,R,R,R",
                 opponent_models=None,
                 placement_network_path=None,
                 num_games=20, 
                 debug=False,
                 deterministic=True,
                 temperature=0.5):
        
        self.model_path = model_path
        self.num_games = num_games
        self.debug = debug
        self.deterministic = deterministic
        self.temperature = temperature
        self.placement_network_path = placement_network_path
        
        # Parse agent specification
        self.agent_spec = agent_spec
        self.agent_types = self._parse_agent_spec(agent_spec)
        
        # Setup opponent models
        self.opponent_models = opponent_models or []
        
        # Load config
        self.config = self._load_config()
        
        # Results tracking
        self.wins = {i: 0 for i in range(4)}
        self.total_vps = {i: 0 for i in range(4)}
        self.game_lengths = []
        self.build_stats = {i: {'settlements': 0, 'cities': 0, 'roads': 0, 'dev_cards': 0} for i in range(4)}
        self.per_game_stats = []
        
        # Action types tracking per agent
        self.action_types_per_game = {i: [] for i in range(4)}
        
        # Print configuration
        if debug:
            self._print_configuration()
    
    def _parse_agent_spec(self, agent_spec):
        """Parse agent specification string to agent types."""
        agent_mapping = {
            'A': AgentType.ALPHAZERO,
            'R': AgentType.RANDOM,
            'H': AgentType.HEURISTIC,
        }
        
        agents = agent_spec.split(',')
        if len(agents) != 4:
            print(f"Warning: Agent spec '{agent_spec}' doesn't have 4 agents. Using default A,R,R,R.")
            return [AgentType.ALPHAZERO, AgentType.RANDOM, AgentType.RANDOM, AgentType.RANDOM]
        
        return [agent_mapping.get(a.strip().upper(), AgentType.RANDOM) for a in agents]
    
    def _load_config(self):
        """Load configuration based on name or from model file."""
        # Start with preset config if available
        config = get_eval_config()
        
        try:
            # Try to load network parameters from model file
            device = torch.device(config.get('device', 'cpu'))
            checkpoint = torch.load(self.model_path, map_location=device)
            model_config = checkpoint.get('config', {})
            self.iteration = checkpoint.get('iteration', 'unknown')
            
            # Update network parameters from model
            if model_config:
                for key in ['state_dim', 'action_dim', 'hidden_dim']:
                    if key in model_config:
                        config[key] = model_config[key]
            
            if self.debug:
                print(f"Loaded model parameters from {self.model_path} (iteration {self.iteration})")
        except Exception as e:
            if self.debug:
                print(f"Could not load parameters from model: {e}")
                print("Using default parameters")
        
        return config
    
    def _print_configuration(self):
        """Print current configuration."""
        print(f"\n=== Evaluation Configuration ===")
        print(f"Model: {self.model_path}")
        print(f"Agent setup: {self.agent_spec}")
        print(f"Opponent models: {', '.join(self.opponent_models) if self.opponent_models else 'None'}")
        if self.placement_network_path:
            print(f"Placement network: {self.placement_network_path}")
        print(f"Games: {self.num_games}")
        print(f"Deterministic: {self.deterministic}, Temperature: {self.temperature}")
        print(f"MCTS config:")
        print(f"  - Simulations: {self.config.get('num_simulations', 100)}")
        print(f"  - c_puct: {self.config.get('c_puct', 1.5)}")
        print(f"  - Batch size: {self.config.get('mcts_batch_size', 8)}")
        print(f"  - Noise: {self.config.get('noise_eps', 0.0)}")
        print(f"Device: {self.config.get('device', 'cpu')}")
    
    def _load_placement_network(self):
        """Load the placement network if specified."""
        if not self.placement_network_path or not os.path.exists(self.placement_network_path):
            return None
        
        try:
            # Import placement network class
            from AlphaZero.core.initial_placement_network import InitialPlacementNetwork
            
            # Load the checkpoint
            device = torch.device(self.config.get('device', 'cpu'))
            checkpoint = torch.load(self.placement_network_path, map_location=device)
            
            if 'network_state_dict' in checkpoint:
                # Get dimensions from state dict
                state_dict = checkpoint['network_state_dict']
                
                # Extract dimensions
                input_dim = 260  # Default
                hidden_dim = 128  # Default
                output_dim = 54  # Fixed (number of spots)
                
                if 'fc1.weight' in state_dict:
                    input_dim = state_dict['fc1.weight'].shape[1]
                    hidden_dim = state_dict['fc1.weight'].shape[0]
                
                # Create network with correct dimensions
                network = InitialPlacementNetwork(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim
                )
                
                # Load weights
                network.load_state_dict(state_dict)
                network.eval()
                
                if self.debug:
                    print(f"Loaded placement network from {self.placement_network_path}")
                    print(f"Network dimensions: input={input_dim}, hidden={hidden_dim}, output={output_dim}")
                
                return network
            else:
                if self.debug:
                    print(f"Invalid placement network format: missing network_state_dict")
                return None
                
        except Exception as e:
            if self.debug:
                print(f"Error loading placement network: {e}")
            return None
    
    def create_game(self):
        """Create a new game with specified agents."""
        board = Board()
        game = GameLogic(board, agent_types=self.agent_types)
        
        # Load placement network if specified
        placement_network = self._load_placement_network()
        
        # Configure AlphaZero agents
        for i, agent_type in enumerate(self.agent_types):
            if agent_type == AgentType.ALPHAZERO:
                # Determine which model to use
                if i == 0:  # Main model
                    model_path = self.model_path
                else:  # Opponent model if available
                    idx = i - 1
                    model_path = self.opponent_models[idx] if idx < len(self.opponent_models) else self.model_path
                
                # Create the AlphaZero agent
                alphazero_agent = self._create_alphazero_agent(i, model_path)
                
                # Set placement network if available
                if placement_network is not None:
                    try:
                        alphazero_agent.set_initial_placement_network(placement_network)
                        if self.debug:
                            print(f"Attached placement network to agent {i}")
                    except Exception as e:
                        if self.debug:
                            print(f"Error attaching placement network to agent {i}: {e}")
                
                game.agents[i] = alphazero_agent
        
        return game
    
    def _create_alphazero_agent(self, player_id, model_path):
        """Create a configured AlphaZero agent."""
        from AlphaZero.core.network import DeepCatanNetwork
        from AlphaZero.model.state_encoder import StateEncoder
        from AlphaZero.model.action_mapper import ActionMapper
        from AlphaZero.core.mcts import MCTS
        from AlphaZero.agent.alpha_agent import AlphaZeroAgent
        
        try:
            # Load the checkpoint
            device = torch.device(self.config.get('device', 'cpu'))
            checkpoint = torch.load(model_path, map_location=device)
            
            # Get configuration values
            state_dim = self.config.get('state_dim', 992)
            action_dim = self.config.get('action_dim', 200)
            hidden_dim = self.config.get('hidden_dim', 256)
            num_simulations = self.config.get('num_simulations', 400)
            c_puct = self.config.get('c_puct', 1.0)
            batch_size = self.config.get('mcts_batch_size', 12)
            noise_eps = self.config.get('noise_eps', 0.0)
            noise_alpha = self.config.get('noise_alpha', 0.3)
            
            # Create network and load weights
            network = DeepCatanNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            )
                
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
                batch_size=batch_size,
                noise_eps=noise_eps,
                noise_alpha=noise_alpha
            )
            
            # Create and configure agent
            alpha_agent = AlphaZeroAgent(
                player_id=player_id, 
                network=network, 
                state_encoder=state_encoder, 
                action_mapper=action_mapper, 
                mcts=mcts
            )
            alpha_agent.deterministic = self.deterministic
            alpha_agent.set_training_mode(False)  # Evaluation mode
            alpha_agent.temperature = self.temperature
            
            if self.debug:
                # Set debug flag if needed
                alpha_agent.debug = True
                
                # Log which model is being used for this agent
                model_name = os.path.basename(model_path)
                print(f"Player {player_id}: AlphaZero agent using {model_name}")
            
            return alpha_agent
            
        except Exception as e:
            print(f"Error creating AlphaZero agent for player {player_id}: {e}")
            print(f"Using random agent as fallback")
            return create_agent(player_id, AgentType.RANDOM)
    
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

    def _track_actions(self, player_idx, action, action_counts):
        """Track action types used by a player."""
        if action_counts is None:
            action_counts = {}
            
        action_type = action.type.name if hasattr(action, 'type') else str(action)
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        return action_counts
    
    def evaluate(self):
        """Run the evaluation process."""
        print(f"=== Evaluating AlphaZero Catan Model ===")
        print(f"Primary model: {self.model_path}")
        print(f"Agent setup: {self.agent_spec}")
        if self.opponent_models:
            print(f"Opponent models: {', '.join(self.opponent_models)}")
        if self.placement_network_path:
            print(f"Using placement network: {self.placement_network_path}")
        print(f"Games: {self.num_games}")
        
        start_time = time.time()
        
        for game_idx in tqdm(range(self.num_games), desc="Evaluation games"):
            game_start = time.time()
            
            # Create a new game
            game = self.create_game()
            
            # Handle the setup phase
            while not game.is_setup_complete():
                game.process_ai_turn()
            
            # Track actions per turn
            action_counts = {i: {} for i in range(4)}
            
            # Main game loop
            moves = 0
            max_moves = self.config.get('max_moves', 200)
            
            while not self._is_game_over(game.state) and moves < max_moves:
                moves += 1
                
                current_player = game.state.current_player_idx
                player_agent_type = self.agent_types[current_player]
                
                # If it's an AlphaZero agent's turn
                if player_agent_type == AgentType.ALPHAZERO:
                    # Get action from agent
                    action = game.get_current_agent().get_action(game.state)
                    
                    # Track action type
                    action_counts[current_player] = self._track_actions(
                        current_player, action, action_counts[current_player]
                    )
                    
                    # Execute action
                    game.do_action(action)
                else:
                    # Other player's turn - let the game process it
                    game.process_ai_turn()
            
            # Collect game results
            game_duration = time.time() - game_start
            
            # Determine winner
            winner = self._get_winner(game.state)
            winner_vp = game.state.players[winner].victory_points
            
            # Update statistics for each player
            player_stats = {}
            for i, player in enumerate(game.state.players):
                agent_type = self.agent_types[i]
                type_name = agent_type.name
                
                # Track victory points
                vp = player.victory_points
                self.total_vps[i] += vp
                
                # Track wins
                if i == winner:
                    self.wins[i] += 1
                
                # Track buildings
                settlements = len(player.settlements)
                cities = len(player.cities)
                roads = len(player.roads)
                dev_cards = len(player.dev_cards) + len(player.played_dev_cards)
                
                self.build_stats[i]['settlements'] += settlements
                self.build_stats[i]['cities'] += cities
                self.build_stats[i]['roads'] += roads
                self.build_stats[i]['dev_cards'] += dev_cards
                
                # Store player-specific stats
                player_stats[i] = {
                    'agent_type': type_name,
                    'victory_points': vp,
                    'is_winner': i == winner,
                    'buildings': {
                        'settlements': settlements,
                        'cities': cities,
                        'roads': roads,
                        'dev_cards': dev_cards
                    },
                    'action_distribution': action_counts[i]
                }
                
                # Store action types for this player
                self.action_types_per_game[i].append(action_counts[i])
            
            # Update game length
            self.game_lengths.append(moves)
            
            # Store game stats
            game_stats = {
                'game_id': game_idx,
                'winner': winner,
                'winner_vp': winner_vp,
                'moves': moves,
                'duration_seconds': game_duration,
                'players': player_stats
            }
            self.per_game_stats.append(game_stats)
            
            if self.debug:
                print(f"\nGame {game_idx+1} result:")
                for i in range(4):
                    agent_type = self.agent_types[i].name
                    vp = game.state.players[i].victory_points
                    win_mark = " (Winner)" if i == winner else ""
                    print(f"  Player {i} ({agent_type}): {vp} VP{win_mark}")
                print(f"  Moves: {moves}, Duration: {game_duration:.2f}s")
        
        # Calculate final statistics
        total_duration = time.time() - start_time
        avg_game_length = sum(self.game_lengths) / self.num_games
        
        # Print summary
        print("\n=== Evaluation Results ===")
        print(f"Games played: {self.num_games}")
        print(f"Total time: {total_duration:.2f}s ({self.num_games/total_duration:.2f} games/s)")
        print(f"Average game length: {avg_game_length:.2f} moves")
        
        print("\nPerformance by player position:")
        for i in range(4):
            agent_type = self.agent_types[i].name
            win_rate = self.wins[i] / self.num_games
            avg_vp = self.total_vps[i] / self.num_games
            
            # Determine which model (if AlphaZero)
            model_info = ""
            if self.agent_types[i] == AgentType.ALPHAZERO:
                if i == 0:
                    model_name = os.path.basename(self.model_path)
                else:
                    idx = i - 1
                    if idx < len(self.opponent_models):
                        model_name = os.path.basename(self.opponent_models[idx])
                    else:
                        model_name = os.path.basename(self.model_path)
                model_info = f" (model: {model_name})"
                
            print(f"Player {i} ({agent_type}{model_info}):")
            print(f"  Win rate: {win_rate:.2f} ({self.wins[i]}/{self.num_games})")
            print(f"  Average VP: {avg_vp:.2f}")
            
            # Print building stats
            if sum(self.build_stats[i].values()) > 0:
                print("  Building stats (avg per game):")
                for building, count in self.build_stats[i].items():
                    print(f"    {building}: {count/self.num_games:.2f}")
            
            # Print most common actions for AlphaZero agents
            if self.agent_types[i] == AgentType.ALPHAZERO:
                action_distribution = {}
                for game_actions in self.action_types_per_game[i]:
                    for action_type, count in game_actions.items():
                        action_distribution[action_type] = action_distribution.get(action_type, 0) + count
                
                if action_distribution:
                    # Sort by frequency
                    action_distribution = {k: v for k, v in sorted(
                        action_distribution.items(), key=lambda item: item[1], reverse=True
                    )}
                    
                    print("  Top 5 actions:")
                    total_actions = sum(action_distribution.values())
                    for j, (action_type, count) in enumerate(list(action_distribution.items())[:5]):
                        print(f"    {action_type}: {count} ({count/total_actions*100:.1f}%)")
        
        # Save results to file
        self.save_results()
        
        return {
            'wins': self.wins,
            'avg_vps': {i: self.total_vps[i]/self.num_games for i in range(4)},
            'avg_game_length': avg_game_length,
            'num_games': self.num_games,
            'per_game_stats': self.per_game_stats
        }
    
    def save_results(self):
        """Save evaluation results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create descriptive filename
        model_name = os.path.basename(self.model_path).replace('.pt', '')
        agent_str = ''.join([a.name[0] for a in self.agent_types])
        filename = f"{results_dir}/eval_{model_name}_{agent_str}_{timestamp}.json"
        
        # Prepare results
        results = {
            'primary_model': self.model_path,
            'agent_setup': [a.name for a in self.agent_types],
            'opponent_models': self.opponent_models,
            'placement_network': self.placement_network_path,
            'timestamp': timestamp,
            'num_games': self.num_games,
            'configuration': self.config,
            'deterministic': self.deterministic,
            'temperature': self.temperature,
            'wins': {i: self.wins[i] for i in range(4)},
            'win_rates': {i: self.wins[i]/self.num_games for i in range(4)},
            'avg_vps': {i: self.total_vps[i]/self.num_games for i in range(4)},
            'avg_game_length': sum(self.game_lengths) / self.num_games,
            'build_stats': {i: {k: v/self.num_games for k, v in stats.items()} 
                            for i, stats in self.build_stats.items()},
            'per_game_stats': self.per_game_stats
        }
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced AlphaZero Catan evaluation")
    
    # Basic parameters
    parser.add_argument("--model", type=str, default="models/best_model.pt",
                        help="Path to the model file to evaluate (default: models/best_model.pt)")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of evaluation games to play (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    

    parser.add_argument("--simulations", type=int, 
                        help="Override MCTS simulation count")
    parser.add_argument("--c-puct", type=float, 
                        help="Override exploration constant for MCTS")
    parser.add_argument("--batch-size", type=int, 
                        help="Override batch size for MCTS")
    
    # Agent setup
    parser.add_argument("--agents", type=str, default="A,R,R,R",
                        help="Agent types: A=AlphaZero, R=Random, H=Heuristic (default: A,R,R,R)")
    parser.add_argument("--opponents", type=str,
                        help="Comma-separated paths to opponent AlphaZero models")
    
    # MCTS behavior
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic MCTS (default: False)")
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="Temperature for MCTS action selection (default: 1.5)")
    
    # Placement network
    parser.add_argument("--placement-network", type=str,
                        help="Path to placement network model file")
    
    # Compare with training evaluator
    parser.add_argument("--compare", action="store_true",
                        help="Compare with training evaluator implementation")
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return
    
    # Parse opponent models if provided
    opponent_models = []
    if args.opponents:
        opponent_models = args.opponents.split(',')
        for model in opponent_models:
            if not os.path.exists(model):
                print(f"Warning: Opponent model file '{model}' not found")
    
    # Check placement network if provided
    if args.placement_network and not os.path.exists(args.placement_network):
        print(f"Warning: Placement network file '{args.placement_network}' not found")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        agent_spec=args.agents,
        opponent_models=opponent_models,
        placement_network_path=args.placement_network,
        num_games=args.games,
        debug=args.debug,
        deterministic=args.deterministic,
        temperature=args.temperature
    )
    
    # Override config values if provided
    if args.simulations:
        evaluator.config['num_simulations'] = args.simulations
    if args.c_puct:
        evaluator.config['c_puct'] = args.c_puct
    if args.batch_size:
        evaluator.config['mcts_batch_size'] = args.batch_size
    
    # Run evaluation
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
                agent = load_alphazero_agent(player_id=player_id, model_path=args.model)
                # Also set placement network if available
                if args.placement_network and os.path.exists(args.placement_network):
                    try:
                        from AlphaZero.core.initial_placement_network import InitialPlacementNetwork
                        checkpoint = torch.load(args.placement_network, map_location='cpu')
                        if 'network_state_dict' in checkpoint:
                            state_dict = checkpoint['network_state_dict']
                            input_dim = 260  # Default
                            hidden_dim = 128  # Default
                            if 'fc1.weight' in state_dict:
                                input_dim = state_dict['fc1.weight'].shape[1]
                                hidden_dim = state_dict['fc1.weight'].shape[0]
                            network = InitialPlacementNetwork(
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                output_dim=54
                            )
                            network.load_state_dict(state_dict)
                            network.eval()
                            agent.set_initial_placement_network(network)
                            print("Placement network loaded for comparison agent")
                    except Exception as e:
                        print(f"Failed to load placement network for comparison: {e}")
                return agent
            
            # Use same config as our evaluator
            config = evaluator.config
            
            # Create and run the training evaluator
            print(f"Running {args.games} games with training evaluator...")
            training_evaluator = Evaluator(create_game, create_agent, config)
            training_results = training_evaluator.evaluate(args.games)
            
            print("\nResults comparison:")
            print(f"Our evaluator:     Win rate = {results['win_rates'][0]:.2f}, Avg VP = {results['avg_vps'][0]:.2f}")
            print(f"Training evaluator: Win rate = {training_results['win_rate']:.2f}, Avg VP = {training_results['avg_vp']:.2f}")
            
        except Exception as e:
            print(f"Failed to run training evaluator comparison: {e}")

if __name__ == "__main__":
    main()
"""
Main entry point for AlphaZero Catan implementation.
"""
import os
import torch
import argparse
import json
import random
import numpy as np
from AlphaZero.training.self_play import TrainingPipeline
from AlphaZero.core.network import CatanNetwork, DeepCatanNetwork

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_config():
    """Load and return the configuration"""
    # Default configuration
    config = {
        # Network parameters
        'state_dim': 992,
        'action_dim': 200,
        'hidden_dim': 256,
        'use_deep_network': False,
        
        # Training parameters
        'learning_rate': 0.001,
        'num_iterations': 50,
        'self_play_games': 20,
        'eval_games': 10,
        'epochs': 10,
        'batch_size': 128,
        'buffer_size': 100000,
        
        # MCTS parameters
        'num_simulations': 100,
        'c_puct': 1.5,
        
        # Game parameters
        'max_moves': 200,
        
        # Paths
        'model_dir': 'models',
        'config_file': 'config.json',
    }
    
    # Check if config file exists
    if os.path.exists(config['config_file']):
        with open(config['config_file'], 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
    
    return config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train or run AlphaZero Catan')
    
    parser.add_argument('--mode', choices=['train', 'play', 'test'], default='train',
                        help='Mode to run in (train, play, or test)')
    
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of training iterations')
    
    parser.add_argument('--self-play-games', type=int, default=None,
                        help='Number of self-play games per iteration')
    
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to load a specific model')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
                        
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update config with command line arguments"""
    if args.iterations:
        config['num_iterations'] = args.iterations
    
    if args.self_play_games:
        config['self_play_games'] = args.self_play_games
    
    if args.config:
        with open(args.config, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
    
    return config

def train_mode(config):
    """Run in training mode"""
    print("=== AlphaZero Catan Training ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create training pipeline
    pipeline = TrainingPipeline(config)
    
    # Load model if specified
    if args.model_path and os.path.exists(args.model_path):
        pipeline.load_model(args.model_path)
    
    # Run training
    pipeline.train(config['num_iterations'])
    
    print("Training completed!")

def test_mode(config):
    """Run in test mode to verify components"""
    print("=== AlphaZero Catan Test Mode ===")
    
    # Import test components
    from AlphaZero.core.network import CatanNetwork
    from AlphaZero.model.state_encoder import StateEncoder
    from AlphaZero.model.action_mapper import ActionMapper
    from AlphaZero.core.mcts import MCTS, MCTSNode
    from AlphaZero.agent.alpha_agent import AlphaZeroAgent
    
    print("Creating components...")
    network = CatanNetwork(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim']
    )
    
    state_encoder = StateEncoder(max_actions=config['action_dim'])
    action_mapper = ActionMapper(max_actions=config['action_dim'])
    
    print("Testing network...")
    # Create a random input
    state = torch.rand(config['state_dim'])
    policy_logits, value = network(state.unsqueeze(0))
    print(f"Network output shapes - Policy: {policy_logits.shape}, Value: {value.shape}")
    
    print("Creating game...")
    from game.board import Board
    from game.game_logic import GameLogic
    from agent.base import AgentType
    
    board = Board()
    game = GameLogic(board, agent_types=[AgentType.RANDOM] * 4)
    
    # Handle the setup phase
    print("Setting up game...")
    while not game.is_setup_complete():
        game.process_ai_turn()
    
    print("Testing state encoder...")
    encoded_state = state_encoder.encode_state(game.state)
    print(f"Encoded state shape: {encoded_state.shape}")
    
    print("Testing action mapper...")
    valid_actions = list(game.state.possible_actions)
    if valid_actions:
        action = valid_actions[0]
        idx = action_mapper.action_to_index(action)
        decoded_action = action_mapper.index_to_action(idx)
        print(f"Action: {action.type}, Index: {idx}")
        print(f"Decoded action: {decoded_action.type}")
    
    print("Testing MCTS initialization...")
    mcts = MCTS(network, state_encoder, action_mapper, num_simulations=10)
    
    print("Testing MCTS node creation...")
    root = MCTSNode(game_state=game.state)
    print(f"Created root node: {root}")
    
    print("Tests completed!")

def play_mode(config):
    """Run in play mode (human vs AI)"""
    print("=== AlphaZero Catan Play Mode ===")
    print("This mode is not implemented yet.")
    # Future implementation for playing against the trained agent
    pass

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Get configuration
    config = get_config()
    config = update_config_from_args(config, args)
    
    # Run in the specified mode
    if args.mode == 'train':
        train_mode(config)
    elif args.mode == 'test':
        test_mode(config)
    elif args.mode == 'play':
        play_mode(config)